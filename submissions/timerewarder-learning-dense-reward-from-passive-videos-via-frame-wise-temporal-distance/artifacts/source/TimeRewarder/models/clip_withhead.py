"""Unified TimeRewarder progress model (CLIPwithHead).

Single definition shared by both stages:
  - training (training/main.py):  forward()          -> temporal-distance losses
  - RL inference (RL/):           predict_progress() -> (n,) bidirectional adjacent-frame reward

Both paths share one __init__ / state_dict layout, so a checkpoint trained via
forward() loads directly for predict_progress() at inference time.
"""
from typing import Tuple, Union
import torch
from torch import nn, einsum
from einops import rearrange, repeat
import warnings
import hashlib
import os
import urllib.request
from tqdm import tqdm

from models.clip_model import CLIP, LayerNorm, Transformer, MultiframeIntegrationTransformer
from models.discretesupport import (
    DiscreteSupport,
    order_head_regression_dim,
    slice_regression_logits,
    strip_legacy_order_head_state_dict,
)


class CLIPwithHead(CLIP):
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        T=8,
        mlp_droprate=0.2,
        mit_layers=1,
        use_cache=True,
        use_bin=True,
        bin_num=20,
        diff_representation=False,
        use_text=False,
        implicit_negative=True,
    ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )

        self.T = T
        self.use_cache = use_cache
        self.mlp_droprate = mlp_droprate
        self.use_bin = use_bin
        self.bin_num = bin_num
        self.diff_representation = diff_representation
        self.use_text = use_text
        self.implicit_negative = implicit_negative

        vision_heads = vision_width // 64
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.cache_text_features = {}

        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.order_attn = CrossAttention(query_dim=embed_dim, context_dim=vision_width, heads=8, dim_head=64, dropout=0.0)

        self.order_ln = LayerNorm(embed_dim)

        self.mit = MultiframeIntegrationTransformer(T=T, embed_dim=embed_dim, layers=mit_layers,)

        if self.diff_representation:
            mlp_input_dim = embed_dim
        else:
            mlp_input_dim = 2 * embed_dim

        self._regression_dim = order_head_regression_dim(bin_num, use_bin=self.use_bin)
        if not self.diff_representation:
            self.order_mlp = nn.Sequential(
                LayerNorm(mlp_input_dim),
                nn.Linear(mlp_input_dim, self._regression_dim),
                nn.Dropout(self.mlp_droprate),
            )
            if self.use_bin:
                self.discrete_support = DiscreteSupport(bins=bin_num)
        else:
            self.layer_norm = LayerNorm(embed_dim)
            self.order_linear = nn.Linear(embed_dim, self._regression_dim)
            self.discrete_support = DiscreteSupport(bins=bin_num)

        self.initialize_parameters()

    def _regression_logits(self, logits):
        return slice_regression_logits(logits, self._regression_dim)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def encode_text(self, text):
        x = self.token_embedding(text)
        eos_indx = text.argmax(dim=-1)
        K, N1, C = x.shape

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(K, -1)
        return x

    def _expand_text_for_video(self, text_feature, n, t):
        if not self.use_text:
            return None
        if text_feature is None:
            raise ValueError("text_feature is required when use_text=True")
        return text_feature.unsqueeze(0).expand(n * t, 1, -1)

    def encode_video_with_text(self, image, text, clean=False):
        b, t, c, h, w = image.size()
        image = image.reshape(-1, c, h, w)
        cls_features, img_features = self.visual(image)
        if clean or not self.use_text:
            return cls_features
        text_img_features = self.order_attn(text, context=img_features)
        text_img_features = text_img_features.reshape(b, t, text_img_features.shape[-2], text_img_features.shape[-1])
        text_img_features = text_img_features.permute(0, 2, 1, 3)
        text_img_features = text_img_features.reshape(-1, t, text_img_features.shape[-1])
        return self.mit(self.order_ln(text_img_features))

    def cache_text(self, text):
        self.eval()
        text_number = text.shape[0]
        with torch.no_grad():
            if text_number not in self.cache_text_features.keys():
                chunk_size = 2048
                text_chunks = text.split(chunk_size)
                text_features = []
                for chunk in text_chunks:
                    text_features.append(self.encode_text(chunk))
                self.cache_text_features[text_number] = torch.cat(text_features)
        self.train()
        return self.cache_text_features[text_number]

    # ------------------------------------------------------------------ #
    # Training: temporal-distance losses over clip pairs.                 #
    # DistributedDataParallel calls forward(), so this must be forward(). #
    # ------------------------------------------------------------------ #
    def forward(self, image, text=None, num_clips=4, progress=None, show_bin=False):
        if len(image.shape) < 6:
            raise ValueError(f'expected 6D image (B,num_clips,t,c,h,w), got {tuple(image.shape)}')
        b, num_clips, t, c, h, w = image.shape

        if self.use_text:
            if text is None:
                raise ValueError('text is required when use_text=True')
            if self.use_cache:
                text_features = self.cache_text(text)
            else:
                text_features = self.encode_text(text)
            if text_features.dim() == 2:
                text_features = text_features.unsqueeze(1)
            if text_features.shape[0] == 1:
                text_features = text_features.expand(b, -1, -1)
            elif text_features.shape[0] != b:
                raise ValueError(
                    f'text batch {text_features.shape[0]} != image batch {b}; '
                    'pass one text row per class or per sample'
                )
        else:
            text_features = None

        image = image.reshape(b * num_clips, t, c, h, w)
        mult = 2 if self.implicit_negative else 1
        if self.implicit_negative:
            image_reverse = torch.flip(image, [1])
            image_batch = torch.cat((image, image_reverse), dim=0)
        else:
            image_batch = image

        if self.use_text:
            _, n, f = text_features.shape
            text_exp = text_features.unsqueeze(1).expand(
                b, mult * num_clips * self.T, n, f
            ).reshape(mult * b * num_clips * self.T, n, f)
            video_features_batch = self.encode_video_with_text(image_batch, text_exp)
        else:
            video_features_batch = self.encode_video_with_text(image_batch, None)

        video_features = video_features_batch[:b * num_clips]
        feat_dim = video_features.shape[-1]
        if self.use_text:
            video_features = video_features.reshape(b, num_clips, n, feat_dim)
        else:
            video_features = video_features.reshape(b, num_clips, feat_dim)
        if self.implicit_negative:
            video_features_reverse = video_features_batch[b * num_clips:]
            if self.use_text:
                video_features_reverse = video_features_reverse.reshape(b, num_clips, n, feat_dim)
            else:
                video_features_reverse = video_features_reverse.reshape(b, num_clips, feat_dim)

        for i in range(num_clips):
            for j in range(i + 1, num_clips):
                if self.use_text:
                    video_features_1 = video_features[:, i, 0]
                    video_features_2 = video_features[:, j, 0]
                else:
                    video_features_1 = video_features[:, i]
                    video_features_2 = video_features[:, j]
                if self.implicit_negative:
                    if self.use_text:
                        video_features_inverse_1 = video_features_reverse[:, i, 0]
                        video_features_inverse_2 = video_features_reverse[:, j, 0]
                    else:
                        video_features_inverse_1 = video_features_reverse[:, i]
                        video_features_inverse_2 = video_features_reverse[:, j]
                progress_interval = progress[:, j] - progress[:, i]
                if i == 0 and j == 1:
                    video_features_1_batch = video_features_1
                    video_features_2_batch = video_features_2
                    progress_label = progress_interval
                    if self.implicit_negative:
                        video_features_inverse_1_batch = video_features_inverse_1
                        video_features_inverse_2_batch = video_features_inverse_2
                else:
                    video_features_1_batch = torch.cat((video_features_1_batch, video_features_1), dim=0)
                    video_features_2_batch = torch.cat((video_features_2_batch, video_features_2), dim=0)
                    progress_label = torch.cat((progress_label, progress_interval), dim=0)
                    if self.implicit_negative:
                        video_features_inverse_1_batch = torch.cat((video_features_inverse_1_batch, video_features_inverse_1), dim=0)
                        video_features_inverse_2_batch = torch.cat((video_features_inverse_2_batch, video_features_inverse_2), dim=0)

        if self.diff_representation:
            video_features_1_batch_normed = self.layer_norm(video_features_1_batch)
            video_features_2_batch_normed = self.layer_norm(video_features_2_batch)
            video_feature_batch_inorder = video_features_2_batch_normed - video_features_1_batch_normed
            if self.implicit_negative:
                video_features_inverse_1_batch_normed = self.layer_norm(video_features_inverse_1_batch)
                video_features_inverse_2_batch_normed = self.layer_norm(video_features_inverse_2_batch)
                video_feature_batch_inverse = video_features_inverse_2_batch_normed - video_features_inverse_1_batch_normed
        else:
            video_feature_batch_inorder = torch.cat((video_features_1_batch, video_features_2_batch), dim=-1)
            if self.implicit_negative:
                video_feature_batch_inverse = torch.cat((video_features_inverse_2_batch, video_features_inverse_1_batch), dim=-1)

        inorder_batch = video_feature_batch_inorder
        if self.implicit_negative:
            inverse_batch = video_feature_batch_inverse
            cat_features_all = torch.cat((inorder_batch, inverse_batch), dim=0)
        else:
            cat_features_all = inorder_batch
        if not self.diff_representation:
            cat_logits_all = self.order_mlp(cat_features_all)
        else:
            cat_logits_all = self.order_linear(cat_features_all)
        logits_order = cat_logits_all
        regression_logits = self._regression_logits(logits_order)
        if self.implicit_negative:
            regression_labels = torch.cat((progress_label, -progress_label), dim=0).to(regression_logits.device)
        else:
            regression_labels = progress_label.to(regression_logits.device)
        if self.use_bin:
            regression_labels_bin = self.discrete_support.scalar_to_vector(regression_labels)
            regression_loss = nn.CrossEntropyLoss()(regression_logits, regression_labels_bin)
        else:
            regression_loss = nn.MSELoss()(regression_logits.squeeze(-1), regression_labels)
        combined_loss = regression_loss

        if not show_bin:
            return combined_loss, regression_loss, logits_order

        with torch.no_grad():
            intervals = [
                (-1.0, -0.5), (-0.5, -0.25), (-0.25, -0.125), (-0.125, -0.0625),
                (-0.0625, 0), (0, 0.0625), (0.0625, 0.125), (0.125, 0.25),
                (0.25, 0.5), (0.5, 1.0),
            ]
            bin_metrics = {}
            r_logits = regression_logits.detach()
            if self.use_bin:
                r_logits = self.discrete_support.vector_to_scalar(r_logits)
            r_labels = regression_labels.detach()
            for low, high in intervals:
                mask = (r_labels >= low) & (r_labels < high)
                if mask.any():
                    selected_logits = r_logits[mask]
                    selected_labels = r_labels[mask]
                    opposite_sign_mask = (selected_logits * selected_labels) < 0
                    interval_str = f'[{low},{high}]'
                    bin_metrics[f'bin_{interval_str}_sign_error'] = opposite_sign_mask.float().mean().item()
                    bin_metrics[f'bin_{interval_str}_mae'] = torch.mean(torch.abs(selected_logits - selected_labels)).item()

        return combined_loss, regression_loss, logits_order, bin_metrics

    # ------------------------------------------------------------------ #
    # Inference: per-frame progress used as the RL reward.                #
    # ------------------------------------------------------------------ #
    def predict_progress(self, image, text_feature):
        if len(image.shape) == 4:
            if self.T == 3:
                image_1 = torch.roll(image, 1, 0)
                image_1[0] = image[0]
                image_2 = torch.roll(image_1, 1, 0)
                image_2[0] = image_1[0]
                image = torch.cat((image_2.unsqueeze(1), image_1.unsqueeze(1), image.unsqueeze(1)), dim=1)
            else:
                image = image.unsqueeze(1)
        n, t, c, h, w = image.shape
        text_exp = self._expand_text_for_video(text_feature, n, t)
        video_features = self.encode_video_with_text(image, text_exp)
        video_features = video_features.reshape(n, -1)
        # Adjacent-frame pairs (prev -> curr), bidirectional reward:
        # forward-pair progress minus reverse-pair progress. The order head is
        # antisymmetric in the ideal model (reverse = -forward), so this doubles the
        # signal while cancelling any direction-independent bias (TimeRewarder paper setting).
        prev_features = torch.roll(video_features, 1, 0)
        prev_features[0] = video_features[0]
        if self.diff_representation:
            prev_normed = self.layer_norm(prev_features)
            curr_normed = self.layer_norm(video_features)
            forward_features = curr_normed - prev_normed
            reverse_features = prev_normed - curr_normed
            all_features = torch.cat((forward_features, reverse_features), dim=0)
            logits = self._regression_logits(self.order_linear(all_features))
        else:
            forward_features = torch.cat((prev_features, video_features), dim=-1)
            reverse_features = torch.cat((video_features, prev_features), dim=-1)
            all_features = torch.cat((forward_features, reverse_features), dim=0)
            logits = self._regression_logits(self.order_mlp(all_features))
        scores = self.discrete_support.vector_to_scalar(logits) if self.use_bin else logits
        reward = scores[:n] - scores[n:]
        return reward.reshape(n)


def build_model(state_dict: dict, T=8, logger=None, use_cache=True, mit_layers=4, mlp_droprate=0.2, lora=False,
                no_pretrain=False, bin_num=20, use_bin=True, diff_representation=False,
                use_text=False, implicit_negative=True):
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']

    # When loading a trained TimeRewarder checkpoint, infer the order-head type
    # (regression vs. K-bin, and concat vs. diff_representation) directly from the
    # saved weights so the head always matches the checkpoint, regardless of the
    # bin_num/use_bin/diff_representation passed in. Base CLIP weights have no order
    # head, so training-from-pretrained falls through to the supplied config.
    if 'order_linear.weight' in state_dict:
        diff_representation, _order_head_w = True, state_dict['order_linear.weight']
    elif 'order_mlp.1.weight' in state_dict:
        diff_representation, _order_head_w = False, state_dict['order_mlp.1.weight']
    else:
        _order_head_w = None
    if _order_head_w is not None:
        use_bin = _order_head_w.shape[0] != 1
        if use_bin:
            bin_num = _order_head_w.shape[0]

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    elif lora:
        vision_width = state_dict['visual.base_model.model.conv1.weight'].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.base_model.model.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.base_model.model.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)

        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIPwithHead(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        T=T, mlp_droprate=mlp_droprate, mit_layers=mit_layers,
        use_cache=use_cache,
        bin_num=bin_num, use_bin=use_bin,
        diff_representation=diff_representation,
        use_text=use_text, implicit_negative=implicit_negative,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    reg_dim = order_head_regression_dim(bin_num, use_bin=use_bin)
    strip_legacy_order_head_state_dict(state_dict, reg_dim)

    if lora:
        import peft
        lora_config = peft.LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=['in_proj_weight', "in_proj_bias", 'out_proj'],
        )
        model.visual = peft.get_peft_model(model.visual, lora_config)
        print('LORA Trainable params:')
        model.visual.print_trainable_parameters()

    if not no_pretrain:
        msg = model.load_state_dict(state_dict, strict=False)
        if logger is not None:
            logger.info(f"load pretrained CLIP: {msg}")
        else:
            print(f"load pretrained CLIP: {msg}")
    else:
        print('Training from scratch!')

    return model.eval()


_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not match")
    return download_target


def load(model_path, name: str = None, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit=False, T=3, logger=None, use_cache=False,
         mit_layers=1, mlp_droprate=0.2, bin_num=-1, use_bin=None,
         diff_representation=False, use_text=False, implicit_negative=True):
    if use_bin is None:
        use_bin = bin_num > 0
    if model_path is None:
        # Training-from-pretrained: fetch the base CLIP weights by ARCH name.
        model_path = _download(_MODELS[name])
    try:
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    use_lora = model_path is not None and 'lora' in model_path

    model = build_model(
        state_dict or model.state_dict(), T=T, logger=logger,
        use_cache=use_cache, mlp_droprate=mlp_droprate, mit_layers=mit_layers,
        lora=use_lora, bin_num=bin_num, use_bin=use_bin,
        diff_representation=diff_representation, use_text=use_text,
        implicit_negative=implicit_negative,
    )

    if str(device) == "cpu":
        model.float()
    return model, model.state_dict()


class CrossAttention(nn.Module):
    def __init__(self, query_dim=512, context_dim=768, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.config = None

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x_cls = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x_cls = x_cls @ self.proj
        return x_cls, x[:, 1:, :]
