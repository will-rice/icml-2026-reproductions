#!/usr/bin/env python3
"""
GPCR Agonist Classifier - TR2-D2 Inference Script
"""

import argparse
import logging
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from models.roformer import Roformer

logger = logging.getLogger(__name__)


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested is None or str(requested).lower() == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda:0")
        return torch.device("cpu")

    try:
        device = torch.device(requested)
    except Exception as exc:
        logger.warning("Invalid device '%s': %s. Falling back to CPU.", requested, exc)
        return torch.device("cpu")
    if device.type != "cuda":
        return device

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        logger.warning("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")

    index = device.index if device.index is not None else 0
    count = torch.cuda.device_count()
    if index is None or index < 0 or index >= count:
        logger.warning(
            "CUDA device %s requested but only %d visible; using cuda:0",
            index,
            count
        )
        return torch.device("cuda:0")

    return torch.device(f"cuda:{index}")

# -------------------------
# Peptide to SMILES
# -------------------------
def peptide_to_smiles(seq: str) -> str:
    from rdkit import Chem
    seq = seq.strip().upper()
    mol = Chem.MolFromSequence(seq)
    if mol is None:
        raise ValueError(f"RDKit failed to convert peptide '{seq}' to SMILES")
    return Chem.MolToSmiles(mol)

# -------------------------
# Self-Attention Block
# -------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm(x + self.dropout(attn_out))
        return x

# -------------------------
# Cross-Attention Module
# -------------------------
class BiMultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.prot_to_lig = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.lig_to_prot = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.prot_ln = nn.LayerNorm(d_model)
        self.lig_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prot_h, lig_h, prot_kpm=None, lig_kpm=None):
        prot_ctx, _ = self.prot_to_lig(prot_h, lig_h, lig_h, key_padding_mask=lig_kpm)
        prot_h_out = self.prot_ln(prot_h + self.dropout(prot_ctx))

        lig_ctx, _ = self.lig_to_prot(lig_h, prot_h, prot_h, key_padding_mask=prot_kpm)
        lig_h_out = self.lig_ln(lig_h + self.dropout(lig_ctx))

        return prot_h_out, lig_h_out

# -------------------------
# TR2-D2 Encoder Wrapper
# -------------------------
class TR2D2RoFormerEncoder(nn.Module):
    def __init__(self, config, tokenizer, checkpoint_path=None, device="cpu"):
        super().__init__()
        self.device = device
        self.encoder = Roformer(config, tokenizer, device=device)

        if checkpoint_path:
            print(f"  Loading TR2-D2 checkpoint...")
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            # Unwrap common checkpoint containers (raw / "state_dict" / "model_state_dict").
            state_dict = ckpt
            for _key in ("state_dict", "model_state_dict"):
                if isinstance(state_dict, dict) and _key in state_dict:
                    state_dict = state_dict[_key]
                    break
            roformer_state = {
                k.replace("model.", "").replace("backbone.", ""): v
                for k, v in state_dict.items()
                if "roformer" in k or "encoder" in k or "backbone" in k
            }
            if not roformer_state:
                raise RuntimeError(
                    f"No RoFormer/encoder keys matched in TR2-D2 checkpoint {checkpoint_path} "
                    f"(top-level keys: {list(state_dict.keys())[:8]}). The ligand encoder would be "
                    f"left randomly initialized, producing meaningless direction scores."
                )
            incompatible = self.encoder.model.load_state_dict(roformer_state, strict=False)
            if incompatible.missing_keys:
                print(f"  [warn] TR2-D2 encoder: {len(incompatible.missing_keys)} keys not in "
                      f"checkpoint (e.g. {incompatible.missing_keys[:3]})")
            print(f"  TR2-D2 checkpoint loaded ({len(roformer_state)} tensors)")

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def forward(self, input_ids, attention_mask, inputs_embeds=None):
        if attention_mask is None:
            raise ValueError("attention_mask is required for ligand encoding.")
        attention_mask = attention_mask.to(self.device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self.device)
            out = self.encoder.model.roformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
        else:
            input_ids = input_ids.to(self.device)
            with torch.no_grad():
                out = self.encoder.model.roformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        return out.last_hidden_state

# -------------------------
# Full GPCR Model
# -------------------------
class ESM_TR2D2_GPCRClassifier(nn.Module):
    """
    GPCR Agonist Classifier with TR2-D2

    Architecture:
    1. ESM2 (protein) + TR2-D2 RoFormer (ligand)
    2. Projections to common dimension
    3. Self-Attention (1 layer each)
    4. BiDirectional Cross-Attention (2 stacked layers)
    5. Masked Average Pooling
    6. MLP Classifier
    """
    def __init__(
        self,
        esm_name,
        tr2d2_config,
        lig_tokenizer,
        tr2d2_checkpoint=None,
        d_model=256,
        n_heads=4,
        n_self_attn_layers=1,
        n_bmca_layers=2,
        dropout=0.3,
        device="cuda",
        esm_cache_dir=None,
        esm_local_files_only=False
    ):
        super().__init__()
        self.device = device

        # Frozen encoders
        print("Loading ESM2 protein encoder...")
        self.esm = EsmModel.from_pretrained(
            esm_name,
            cache_dir=esm_cache_dir,
            local_files_only=esm_local_files_only
        )
        for p in self.esm.parameters():
            p.requires_grad = False
        self.esm.eval()

        print("Loading TR2-D2 ligand encoder...")
        self.ligand_encoder = TR2D2RoFormerEncoder(
            tr2d2_config, lig_tokenizer, tr2d2_checkpoint, device
        )

        esm_dim = self.esm.config.hidden_size
        lig_dim = tr2d2_config.roformer.hidden_size

        self.prot_proj = nn.Linear(esm_dim, d_model)
        self.lig_proj = nn.Linear(lig_dim, d_model)

        # Self-attention
        self.prot_self_attn_layers = nn.ModuleList([
            SelfAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_self_attn_layers)
        ])
        self.lig_self_attn_layers = nn.ModuleList([
            SelfAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_self_attn_layers)
        ])

        # Cross-attention
        self.bmca_layers = nn.ModuleList([
            BiMultiHeadCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_bmca_layers)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )

    def forward(self, prot_tokens, lig_tokens, lig_inputs_embeds=None):
        prot_kpm = prot_tokens["attention_mask"].eq(0)
        lig_kpm = lig_tokens["attention_mask"].eq(0)

        with torch.no_grad():
            prot_out = self.esm(**prot_tokens).last_hidden_state

        lig_out = self.ligand_encoder(
            lig_tokens["input_ids"],
            lig_tokens["attention_mask"],
            inputs_embeds=lig_inputs_embeds
        )

        prot_h = self.prot_proj(prot_out)
        lig_h = self.lig_proj(lig_out)

        # Self-attention
        for self_attn in self.prot_self_attn_layers:
            prot_h = self_attn(prot_h, key_padding_mask=prot_kpm)
        for self_attn in self.lig_self_attn_layers:
            lig_h = self_attn(lig_h, key_padding_mask=lig_kpm)

        # Cross-attention (2 stacked)
        for bmca in self.bmca_layers:
            prot_h, lig_h = bmca(prot_h, lig_h, prot_kpm=prot_kpm, lig_kpm=lig_kpm)

        # Masked average pooling
        prot_mask = prot_tokens["attention_mask"].unsqueeze(-1)
        lig_mask = lig_tokens["attention_mask"].unsqueeze(-1)

        prot_repr = (prot_h * prot_mask).sum(dim=1) / prot_mask.sum(dim=1).clamp(min=1)
        lig_repr = (lig_h * lig_mask).sum(dim=1) / lig_mask.sum(dim=1).clamp(min=1)

        return self.classifier(torch.cat([prot_repr, lig_repr], dim=-1))

# -------------------------
# Tokenization
# -------------------------
def create_tr2d2_config(vocab_size):
    return SimpleNamespace(
        roformer=SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=768,
            n_layers=8,
            n_heads=8,
            max_position_embeddings=1035
        )
    )


def _load_state_dict_flexible(model: nn.Module, state_dict: Dict, strict: bool = True) -> None:
    try:
        model.load_state_dict(state_dict, strict=strict)
        return
    except RuntimeError as exc:
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        logger.warning("Strict load failed: %s", exc)
        logger.warning(
            "Retrying with filtered keys (%d/%d) and strict=False",
            len(filtered),
            len(state_dict)
        )
        incompatible = model.load_state_dict(filtered, strict=False)
        # ESM keys are expected to be absent (ESM2 is re-loaded pretrained in __init__); any
        # missing NON-esm key is a trained head left at random init — surface it loudly.
        trained_missing = [k for k in incompatible.missing_keys if not k.startswith("esm")]
        if trained_missing:
            logger.warning(
                "%d TRAINED (non-ESM) oracle keys did not load and remain randomly initialized: %s",
                len(trained_missing), trained_missing[:10]
            )
        if incompatible.missing_keys:
            logger.warning("Missing keys (first 10): %s", incompatible.missing_keys[:10])
        if incompatible.unexpected_keys:
            logger.warning("Unexpected keys (first 10): %s", incompatible.unexpected_keys[:10])

def tokenize_protein(seq, tokenizer, device):
    out = tokenizer(
        seq,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
        add_special_tokens=True
    )
    return {k: v.to(device) for k, v in out.items()}

def tokenize_ligand(smiles, tokenizer, max_len, device):
    enc = tokenizer(
        smiles,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=True
    )
    ids = enc["input_ids"].squeeze(0)
    att = enc["attention_mask"].squeeze(0)

    pad = max_len - ids.numel()
    if pad > 0:
        ids = torch.cat([ids, torch.full((pad,), tokenizer.pad_token_id)])
        att = torch.cat([att, torch.zeros(pad)])

    return {
        "input_ids": ids.unsqueeze(0).to(device),
        "attention_mask": att.unsqueeze(0).to(device)
    }

# -------------------------
# Training-Compatible Oracle Wrapper
# -------------------------
class DirectionalOracle(nn.Module):
    """
    Batch-capable oracle wrapper with TD3B-compatible predict_with_confidence().

    This class is intended for training integration where peptide/protein tokens
    are provided directly (batched) and the oracle runs in inference-only mode.
    """
    def __init__(
        self,
        model_ckpt: str,
        tr2d2_checkpoint: str,
        tokenizer_vocab: str,
        tokenizer_splits: str,
        esm_name: str = "facebook/esm2_t33_650M_UR50D",
        d_model: int = 256,
        n_heads: int = 4,
        n_self_attn_layers: int = 1,
        n_bmca_layers: int = 2,
        dropout: float = 0.3,
        max_ligand_length: int = 768,
        max_protein_length: int = 1024,
        device: Optional[str] = None,
        esm_cache_dir: Optional[str] = None,
        esm_local_files_only: bool = False
    ):
        super().__init__()

        if isinstance(device, torch.device):
            device = str(device)
        self.device = resolve_device(device)

        self.max_ligand_length = max_ligand_length
        self.max_protein_length = max_protein_length
        self._warned_ligand_truncation = False
        self._warned_protein_truncation = False

        self.lig_tokenizer = SMILES_SPE_Tokenizer(tokenizer_vocab, tokenizer_splits)
        self.prot_tokenizer = EsmTokenizer.from_pretrained(
            esm_name,
            cache_dir=esm_cache_dir,
            local_files_only=esm_local_files_only
        )

        tr2d2_cfg = create_tr2d2_config(self.lig_tokenizer.vocab_size)
        self.model = ESM_TR2D2_GPCRClassifier(
            esm_name=esm_name,
            tr2d2_config=tr2d2_cfg,
            lig_tokenizer=self.lig_tokenizer,
            tr2d2_checkpoint=tr2d2_checkpoint,
            d_model=d_model,
            n_heads=n_heads,
            n_self_attn_layers=n_self_attn_layers,
            n_bmca_layers=n_bmca_layers,
            dropout=dropout,
            device=self.device,
            esm_cache_dir=esm_cache_dir,
            esm_local_files_only=esm_local_files_only
        )

        state_dict = torch.load(model_ckpt, map_location=self.device, weights_only=False)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        _load_state_dict_flexible(self.model, state_dict, strict=True)
        self.model.to(self.device).eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self._lig_pad_token_id = self.lig_tokenizer.pad_token_id
        if self._lig_pad_token_id is None:
            self._lig_pad_token_id = 0
        self._prot_pad_token_id = self.prot_tokenizer.pad_token_id
        if self._prot_pad_token_id is None:
            self._prot_pad_token_id = 0

    def encode_protein(self, protein_seq: str) -> torch.Tensor:
        tokens = self.prot_tokenizer(
            protein_seq,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_protein_length,
            add_special_tokens=True
        )
        return tokens["input_ids"].to(self.device)

    def _normalize_token_dict(
        self,
        tokens: torch.Tensor,
        pad_token_id: int,
        max_length: int,
        warned_attr: str
    ) -> Dict[str, torch.Tensor]:
        if isinstance(tokens, dict):
            input_ids = tokens.get("input_ids")
            if input_ids is None:
                raise ValueError("Token dict must include input_ids.")
            attention_mask = tokens.get("attention_mask")
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = (input_ids != pad_token_id).long()
            else:
                attention_mask = attention_mask.to(self.device)
        else:
            input_ids = tokens
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(self.device)
            attention_mask = (input_ids != pad_token_id).long()

        if max_length is not None and input_ids.size(1) > max_length:
            if not getattr(self, warned_attr):
                logger.warning(
                    "Truncating input from length %d to max_length=%d",
                    input_ids.size(1),
                    max_length
                )
                setattr(self, warned_attr, True)
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _normalize_prob_inputs(
        self,
        probs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_length: int,
        warned_attr: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if probs.dim() == 2:
            probs = probs.unsqueeze(0)
        probs = probs.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones(
                probs.size(0), probs.size(1), device=self.device, dtype=torch.long
            )
        else:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.to(self.device).long()

        if max_length is not None and probs.size(1) > max_length:
            if not getattr(self, warned_attr):
                logger.warning(
                    "Truncating input from length %d to max_length=%d",
                    probs.size(1),
                    max_length
                )
                setattr(self, warned_attr, True)
            probs = probs[:, :max_length]
            attention_mask = attention_mask[:, :max_length]

        return probs, attention_mask

    @torch.no_grad()
    def predict_with_confidence(
        self,
        peptide_tokens: torch.Tensor,
        protein_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lig_tokens = self._normalize_token_dict(
            peptide_tokens,
            self._lig_pad_token_id,
            self.max_ligand_length,
            "_warned_ligand_truncation"
        )
        prot_tokens = self._normalize_token_dict(
            protein_tokens,
            self._prot_pad_token_id,
            self.max_protein_length,
            "_warned_protein_truncation"
        )

        lig_batch = lig_tokens["input_ids"].size(0)
        prot_batch = prot_tokens["input_ids"].size(0)
        if prot_batch == 1 and lig_batch > 1:
            prot_tokens = {k: v.expand(lig_batch, -1) for k, v in prot_tokens.items()}
        elif prot_batch != lig_batch:
            raise ValueError(
                f"Batch size mismatch: peptide_tokens={lig_batch}, protein_tokens={prot_batch}"
            )

        logits = self.model(prot_tokens, lig_tokens)
        probs = F.softmax(logits, dim=-1)
        p_agonist = probs[:, 1]
        confidence = torch.max(probs, dim=-1).values
        return p_agonist, confidence

    def predict_from_probs(
        self,
        ligand_probs: torch.Tensor,
        protein_tokens: torch.Tensor,
        ligand_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        lig_probs, lig_attention = self._normalize_prob_inputs(
            ligand_probs,
            ligand_attention_mask,
            self.max_ligand_length,
            "_warned_ligand_truncation",
        )
        prot_tokens = self._normalize_token_dict(
            protein_tokens,
            self._prot_pad_token_id,
            self.max_protein_length,
            "_warned_protein_truncation"
        )

        lig_batch = lig_probs.size(0)
        prot_batch = prot_tokens["input_ids"].size(0)
        if prot_batch == 1 and lig_batch > 1:
            prot_tokens = {k: v.expand(lig_batch, -1) for k, v in prot_tokens.items()}
        elif prot_batch != lig_batch:
            raise ValueError(
                f"Batch size mismatch: ligand_probs={lig_batch}, protein_tokens={prot_batch}"
            )

        emb_weight = self.model.ligand_encoder.encoder.model.roformer.embeddings.word_embeddings.weight
        if lig_probs.size(-1) != emb_weight.size(0):
            raise ValueError(
                f"Ligand vocab mismatch: probs={lig_probs.size(-1)} vs oracle={emb_weight.size(0)}"
            )
        lig_inputs_embeds = lig_probs @ emb_weight
        lig_input_ids = torch.zeros(
            lig_probs.size(0), lig_probs.size(1), device=lig_probs.device, dtype=torch.long
        )
        lig_tokens = {"input_ids": lig_input_ids, "attention_mask": lig_attention}
        logits = self.model(prot_tokens, lig_tokens, lig_inputs_embeds=lig_inputs_embeds)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]

# -------------------------
# Prediction
# -------------------------
@torch.no_grad()
def predict(model, prot_tok, lig_tok, protein_seq, peptide_seq, device, threshold=0.5):
    """
    Predict agonist activity

    Returns:
        dict with keys: smiles, non_agonist_prob, agonist_prob, prediction, confidence
    """
    # Convert peptide to SMILES
    smiles = peptide_to_smiles(peptide_seq)

    # Tokenize
    prot_tokens = tokenize_protein(protein_seq, prot_tok, device)
    lig_tokens = tokenize_ligand(smiles, lig_tok, 768, device)  # FIXED: 768 not 256!

    # Predict
    logits = model(prot_tokens, lig_tokens)
    probs = F.softmax(logits, dim=-1).squeeze(0)

    p_non_agonist = probs[0].item()
    p_agonist = probs[1].item()
    prediction = "agonist" if p_agonist >= threshold else "non-agonist"

    return {
        "smiles": smiles,
        "non_agonist_prob": p_non_agonist,
        "agonist_prob": p_agonist,
        "prediction": prediction,
        "confidence": max(p_non_agonist, p_agonist)
    }

# -------------------------
# MAIN
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GPCR Agonist Classifier - TR2-D2 Inference"
    )
    parser.add_argument("--model_ckpt", required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--tr2d2_checkpoint", required=True,
                       help="Path to TR2-D2 pretrained checkpoint")
    parser.add_argument("--tokenizer_vocab", required=True,
                       help="Path to tokenizer vocabulary")
    parser.add_argument("--tokenizer_splits", required=True,
                       help="Path to tokenizer splits")
    parser.add_argument("--protein_seq", required=True,
                       help="GPCR protein sequence")
    parser.add_argument("--ligand_peptide", required=True,
                       help="Ligand peptide sequence")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Classification threshold (default: 0.5)")
    parser.add_argument("--d_model", type=int, default=256,
                       help="Hidden dimension (must match training)")
    parser.add_argument("--n_heads", type=int, default=4,
                       help="Number of attention heads (must match training)")
    parser.add_argument("--n_self_attn_layers", type=int, default=1,
                       help="Number of self-attention layers (must match training)")
    parser.add_argument("--n_bmca_layers", type=int, default=2,
                       help="Number of cross-attention layers (must match training)")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate (must match training)")
    parser.add_argument("--device", default=None,
                       help="Device (cuda/cpu, default: auto)")
    parser.add_argument("--esm_name", default="facebook/esm2_t33_650M_UR50D",
                       help="ESM model name or local path")
    parser.add_argument("--esm_cache_dir", default=None,
                       help="Optional cache directory for ESM model")
    parser.add_argument("--esm_local_files_only", action="store_true",
                       help="Load ESM from local cache only (no network)")

    args = parser.parse_args()

    # Device
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print("")

    # Load tokenizers
    print("Loading tokenizers...")
    prot_tok = EsmTokenizer.from_pretrained(
        args.esm_name,
        cache_dir=args.esm_cache_dir,
        local_files_only=args.esm_local_files_only
    )
    lig_tok = SMILES_SPE_Tokenizer(args.tokenizer_vocab, args.tokenizer_splits)
    print(f"  Vocab size: {lig_tok.vocab_size}")
    print("")

    # Create config
    tr2d2_cfg = create_tr2d2_config(lig_tok.vocab_size)

    # Load model
    print("Loading model...")
    model = ESM_TR2D2_GPCRClassifier(
        esm_name=args.esm_name,
        tr2d2_config=tr2d2_cfg,
        lig_tokenizer=lig_tok,
        tr2d2_checkpoint=args.tr2d2_checkpoint,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_self_attn_layers=args.n_self_attn_layers,
        n_bmca_layers=args.n_bmca_layers,
        dropout=args.dropout,
        device=device,
        esm_cache_dir=args.esm_cache_dir,
        esm_local_files_only=args.esm_local_files_only
    )

    # Load trained weights
    print("  Loading trained weights...")
    state_dict = torch.load(args.model_ckpt, map_location=device)
    _load_state_dict_flexible(model, state_dict, strict=True)
    model.to(device).eval()
    print("  Model ready.")
    print("")

    # Predict
    print("Running inference...")
    result = predict(
        model, prot_tok, lig_tok,
        args.protein_seq, args.ligand_peptide,
        device, args.threshold
    )

    # Display results
    print("")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Protein:  {args.protein_seq[:50]}{'...' if len(args.protein_seq) > 50 else ''}")
    print(f"Ligand:   {args.ligand_peptide}")
    print(f"SMILES:   {result['smiles']}")
    print("")
    print(f"Non-agonist probability: {result['non_agonist_prob']:.4f}")
    print(f"Agonist probability:     {result['agonist_prob']:.4f}")
    print("")
    print(f"Prediction (threshold={args.threshold}): {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
