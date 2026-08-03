"""Statically audit reward construction in verified inert upstream sources."""

import ast
import hashlib
from pathlib import Path

from timerewarder_repro.acquisition import verify_acquisition


REQUIRED_SOURCES = (
    "training/datasets/build.py",
    "models/clip_withhead.py",
    "models/discretesupport.py",
    "RL/agent/rl_agent.py",
    "RL/train.py",
    "RL/replay_buffer.py",
    "RL/utils.py",
    "LICENSE",
)

APPROVED_SPAN_SHA256 = {
    ("training/datasets/build.py", "BaseDataset", "_distance_probs"): (
        "3ae0c165ddc6489d75afa53fda93eb9a18220d6501d1da8050c8adaee080b850"
    ),
    ("training/datasets/build.py", "BaseDataset", "_progress_from_sample_ids"): (
        "99d8e5438618c6452f4de9ac72e5bae2893ef3421faac93390c7c21f1c2c985b"
    ),
    ("training/datasets/build.py", "BaseDataset", "prepare_train_frames"): (
        "20c29dbc5491863bfd17e3e7ce1fdc79ad2c14b74a52f7b8b87c8ecf3c069c61"
    ),
    ("training/datasets/build.py", "BaseDataset", "prepare_test_frames"): (
        "aecb9ea0021ec8d3bf48bbe607ccc9bdba855cced546cb57f74b8973cb1197e0"
    ),
    ("training/datasets/build.py", "VideoDataset", "load_annotations"): (
        "6dc0d6bb0285eb3c4f3ab4011d92b944e02ddd3cdb419bdfda23603ad748282f"
    ),
    ("models/clip_withhead.py", "CLIPwithHead", "_regression_logits"): (
        "dda7208dd4b8e3b0c61d3682b9d423fc2fe1a85498ced2351ecb4efeb58a6dae"
    ),
    ("models/clip_withhead.py", "CLIPwithHead", "predict_progress"): (
        "8282824a1ed4d07f0c705578c177f1e7085ddbcf0cc90f68764aaccd6b6dd0d1"
    ),
    ("models/discretesupport.py", None, "transform_one"): (
        "703735c1c8259be5002679a90e8d57214aaffc5411c82b276e8a87d1bc1b7890"
    ),
    ("models/discretesupport.py", "DiscreteSupport", "vector_to_scalar"): (
        "f3ca4a69eb00b7e706d93edbdf89061efaa05550109897851e0ed8767057ed4a"
    ),
    ("models/discretesupport.py", None, "slice_regression_logits"): (
        "c6d7a4a882dea513c69b04941f1ff19098d295ae33f715463a3b742f7a6a1edc"
    ),
    ("RL/agent/rl_agent.py", "RLAgent", "update_critic"): (
        "7b586f8f093ca255ac28740a4a2800ce7c4117a8b460f567b57aa6aa6e78a361"
    ),
    ("RL/agent/rl_agent.py", "RLAgent", "update"): (
        "c14e9af72274eb3c65e346db9ba36b0b86e571254382d04e0363f404a327d8fa"
    ),
    ("RL/agent/rl_agent.py", "RLAgent", "clip_rewarder"): (
        "01cada854558849edc8b3abd78c94109286930fac726810c359d449942ad1887"
    ),
    ("RL/train.py", "WorkspaceIL", "_log_timerewarder_reward_curves"): (
        "bcd54b9038b7a325a5119faac71a9b3bc6142216dcc33dc9c1b73fa965b597ef"
    ),
    ("RL/train.py", "WorkspaceIL", "setup"): (
        "8bab1376bb7aa966f37e8ae189d07492c6353e4f197b74157ae240cea8eaba34"
    ),
    ("RL/train.py", "WorkspaceIL", "replay_iter"): (
        "092696c857b9fc5ae0be75facad2c0fccc6313ea5d5304580c46cca41b7fc9fd"
    ),
    ("RL/train.py", "WorkspaceIL", "train_il"): (
        "a73def725fd3e47ef0643b94b1beeb45d40af7c0d2be67a04d13352eda0cfbae"
    ),
    ("RL/replay_buffer.py", None, "episode_len"): (
        "cf3ce2a7624d3a8d55ea029f6f720ad84b57be9d340c2ab11e4cb1a24fb0b2de"
    ),
    ("RL/replay_buffer.py", None, "save_episode"): (
        "d5528f33cb83237368ef33bb10664b93a6a1e0e0c4c99428e7576bd483a391ed"
    ),
    ("RL/replay_buffer.py", None, "load_episode"): (
        "fe7639aeecc9b1058a1767617f2a41ff6fa3634fac8bb76c82b2e5142bc1ea78"
    ),
    ("RL/replay_buffer.py", "ReplayBufferStorage", "__init__"): (
        "3887a4aab69dc8a89828ca8214ed754c3915dc440ee2c9a256d863a36f0a0005"
    ),
    ("RL/replay_buffer.py", "ReplayBufferStorage", "add"): (
        "905abf0eeaee879600c1d95df4b1c40579ddb148e942b8fc71f917375b19941b"
    ),
    ("RL/replay_buffer.py", "ReplayBufferStorage", "_preload"): (
        "534dedc7e718663b5b3e0e8a4d3b9392f1a26f7de68286b014cb5092bdebd826"
    ),
    ("RL/replay_buffer.py", "ReplayBufferStorage", "_store_episode"): (
        "1acabd4a1df6a048495abf644ba7920230ee77ea07175df9333170861b04d07b"
    ),
    ("RL/replay_buffer.py", "ReplayBuffer", "__init__"): (
        "6d5249a3d04edce426fcb3127cbc229fd713ecd5dc05e151c32a61c61827fb73"
    ),
    ("RL/replay_buffer.py", "ReplayBuffer", "_sample_episode"): (
        "dc8673052b3a3fc72c25477fce26bff92cac6be32721e401b62f87921a3c2b8c"
    ),
    ("RL/replay_buffer.py", "ReplayBuffer", "_store_episode"): (
        "4ad9a72e784647f84ca4034f2fcfc6a9e4d70d946c316f3a8849220e6ef9ba06"
    ),
    ("RL/replay_buffer.py", "ReplayBuffer", "_try_fetch"): (
        "4f4411db93c4bae6b8cc676de31c1b8ba2bd0d569ae7a9159d744f6a91344d27"
    ),
    ("RL/replay_buffer.py", "ReplayBuffer", "_sample"): (
        "1f98d20bd58b3ffb92c1b9f498ae8036652fb7db245041c77b932b41465318c2"
    ),
    ("RL/replay_buffer.py", "ReplayBuffer", "__iter__"): (
        "f31fae887167ed4cab3f5fe1859335102be443b07b6433f7bf06011e71216b56"
    ),
    ("RL/replay_buffer.py", None, "_worker_init_fn"): (
        "7149bb9d772da08804e7bba3168a1b618635a7ac05649f36cc4e82e9efc785ec"
    ),
    ("RL/replay_buffer.py", None, "make_replay_loader"): (
        "e15e485b51aa7861174d75d332cd51cea92f008f02f7f953c20769d6645bc07f"
    ),
    ("RL/utils.py", None, "to_torch"): (
        "a0b54821613c4a25fa453e72043d4956d8f48a3361dcfc32b89149375ff3381d"
    ),
}


def audit_sources(
    manifest_path: Path, receipt_path: Path, source_root: Path
) -> dict[str, object]:
    """Explain approved reward flow only after exact source-span verification."""
    verified = verify_acquisition(manifest_path, receipt_path, source_root)
    texts: dict[str, str] = {}
    source_paths: dict[str, str] = {}
    for item in verified:
        upstream_path = str(item.get("upstream_path", item["path"]))
        if upstream_path in texts:
            raise ValueError(f"duplicate upstream source path: {upstream_path}")
        payload = (source_root / str(item["path"])).read_bytes()
        try:
            texts[upstream_path] = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"audited source is not strict UTF-8: {upstream_path}"
            ) from error
        source_paths[upstream_path] = str(item["path"])
    missing = set(REQUIRED_SOURCES) - texts.keys()
    if missing:
        raise ValueError(f"missing required audit sources: {sorted(missing)}")

    trees = {}
    for path in REQUIRED_SOURCES:
        if not path.endswith(".py"):
            continue
        try:
            trees[path] = ast.parse(texts[path])
        except SyntaxError as error:
            names = ", ".join(
                method_name
                for source_path, _, method_name in APPROVED_SPAN_SHA256
                if source_path == path
            )
            raise ValueError(
                f"invalid audited Python source: {path}; "
                f"span hash unavailable for {names}"
            ) from error

    methods = {}
    observed_hashes = {}
    for key, approved_hash in APPROVED_SPAN_SHA256.items():
        path, class_name, function_name = key
        node = function(trees[path], class_name, function_name)
        observed_hash = hashlib.sha256(function_span(texts[path], node)).hexdigest()
        if observed_hash != approved_hash:
            qualified_name = (
                f"{class_name}.{function_name}" if class_name else function_name
            )
            raise ValueError(f"{qualified_name} span hash mismatch: {observed_hash}")
        methods[key] = node
        owner = class_name or "<module>"
        observed_hashes[f"{path}:{owner}.{function_name}"] = observed_hash

    build = "training/datasets/build.py"
    model = "models/clip_withhead.py"
    support = "models/discretesupport.py"
    agent = "RL/agent/rl_agent.py"
    train = "RL/train.py"
    replay = "RL/replay_buffer.py"
    utils = "RL/utils.py"
    distance = methods[(build, "BaseDataset", "_distance_probs")]
    progress = methods[(build, "BaseDataset", "_progress_from_sample_ids")]
    prepare_train = methods[(build, "BaseDataset", "prepare_train_frames")]
    prepare_test = methods[(build, "BaseDataset", "prepare_test_frames")]
    annotations = methods[(build, "VideoDataset", "load_annotations")]
    regression = methods[(model, "CLIPwithHead", "_regression_logits")]
    prediction = methods[(model, "CLIPwithHead", "predict_progress")]
    transform = methods[(support, None, "transform_one")]
    decode = methods[(support, "DiscreteSupport", "vector_to_scalar")]
    slice_logits = methods[(support, None, "slice_regression_logits")]
    update_critic = methods[(agent, "RLAgent", "update_critic")]
    update = methods[(agent, "RLAgent", "update")]
    rewarder = methods[(agent, "RLAgent", "clip_rewarder")]
    logger = methods[(train, "WorkspaceIL", "_log_timerewarder_reward_curves")]
    setup = methods[(train, "WorkspaceIL", "setup")]
    replay_iter = methods[(train, "WorkspaceIL", "replay_iter")]
    train_il = methods[(train, "WorkspaceIL", "train_il")]
    storage_add = methods[(replay, "ReplayBufferStorage", "add")]
    storage_store = methods[(replay, "ReplayBufferStorage", "_store_episode")]
    save_episode = methods[(replay, None, "save_episode")]
    load_episode = methods[(replay, None, "load_episode")]
    buffer_store = methods[(replay, "ReplayBuffer", "_store_episode")]
    buffer_fetch = methods[(replay, "ReplayBuffer", "_try_fetch")]
    buffer_sample = methods[(replay, "ReplayBuffer", "_sample")]
    buffer_iter = methods[(replay, "ReplayBuffer", "__iter__")]
    make_loader = methods[(replay, None, "make_replay_loader")]
    to_torch = methods[(utils, None, "to_torch")]

    require_statements(
        distance,
        [
            "distance_probs = 1.0 / (possible_distances ** self.weightedsample_distance_power)",
            "return distance_probs / distance_probs.sum()",
        ],
    )
    require_statements(
        progress,
        ["return frame_inds[sample_id].float() / self.progress_max_frames"],
    )
    for prepare in (prepare_train, prepare_test):
        require_statements(
            prepare,
            [
                "distance_probs = self._distance_probs(n)",
                "aug1['progress'] = self._progress_from_sample_ids(sample_id, aug1['frame_inds'])",
            ],
        )
    annotation_fields = dict_fields(annotations)
    if annotation_fields != ["filename", "label"]:
        raise ValueError("load_annotations annotation fields changed")
    if any(
        "action" in identifier
        for function in (annotations, progress, prepare_train, prepare_test)
        for node in ast.walk(function)
        for identifier in node_identifiers(node)
    ):
        raise ValueError("audited annotation or progress path consumes action data")

    require_statements(
        regression, ["return slice_regression_logits(logits, self._regression_dim)"]
    )
    require_statements(
        slice_logits,
        ["return logits[..., LEGACY_ORDER_CLS_DIM:]"],
    )
    require_statements(
        transform,
        ["return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1) + 0.001 * x"],
    )
    require_statements(
        prediction,
        [
            "prev_features = torch.roll(video_features, 1, 0)",
            "prev_features[0] = video_features[0]",
            "forward_features = torch.cat((prev_features, video_features), dim=-1)",
            "reverse_features = torch.cat((video_features, prev_features), dim=-1)",
            "all_features = torch.cat((forward_features, reverse_features), dim=0)",
            "scores = self.discrete_support.vector_to_scalar(logits) if self.use_bin else logits",
            "reward = scores[:n] - scores[n:]",
            "return reward.reshape(n)",
        ],
    )
    require_statements(
        decode,
        [
            "probs = torch.softmax(logits / softmax_temp, dim=-1)",
            "value = (support * probs).sum(-1, keepdim=True)",
            "value = sign * abs_value",
            "return value",
        ],
    )
    require_statements(
        rewarder,
        [
            "reward = self.cost_encoder.predict_progress(obs, self.text_feature)",
            "value = reward.cumsum(dim=0)",
            "raw_reward = reward.cpu().numpy()",
            "reward = raw_reward.copy()",
            "reward += goal_achieved * self.suc_scale * self.suc_signal_scale",
            "return reward, value, raw_reward, self.suc_signal_scale * self.suc_scale",
        ],
    )
    require_statements(
        logger,
        ["plt.plot(value_clip)", "plt.title('value=cumsum(reward)')"],
    )
    require_statements(
        train_il,
        [
            "new_rewards_clip, value_clip, ori_value, reward_scale = self.agent.clip_rewarder(reward_obs, goal_achieved, self.global_step, suc_signal=self.cfg.suc_signal)",
            "new_rewards = new_rewards_clip",
            "elt = elt._replace(reward=float(new_rewards[i - 1]))",
            "self.replay_storage.add(elt)",
            "metrics = self.agent.update(self.replay_iter, self.expert_replay_iter, self.global_step, self.cfg.bc_ratio)",
        ],
    )
    require_statements(
        setup,
        [
            "self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / 'buffer')",
            "self.replay_loader = make_replay_loader(self.work_dir / 'buffer', self.cfg.replay_buffer_size, self.cfg.batch_size, self.cfg.replay_buffer_num_workers, self.cfg.save_experiences, self.cfg.nstep, self.cfg.suite.discount)",
        ],
    )
    require_statements(
        replay_iter,
        [
            "self._replay_iter = iter(self.replay_loader)",
            "return self._replay_iter",
        ],
    )
    require_statements(
        storage_add,
        [
            "value = time_step[spec.name]",
            "self._current_episode[spec.name].append(value)",
            "self._store_episode(episode)",
        ],
    )
    require_statements(
        storage_store, ["save_episode(episode, self._replay_dir / eps_fn)"]
    )
    require_statements(save_episode, ["np.savez_compressed(bs, **episode)"])
    require_statements(load_episode, ["episode = np.load(f)", "return episode"])
    require_statements(buffer_store, ["episode = load_episode(eps_fn)"])
    require_statements(buffer_fetch, ["if not self._store_episode(eps_fn):\n    break"])
    require_statements(
        buffer_sample,
        [
            "step_reward = episode['reward'][idx + i]",
            "reward += discount * step_reward",
            "return (obs, action, reward, discount, next_obs)",
        ],
    )
    require_statements(buffer_iter, ["yield self._sample()"])
    require_statements(
        make_loader,
        [
            "iterable = ReplayBuffer(replay_dir, max_size_per_worker, num_workers, nstep, discount, fetch_every=1000, save_experiences=save_experiences)",
            "return loader",
        ],
    )
    require_statements(
        to_torch, ["return tuple(torch.as_tensor(x, device=device) for x in xs)"]
    )
    require_statements(
        update,
        [
            "obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)",
            "metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))",
        ],
    )
    require_statements(update_critic, ["target_Q = reward + (discount * target_V)"])

    method_ref = source_paths[model]
    rewarder_ref = source_paths[agent]
    train_ref = source_paths[train]
    replay_ref = source_paths[replay]
    utils_ref = source_paths[utils]
    return {
        "annotation_fields": annotation_fields,
        "action_sequence_consumed": False,
        "action_sequence_scope": "annotation and progress-label path",
        "progress_expression": "frame_inds[sample_id] / progress_max_frames",
        "distance_sampling": "1 / distance**power",
        "pairing": ["predecessor,current", "current,predecessor"],
        "dense_reward": {
            "use_bin": "decode(forward) - decode(reverse)",
            "use_bin_false": "forward logit - reverse logit",
        },
        "first_frame": "self-pair",
        "cumulative_value_use": "visualization",
        "success_term": {
            "when_enabled": "added to per-transition replay reward",
            "pre_success_output": "raw_reward",
        },
        "replay_uses_per_transition_reward": True,
        "replay_update_path": (
            "stored transition reward sampled and batched into update_critic"
        ),
        "function_span_sha256": observed_hashes,
        "observation_source_refs": {
            "action_sequence_consumed": [source_paths[build]],
            "dense_reward": [method_ref, source_paths[support]],
            "cumulative_value_use": [rewarder_ref, train_ref],
            "success_term": [rewarder_ref, train_ref],
            "replay_uses_per_transition_reward": [
                rewarder_ref,
                train_ref,
                replay_ref,
            ],
            "replay_update_path": [train_ref, replay_ref, utils_ref, rewarder_ref],
        },
        "source_refs": [item["path"] for item in verified],
    }


def function(
    tree: ast.Module, class_name: str | None, function_name: str
) -> ast.FunctionDef:
    scope = tree.body
    if class_name is not None:
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise ValueError(f"expected one class {class_name}")
        scope = classes[0].body
    functions = [
        node
        for node in scope
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if len(functions) != 1:
        owner = f"{class_name}." if class_name else ""
        raise ValueError(f"expected one {owner}{function_name}")
    return functions[0]


def function_span(text: str, node: ast.FunctionDef) -> bytes:
    lines = text.splitlines(keepends=True)
    start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
    return "".join(lines[start - 1 : node.end_lineno]).encode("utf-8")


def require_statements(function: ast.FunctionDef, sources: list[str]) -> None:
    dumps = {ast.dump(node, include_attributes=False) for node in ast.walk(function)}
    for source in sources:
        expected = ast.parse(source).body[0]
        if ast.dump(expected, include_attributes=False) not in dumps:
            raise ValueError(
                f"{function.name} explanation expression changed: {source}"
            )


def dict_fields(function: ast.FunctionDef) -> list[str]:
    candidates = [
        [keyword.arg for keyword in node.keywords]
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and not node.args
    ]
    if len(candidates) != 1 or any(field is None for field in candidates[0]):
        raise ValueError("load_annotations annotation construction changed")
    return candidates[0]


def node_identifiers(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    return ()
