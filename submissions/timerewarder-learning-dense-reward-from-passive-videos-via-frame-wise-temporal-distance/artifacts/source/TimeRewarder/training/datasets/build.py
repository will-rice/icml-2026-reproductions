from logging import Logger
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import numpy as np
from functools import partial

import io
import os
import os.path as osp
from collections.abc import Mapping, Sequence
from torch.utils.data import Dataset
import copy
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from .pipeline import *
from torch.utils.data.dataloader import default_collate
from mmcv.parallel import collate
import pandas as pd

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


def _video_frame_count(filename):
    import decord
    with open(filename, 'rb') as f:
        file_obj = io.BytesIO(f.read())
    return len(decord.VideoReader(file_obj, num_threads=1))


def compute_dataset_max_frames(data_prefix, ann_files):
    max_frames = 0
    seen = set()
    for ann_file in ann_files:
        if ann_file is None or not osp.isfile(ann_file):
            continue
        with open(ann_file, 'r') as fin:
            for line in fin:
                path = line.strip().split()[0]
                if data_prefix is not None:
                    path = osp.join(data_prefix, path)
                if path in seen:
                    continue
                seen.add(path)
                max_frames = max(max_frames, _video_frame_count(path))
    if max_frames <= 0:
        raise ValueError('Failed to resolve dataset max video length from ann files.')
    return max_frames


def resolve_progress_max_frames(config, logger=None):
    raw = config.DATA.PROGRESS_MAX_FRAMES
    if isinstance(raw, str) and raw.lower() == 'auto':
        max_frames = compute_dataset_max_frames(
            config.DATA.ROOT,
            [config.DATA.TRAIN_FILE, config.DATA.VAL_FILE],
        )
        if logger is not None:
            logger.info(f'DATA.PROGRESS_MAX_FRAMES=auto -> {max_frames}')
        return max_frames
    max_frames = int(raw)
    if max_frames <= 0:
        raise ValueError(f'Invalid DATA.PROGRESS_MAX_FRAMES={raw!r}; use a positive int or "auto".')
    if logger is not None:
        logger.info(f'DATA.PROGRESS_MAX_FRAMES={max_frames}')
    return max_frames


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        ann_file,
        pipeline,
        data_prefix=None,
        test_mode=False,
        start_index=1,
        modality='RGB',
        sample_by_class=False,
        power=0,
        clip_num=2,
        clip_len=8,
        frame_step=4,
        augmentation_factor=100,
        weightedsample_train=True,
        weightedsample_distance_power=1,
        progress_max_frames=None,
    ):
        super().__init__()
        self.ann_file = ann_file
        self.data_prefix = osp.realpath(
            data_prefix) if data_prefix is not None and osp.isdir(
                data_prefix) else data_prefix
        self.test_mode = test_mode
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        self.clip_num = clip_num
        self.clip_len = clip_len
        self.frame_step = frame_step
        self.augmentation_factor = augmentation_factor
        self.weightedsample_train = weightedsample_train
        self.weightedsample_distance_power = weightedsample_distance_power
        if progress_max_frames is None:
            raise ValueError('progress_max_frames is required')
        self.progress_max_frames = progress_max_frames

        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

            class_prob = []
            for _, samples in self.video_infos_by_class.items():
                class_prob.append(len(samples) / len(self.video_infos))
            class_prob = [x**self.power for x in class_prob]

            summ = sum(class_prob)
            class_prob = [x / summ for x in class_prob]

            self.class_prob = dict(zip(self.video_infos_by_class, class_prob))

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    def _distance_probs(self, n):
        """P(delta) ∝ 1 / delta^power. power=1 inverse; power=2 inverse-square."""
        possible_distances = torch.arange(1, n, dtype=torch.float32)
        distance_probs = 1.0 / (possible_distances ** self.weightedsample_distance_power)
        return distance_probs / distance_probs.sum()

    def _progress_from_sample_ids(self, sample_id, frame_inds):
        if isinstance(frame_inds, np.ndarray):
            frame_inds = torch.from_numpy(frame_inds)
        elif not isinstance(frame_inds, torch.Tensor):
            frame_inds = torch.tensor(frame_inds)
        return frame_inds[sample_id].float() / self.progress_max_frames

    def prepare_train_frames(self, idx, sampleclip=True, weightedsample=False):
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        aug1 = self.pipeline(results)

        if sampleclip:
            step = 1
            frames = aug1['imgs']
            frame_step = 1
            frames_unfold = frames.unfold(0, self.clip_len * frame_step, step).permute(0, 4, 1, 2, 3)
            frames_unfold = frames_unfold[:, ::frame_step]
            if not weightedsample:
                sample_id = torch.randperm(frames_unfold.shape[0])[:self.clip_num]
                sample_id = sample_id.sort()[0]
            else:
                n = frames_unfold.shape[0]
                distance_probs = self._distance_probs(n)
                distance_idx = torch.multinomial(distance_probs, 1).item()
                distance = distance_idx + 1
                max_first_idx = n - distance - 1
                first_idx = torch.randint(0, max_first_idx + 1, (1,)).item()
                second_idx = first_idx + distance
                sample_id = torch.tensor([first_idx, second_idx])

            aug1['imgs'] = frames_unfold[sample_id]
            aug1['progress'] = self._progress_from_sample_ids(sample_id, aug1['frame_inds'])

        return aug1

    def prepare_test_frames(self, idx, sampleclip=True, weightedsample=True):
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        aug1 = self.pipeline(results)

        if sampleclip:
            step = 1
            frames = aug1['imgs']
            frame_step = self.frame_step
            frames_unfold = frames.unfold(0, self.clip_len * frame_step, step).permute(0, 4, 1, 2, 3)
            frames_unfold = frames_unfold[:, ::frame_step]
            n = frames_unfold.shape[0]
            distance_probs = self._distance_probs(n)
            distance_idx = torch.multinomial(distance_probs, 1).item()
            distance = distance_idx + 1
            max_first_idx = n - distance - 1
            first_idx = torch.randint(0, max_first_idx + 1, (1,)).item()
            second_idx = first_idx + distance
            sample_id = torch.tensor([first_idx, second_idx])
            aug1['imgs'] = frames_unfold[sample_id]
            aug1['progress'] = self._progress_from_sample_ids(sample_id, aug1['frame_inds'])

        return aug1

    def __len__(self):
        if self.test_mode:
            return len(self.video_infos) * 10
        return len(self.video_infos) * self.augmentation_factor

    def __getitem__(self, idx):
        original_idx = idx % len(self.video_infos)
        if self.test_mode:
            return self.prepare_test_frames(original_idx)
        return self.prepare_train_frames(
            original_idx, sampleclip=True, weightedsample=self.weightedsample_train)


class VideoDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, labels_file=None, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        self.labels_file = labels_file

    @property
    def classes(self):
        if not self.labels_file:
            return [['task']]
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                parts = line.strip().split()
                filename = parts[0]
                label = int(parts[1]) if len(parts) > 1 else 0
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label))
        return video_infos


class SubsetRandomSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


def mmcv_collate(batch, samples_per_gpu=1):
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')
    if isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: mmcv_collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


def build_dataloader(logger, config):
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)
    progress_max_frames = resolve_progress_max_frames(config, logger)

    data_frame_number = config.DATA.NUM_FRAMES

    train_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=data_frame_number),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(
            type='MultiScaleCrop',
            input_size=config.DATA.INPUT_SIZE,
            scales=(1, 0.95, 0.875, 0.8),
            random_crop=True,
            max_wh_scale_gap=1),
        dict(type='Resize', scale=(config.DATA.INPUT_SIZE, config.DATA.INPUT_SIZE), keep_ratio=False),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='ColorJitter', p=config.AUG.COLOR_JITTER),
        dict(type='GrayScale', p=config.AUG.GRAY_SCALE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label', 'frame_inds'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label', 'frame_inds']),
    ]
    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=data_frame_number, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label', 'frame_inds'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label', 'frame_inds'])
    ]

    labels_file = config.DATA.LABEL_LIST or None

    if config.DATA.TRAIN_FILE is not None:
        train_data = VideoDataset(
            ann_file=config.DATA.TRAIN_FILE, data_prefix=config.DATA.ROOT,
            labels_file=labels_file, pipeline=train_pipeline,
            clip_num=config.DATA.NUM_CLIPS, clip_len=config.DATA.NUM_FRAMES_CLIP,
            weightedsample_train=config.DATA.WEIGHTED_SAMPLE,
            weightedsample_distance_power=config.DATA.WEIGHTED_SAMPLE_DISTANCE_POWER,
            progress_max_frames=progress_max_frames,
            augmentation_factor=config.DATA.SAMPLES_PER_VIDEO,
        )
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        train_loader = DataLoader(
            train_data, sampler=sampler_train,
            batch_size=config.TRAIN.BATCH_SIZE,
            num_workers=16,
            pin_memory=True,
            drop_last=True,
            collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
        )
    else:
        train_data = None
        train_loader = None

    val_data = VideoDataset(
        ann_file=config.DATA.VAL_FILE, data_prefix=config.DATA.ROOT,
        labels_file=labels_file, pipeline=val_pipeline,
        clip_num=config.DATA.NUM_CLIPS_VAL, clip_len=config.DATA.NUM_FRAMES_CLIP,
        test_mode=True, frame_step=config.DATA.CLIP_FRAME_STEP,
        weightedsample_distance_power=config.DATA.WEIGHTED_SAMPLE_DISTANCE_POWER,
        progress_max_frames=progress_max_frames,
    )
    indices = np.arange(dist.get_rank(), len(val_data), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)
    val_loader = DataLoader(
        val_data, sampler=sampler_val,
        batch_size=config.TEST.BATCH_SIZE,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.TEST.BATCH_SIZE),
    )

    return train_data, val_data, train_loader, val_loader
