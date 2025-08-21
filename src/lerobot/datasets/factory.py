#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc.
# Licensed under the Apache License, Version 2.0

import logging
from pprint import pformat
import os
import numpy as np

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.transforms import ImageTransforms

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],   # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """
    Build delta_timestamps from the config's *_delta_indices and the dataset fps.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    return delta_timestamps or None


def _concat_stats(jointstat: dict, gripperstat: dict) -> dict:
    """
    Concatenate per-key stats dictionaries (mean/std/min/max) along the last axis.
    Assumes arrays, not torch tensors.
    """
    out = {}
    for k in ("mean", "std", "min", "max"):
        if k in jointstat and k in gripperstat:
            a = np.asarray(jointstat[k])
            b = np.asarray(gripperstat[k])
            # make sure gripper stat shapes broadcast (e.g., (8,) vs (1,))
            if b.ndim < a.ndim:
                # expand dims until ranks match, then broadcast
                while b.ndim < a.ndim:
                    b = np.expand_dims(b, axis=0)
                b = np.broadcast_to(b, a.shape[:-1] + b.shape[-1:])
            out[k] = np.concatenate([a, b], axis=-1)
    # carry count (use joints count)
    if "count" in jointstat:
        out["count"] = jointstat["count"]
    return out


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    # Log where this function is coming from to ensure the right factory is used
    logging.warning(
        f"⚠️ make_dataset loaded from: {__file__}"
    )

    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
        if cfg.dataset.image_transforms.enable
        else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        # ----- Build/edit metadata -----
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )

        # We require these source keys
        required = ["observation.state.joints", "observation.state.gripper"]
        missing = [k for k in required if k not in ds_meta.features]
        if missing:
            raise KeyError(
                f"Dataset is missing required keys for action/state merge: {missing}. "
                f"Available keys: {list(ds_meta.features.keys())}"
            )

        # Create merged stats for both `action` and `observation.state` = concat(joints(7), gripper(1)) -> (8,)
        jointstat = ds_meta.stats["observation.state.joints"]
        gripperstat = ds_meta.stats["observation.state.gripper"]
        merged_stat = _concat_stats(jointstat, gripperstat)  # (8,)

        ds_meta.stats["action"] = merged_stat
        ds_meta.stats["observation.state"] = merged_stat

        # Create features for both keys
        ft = ds_meta.features
        joint_ft = ft["observation.state.joints"]
        gripper_ft = ft["observation.state.gripper"]

        merged_names = (joint_ft.get("names") or []) + (gripper_ft.get("names") or [])

        merged_feature = {
            "dtype": "float32",
            "shape": list(merged_stat["mean"].shape),  # e.g., [8]
            "names": merged_names,
        }
        ds_meta.features["action"] = merged_feature
        ds_meta.features["observation.state"] = merged_feature

        # optional: prune the raw component fields to simplify model inputs
        for thing in (ds_meta.stats, ds_meta.features):
            for k in [
                "observation.state.joints",
                "observation.state.gripper",
                "observation.state.position",
                "observation.image.side",  # drop if you don't want this camera
            ]:
                if k in thing:
                    del thing[k]

        # ----- Create dataset -----
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )

        # Inject both columns (`action` and `observation.state`) into the HF dataset table
        def add_action_and_state(batch):
            # joints: (B,7), gripper: (B,) or (B,1)
            joints = np.asarray(batch["observation.state.joints"])
            gripper = np.asarray(batch["observation.state.gripper"])
            if gripper.ndim == 1:
                gripper = gripper[:, None]
            merged = np.concatenate([joints, gripper], axis=-1)  # (B,8)
            return {"action": merged, "observation.state": merged}

        dataset.hf_dataset = dataset.hf_dataset.map(add_action_and_state, batched=True)

        # Point dataset.meta to our edited metadata (with merged features/stats)
        dataset.meta = ds_meta

        # Debug prints
        logging.warning(
            "✅ make_dataset added 'action' (7 joints + 1 gripper) "
            "and 'observation.state' (same 8-D). Features now: %s",
            list(dataset.meta.features.keys()),
        )

    else:
        # Multi dataset (currently deactivated)
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        # If you ever re-enable:
        # dataset = MultiLeRobotDataset(
        #     cfg.dataset.repo_id,
        #     image_transforms=image_transforms,
        #     video_backend=cfg.dataset.video_backend,
        # )
        # logging.info(
        #     "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
        #     f"{pformat(dataset.repo_id_to_index, indent=2)}"
        # )

    # Optional: apply ImageNet stats for vision keys
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(
                    stats, dtype=torch.float32
                )

    return dataset

