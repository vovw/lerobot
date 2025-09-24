#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

import argparse
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset


def compare_values(left, right, rtol=0.0, atol=0.0):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and torch.allclose(left, right, rtol=rtol, atol=atol)
    if isinstance(left, float) and isinstance(right, torch.Tensor):
        return left == right.item()
    if isinstance(left, torch.Tensor) and isinstance(right, float):
        return left.item() == right
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-v3-repo-id", type=str, required=True)
    parser.add_argument("--target-repo-id", type=str, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional local root directory of the source dataset (bypass Hub).",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=None,
        help="Optional local root directory of the target dataset (bypass Hub).",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to compare.")
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="If set, download and compare image/video-derived tensors as well.",
    )
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)

    args = parser.parse_args()

    # Friendly checks when local roots are provided
    if args.source_root is not None:
        info_path = args.source_root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"--source-root does not look like a LeRobot dataset root (missing {info_path})."
            )
    if args.target_root is not None:
        info_path = args.target_root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(
                f"--target-root does not look like a LeRobot dataset root (missing {info_path})."
            )

    # Prefer streaming loaders when a local root is provided to avoid Hub fallback
    if args.source_root is not None:
        ds_src = StreamingLeRobotDataset(
            args.source_v3_repo_id,
            root=args.source_root,
            streaming=True,
            shuffle=False,
        )
    else:
        ds_src = LeRobotDataset(
            args.source_v3_repo_id,
            root=args.source_root,
            download_videos=args.check_images,
        )

    if args.target_root is not None:
        ds_tgt = StreamingLeRobotDataset(
            args.target_repo_id,
            root=args.target_root,
            streaming=True,
            shuffle=False,
        )
    else:
        ds_tgt = LeRobotDataset(
            args.target_repo_id,
            root=args.target_root,
            download_videos=args.check_images,
        )

    if ds_src.meta.fps != ds_tgt.meta.fps:
        raise AssertionError(f"FPS differ: {ds_src.meta.fps} vs {ds_tgt.meta.fps}")

    # Ignore internal/indexing keys which may have benign dtype/shape diffs across builds
    INTERNAL_KEYS = {"timestamp", "episode_index", "index", "frame_index", "task_index"}
    src_feats = {k: v for k, v in ds_src.meta.features.items() if k not in INTERNAL_KEYS}
    tgt_feats = {k: v for k, v in ds_tgt.meta.features.items() if k not in INTERNAL_KEYS}
    if src_feats != tgt_feats:
        missing_left = sorted(set(tgt_feats) - set(src_feats))
        missing_right = sorted(set(src_feats) - set(tgt_feats))
        raise AssertionError(
            f"Feature schemas differ (excluding internal keys). Missing in source: {missing_left}; missing in target: {missing_right}"
        )

    num = min(ds_src.num_frames, ds_tgt.num_frames)
    if args.max_frames is not None:
        num = min(num, args.max_frames)

    # Optionally skip camera keys
    camera_keys = set(ds_src.meta.camera_keys)

    # Iterate using consistent access; lockstep when streaming is involved
    use_streaming = isinstance(ds_src, StreamingLeRobotDataset) or isinstance(ds_tgt, StreamingLeRobotDataset)
    if use_streaming:
        src_iter = iter(ds_src) if isinstance(ds_src, StreamingLeRobotDataset) else (ds_src[i] for i in range(num))
        tgt_iter = iter(ds_tgt) if isinstance(ds_tgt, StreamingLeRobotDataset) else (ds_tgt[i] for i in range(num))
        i = 0
        while i < num:
            try:
                left = next(src_iter)
                right = next(tgt_iter)
            except StopIteration:
                break
            # Compare below
            if left.keys() != right.keys():
                raise AssertionError(f"Keys differ at frame {i}: {set(left.keys()) ^ set(right.keys())}")
            for k in left:
                if (not args.check_images) and (k in camera_keys):
                    continue
                lv = left[k]
                rv = right[k]
                if not compare_values(lv, rv, rtol=args.rtol, atol=args.atol):
                    raise AssertionError(f"Mismatch at frame {i} for key '{k}'")
            i += 1
        print(f"Success: compared {i} frames; datasets match.")
        return
    else:
        for i in range(num):
            left = ds_src[i]
            right = ds_tgt[i]

        if left.keys() != right.keys():
            raise AssertionError(f"Keys differ at frame {i}: {set(left.keys()) ^ set(right.keys())}")

        for k in left:
            if (not args.check_images) and (k in camera_keys):
                continue
            lv = left[k]
            rv = right[k]
            if not compare_values(lv, rv, rtol=args.rtol, atol=args.atol):
                raise AssertionError(f"Mismatch at frame {i} for key '{k}'")

    print(f"Success: compared {num} frames; datasets match.")


if __name__ == "__main__":
    main()


