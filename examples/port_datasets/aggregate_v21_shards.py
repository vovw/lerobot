#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

import argparse
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.utils.utils import init_logging


class AggregateV21Shards(PipelineStep):
    def __init__(self, shard_repo_ids: list[str], aggregated_repo_id: str):
        super().__init__()
        self.shard_repo_ids = shard_repo_ids
        self.aggregated_repo_id = aggregated_repo_id

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        init_logging()
        if rank == 0:
            aggregate_datasets(self.shard_repo_ids, self.aggregated_repo_id)


def make_aggr_executor(shard_repo_ids, aggregated_repo_id, job_name, logs_dir, slurm=True):
    kwargs = {
        "pipeline": [AggregateV21Shards(shard_repo_ids, aggregated_repo_id)],
        "logging_dir": str(logs_dir / job_name),
    }
    if slurm:
        kwargs.update({"job_name": job_name, "tasks": 1, "workers": 1, "time": "08:00:00"})
        executor = SlurmPipelineExecutor(**kwargs)
    else:
        kwargs.update({"tasks": 1, "workers": 1})
        executor = LocalPipelineExecutor(**kwargs)
    return executor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-repo-id", type=str, required=True, help="Base repo id used for shards.")
    parser.add_argument("--workers", type=int, required=True, help="Number of shards/workers.")
    parser.add_argument("--aggregated-repo-id", type=str, required=True, help="Output dataset repo id.")
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--job-name", type=str, default="aggregate_v21_shards")
    parser.add_argument("--slurm", type=int, default=0)

    args = parser.parse_args()
    shard_repo_ids = [f"{args.base_repo_id}_world_{args.workers}_rank_{r}" for r in range(args.workers)]
    executor = make_aggr_executor(shard_repo_ids, args.aggregated_repo_id, args.job_name, args.logs_dir, slurm=args.slurm == 1)
    executor.run()


if __name__ == "__main__":
    main()


