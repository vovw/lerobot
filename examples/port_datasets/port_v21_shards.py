#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep


class PortV21Shards(PipelineStep):
    def __init__(self, repo_id: str, out_repo_prefix: str | None = None):
        super().__init__()
        self.repo_id = repo_id
        self.out_repo_prefix = out_repo_prefix or repo_id

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        from datasets.utils.tqdm import disable_progress_bars
        from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
        from lerobot.utils.utils import init_logging

        init_logging()
        disable_progress_bars()

        out_repo_id = f"{self.out_repo_prefix}_world_{world_size}_rank_{rank}"

        convert_dataset(
            repo_id=self.repo_id,
            num_shards=world_size,
            shard_index=rank,
            out_repo_id=out_repo_id,
            push_to_hub=False,
        )


def make_port_v21_executor(repo_id, out_repo_prefix, job_name, logs_dir, workers, partition, cpus_per_task, mem_per_cpu, slurm=True):
    kwargs = {
        "pipeline": [
            PortV21Shards(repo_id, out_repo_prefix),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": workers,
                "workers": workers,
                "time": "08:00:00",
                "partition": partition,
                "cpus_per_task": cpus_per_task,
                "sbatch_args": {"mem-per-cpu": mem_per_cpu},
            }
        )
        executor = SlurmPipelineExecutor(**kwargs)
    else:
        kwargs.update(
            {
                "tasks": workers,
                "workers": workers,
            }
        )
        executor = LocalPipelineExecutor(**kwargs)

    return executor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Source dataset repo-id on the Hub (expects revision v2.1 to exist).",
    )
    parser.add_argument(
        "--out-repo-prefix",
        type=str,
        default=None,
        help="Optional prefix for produced shard repo ids. Defaults to --repo-id.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        required=True,
        help="Path to logs directory for datatrove.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="port_v21_shards",
        help="Job name for slurm and logs subdirectory.",
    )
    parser.add_argument(
        "--slurm",
        type=int,
        default=0,
        help="Launch over slurm. Use --slurm 0 for local execution.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of shards/workers.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="Slurm partition.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=4,
        help="CPUs per task.",
    )
    parser.add_argument(
        "--mem-per-cpu",
        type=str,
        default="1950M",
        help="Memory per CPU.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    kwargs["slurm"] = kwargs.pop("slurm") == 1
    executor = make_port_v21_executor(**kwargs)
    executor.run()


if __name__ == "__main__":
    main()


