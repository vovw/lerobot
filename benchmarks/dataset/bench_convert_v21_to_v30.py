#!/usr/bin/env python

import argparse
import time
from pathlib import Path

from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset


def fmt(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def bench_non_sharded(repo_id: str):
    t0 = time.time()
    convert_dataset(repo_id=repo_id, push_to_hub=False)
    return time.time() - t0


def bench_sharded(repo_id: str, workers: int):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.base import PipelineStep

    class Runner(PipelineStep):
        def __init__(self, repo_id: str):
            super().__init__()
            self.repo_id = repo_id

        def run(self, data=None, rank: int = 0, world_size: int = 1):
            convert_dataset(
                repo_id=self.repo_id,
                num_shards=world_size,
                shard_index=rank,
                out_repo_id=f"{self.repo_id}_world_{world_size}_rank_{rank}",
                push_to_hub=False,
            )

    exec = LocalPipelineExecutor(pipeline=[Runner(repo_id)], tasks=workers, workers=workers)
    t0 = time.time()
    exec.run()
    return time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    t_non = bench_non_sharded(args.repo_id)
    t_shard = bench_sharded(args.repo_id, args.workers)

    print({
        "repo_id": args.repo_id,
        "non_sharded_sec": round(t_non, 3),
        "non_sharded_hms": fmt(t_non),
        "sharded_workers": args.workers,
        "sharded_sec": round(t_shard, 3),
        "sharded_hms": fmt(t_shard),
        "speedup_x": round(t_non / t_shard if t_shard > 0 else float('inf'), 2),
    })


if __name__ == "__main__":
    main()


