"""GPU utility helpers."""

from __future__ import annotations

import subprocess


def get_least_utilized_gpu():
    try:
        gpu_stats = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8',
        )
        gpu_memory = [tuple(map(int, x.split(','))) for x in gpu_stats.strip().splitlines() if x.strip()]
        gpu_memory = [(total - free, idx) for idx, (free, total) in enumerate(gpu_memory)]
        return sorted(gpu_memory, key=lambda x: x[0])[0][1]
    except Exception:
        return 'cpu'
