from homopt.utils import get_least_utilized_gpu
import homopt.utils.runtime as runtime_mod


def test_get_least_utilized_gpu_falls_back_to_cpu(monkeypatch):
    def raise_error(*args, **kwargs):
        raise RuntimeError('nvidia-smi unavailable')

    monkeypatch.setattr(runtime_mod.subprocess, 'check_output', raise_error)
    assert get_least_utilized_gpu() == 'cpu'
