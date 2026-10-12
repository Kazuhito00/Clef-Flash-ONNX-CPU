# -*- coding: utf-8 -*-
"""Provider selection (--device / --require-gpu) and the check of what onnxruntime really used. No model, no session."""
import pytest

from clef_decide import device as dev

CPU_ONLY = ["AzureExecutionProvider", "CPUExecutionProvider"]
WITH_CUDA = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
BOTH = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_cpu_is_the_default_and_needs_no_gpu():
    assert dev.resolve_providers("cpu", CPU_ONLY) == ["CPUExecutionProvider"]
    assert dev.resolve_providers("cpu", WITH_CUDA) == ["CPUExecutionProvider"]
    assert dev.resolve_providers(available=WITH_CUDA) == ["CPUExecutionProvider"]


def test_cuda_asks_for_cuda_then_cpu():
    assert dev.resolve_providers("cuda", WITH_CUDA) == BOTH
    assert dev.resolve_providers("auto", WITH_CUDA) == BOTH


def test_auto_without_cuda_is_a_quiet_cpu(capsys):
    assert dev.resolve_providers("auto", CPU_ONLY) == ["CPUExecutionProvider"]
    assert capsys.readouterr().err == ""


def test_cuda_without_cuda_warns_and_uses_cpu(capsys):
    assert dev.resolve_providers("cuda", CPU_ONLY) == ["CPUExecutionProvider"]
    err = capsys.readouterr().err
    assert "warning" in err and "onnxruntime-gpu" in err


@pytest.mark.parametrize("device", ["cuda", "auto"])
def test_require_gpu_raises_without_cuda(device):
    with pytest.raises(dev.GpuUnavailableError, match="onnxruntime-gpu"):
        dev.resolve_providers(device, CPU_ONLY, require_gpu=True)


def test_require_gpu_with_cpu_device_is_contradictory():
    with pytest.raises(dev.GpuUnavailableError):
        dev.resolve_providers("cpu", WITH_CUDA, require_gpu=True)


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        dev.resolve_providers("tpu", WITH_CUDA)


def test_check_used_cpu_request_never_warns(capsys):
    assert dev.check_used(["CPUExecutionProvider"], ["CPUExecutionProvider"]) is True
    assert capsys.readouterr().err == ""


def test_check_used_detects_a_silent_cpu_fallback(capsys):
    assert dev.check_used(BOTH, BOTH) is True
    assert capsys.readouterr().err == ""
    assert dev.check_used(BOTH, ["CPUExecutionProvider"], what="model-fp32.onnx") is False
    assert "model-fp32.onnx" in capsys.readouterr().err
    with pytest.raises(dev.GpuUnavailableError, match="CPUExecutionProvider"):
        dev.check_used(BOTH, ["CPUExecutionProvider"], require_gpu=True)


def test_create_session_falls_back_or_raises_when_cuda_fails_to_load(monkeypatch, capsys):
    from clef_decide import runtime

    calls = []

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None, **kw):
            calls.append(list(providers))
            if providers[0] == "CUDAExecutionProvider":
                raise RuntimeError("CUDA failure 2: out of memory")
            self._p = providers

        def get_providers(self):
            return self._p

    monkeypatch.setattr(runtime.ort, "InferenceSession", FakeSession)
    s = runtime._create_session("x/model.onnx", None, BOTH)
    assert s.get_providers() == ["CPUExecutionProvider"] and calls[-1] == ["CPUExecutionProvider"]
    assert "out of memory" in capsys.readouterr().err
    with pytest.raises(dev.GpuUnavailableError, match="out of memory"):
        runtime._create_session("x/model.onnx", None, BOTH, require_gpu=True)


def test_create_session_catches_a_provider_dropped_without_an_error(monkeypatch, capsys):
    from clef_decide import runtime

    class Dropped:
        def __init__(self, path, sess_options=None, providers=None, **kw):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(runtime.ort, "InferenceSession", Dropped)
    runtime._create_session("m.onnx", None, BOTH)
    assert "runs on the CPU" in capsys.readouterr().err
    with pytest.raises(dev.GpuUnavailableError):
        runtime._create_session("m.onnx", None, BOTH, require_gpu=True)


def test_decider_resolves_device_before_opening_a_session(monkeypatch, tmp_path):
    """device / require_gpu reach resolve_providers; the model files are not needed for the error."""
    from clef_decide import runtime

    monkeypatch.setattr(dev, "available_providers", lambda: ["CPUExecutionProvider"])
    (tmp_path / "model-fp32.onnx").write_bytes(b"")
    with pytest.raises(dev.GpuUnavailableError):
        runtime.Decider(tmp_path, device="cuda", require_gpu=True)


def test_cuda_detection_for_the_graph_optimization_switch():
    """The main graph runs with graph optimization off when a CUDA provider is requested (onnxruntime MemcpyToHost bug)."""
    from clef_decide import runtime

    assert runtime._uses_cuda(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert runtime._uses_cuda(WITH_CUDA)
    assert not runtime._uses_cuda(["CPUExecutionProvider"])
    assert not runtime._uses_cuda(None)
    assert not runtime._uses_cuda([])
