"""CPU doubles for CUDA capture selection, recapture, GC and eager fallback."""

import gc
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mjlab.sim import Simulation, SimulationCfg
from mjlab.sim import sim as module


@pytest.fixture
def capture_sim(monkeypatch):
  calls = []
  sim: Any = Simulation.__new__(Simulation)  # Explicit partial CPU test double.
  sim.cfg = SimulationCfg()
  sim.wp_device = SimpleNamespace(is_cuda=True)
  sim._wp_model = object()
  sim._wp_data = object()
  sim._reset_mask_wp = object()
  sim._reset_mask = torch.zeros(2, dtype=torch.bool)
  sim._data_bridge = object()
  sim.nan_guard = SimpleNamespace(watch=lambda _: nullcontext())
  sim._sensor_context = SimpleNamespace(
    prepare=lambda: calls.append("prepare"), finalize=lambda: calls.append("finalize")
  )
  sim.use_cuda_graph = True

  class Capture:
    def __enter__(self):
      assert not gc.isenabled()
      self.graph = object()
      calls.append("capture")
      return self

    def __exit__(self, *_):
      pass

  monkeypatch.setattr(module.wp, "ScopedDevice", lambda _: nullcontext())
  monkeypatch.setattr(module.wp, "ScopedCapture", Capture)
  monkeypatch.setattr(module.wp, "capture_launch", lambda _: calls.append("launch"))
  for name in ("step", "forward", "reset_data"):
    monkeypatch.setattr(
      module.mjwarp, name, lambda *a, _name=name, **k: calls.append(_name)
    )
  monkeypatch.setattr(sim, "_sense_kernel", lambda: calls.append("sense"))
  return sim, calls


@pytest.mark.parametrize("partial", [False, True])
def test_capture_recapture_and_eager_fallback(capture_sim, partial):
  sim, calls = capture_sim
  if partial:
    sim.cfg.capture_reset_sense = False
  expected = ["capture", "step", "capture", "forward"]
  if not partial:
    expected += ["capture", "reset_data", "capture", "sense"]
  enabled = gc.isenabled()
  sim.create_graph()
  assert calls == expected
  assert gc.isenabled() == enabled
  first_step = sim.step_graph
  calls.clear()
  # The real sensor-context entry point recaptures and clears old graph handles.
  sim.set_sensor_context(sim._sensor_context)
  assert calls == expected
  assert sim.step_graph is not first_step
  assert (sim.reset_graph is None) == partial
  assert (sim.sense_graph is None) == partial
  calls.clear()
  sim.step()
  sim.forward()
  sim.reset()
  sim.sense()
  assert calls == (
    ["launch", "launch", "reset_data", "prepare", "sense", "finalize"]
    if partial
    else ["launch", "launch", "launch", "prepare", "launch", "finalize"]
  )
  # Existing unsupported-CUDA / CPU gate wins even when a policy is selected.
  sim.use_cuda_graph = False
  calls.clear()
  sim.create_graph()
  assert calls == []
  assert all(
    getattr(sim, name + "_graph") is None
    for name in ("step", "forward", "reset", "sense")
  )
  sim.step()
  sim.forward()
  sim.reset()
  sim.sense()
  assert calls == ["step", "forward", "reset_data", "prepare", "sense", "finalize"]


def test_capture_failure_restores_gc(capture_sim, monkeypatch):
  sim, _ = capture_sim
  enabled = gc.isenabled()

  def fail(*_):
    raise RuntimeError("capture failure")

  monkeypatch.setattr(module.mjwarp, "step", fail)
  with pytest.raises(RuntimeError, match="capture failure"):
    sim.create_graph()
  assert gc.isenabled() == enabled


@pytest.mark.parametrize(
  "cuda,driver,mempool",
  [
    (False, (13, 0), True),
    (True, None, True),
    (True, (12, 3), True),
    (True, (13, 0), False),
    (True, (13, 0), True),
  ],
)
def test_original_device_gate_unchanged(
  capture_sim, monkeypatch, cuda, driver, mempool
):
  sim, _ = capture_sim
  sim.cfg.capture_reset_sense = False
  sim.wp_device.is_cuda = cuda
  monkeypatch.setattr(module.wp, "get_cuda_driver_version", lambda: driver)
  monkeypatch.setattr(module.wp, "is_mempool_enabled", lambda _: mempool)
  assert sim._should_use_cuda_graph() == bool(cuda and driver == (13, 0) and mempool)


@pytest.mark.parametrize("partial", [False, True])
def test_optimized_lazy_graphs_coexist_with_partial_capture(
  capture_sim, monkeypatch, partial
):
  sim, calls = capture_sim
  sim.cfg.capture_reset_sense = not partial
  for name in ("fwd_kinematics", "com_vel", "set_const"):
    monkeypatch.setattr(
      module.mjwarp, name, lambda *a, _name=name, **k: calls.append(_name)
    )
  sim.create_graph()
  calls.clear()
  sim.forward_kinematics()
  sim.forward_kinematics()
  sim.recompute_constants(module.RecomputeLevel.set_const)
  sim.recompute_constants(module.RecomputeLevel.set_const)
  assert calls == [
    "capture",
    "fwd_kinematics",
    "com_vel",
    "launch",
    "launch",
    "capture",
    "set_const",
    "launch",
    "launch",
  ]
  sim.set_sensor_context(sim._sensor_context)
  assert sim._kinematics_graph is None and sim._recompute_graphs == {}
  sim.use_cuda_graph = False
  calls.clear()
  sim.forward_kinematics()
  sim.recompute_constants(module.RecomputeLevel.set_const)
  assert calls == ["fwd_kinematics", "com_vel", "set_const"]
