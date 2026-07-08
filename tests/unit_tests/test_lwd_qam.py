import importlib.util
import sys
from pathlib import Path

import torch


_QAM_PATH = Path(__file__).parents[2] / "rlinf" / "algorithms" / "lwd" / "qam.py"
_QAM_SPEC = importlib.util.spec_from_file_location("lwd_qam_test_module", _QAM_PATH)
assert _QAM_SPEC is not None
qam = importlib.util.module_from_spec(_QAM_SPEC)
sys.modules[_QAM_SPEC.name] = qam
assert _QAM_SPEC.loader is not None
_QAM_SPEC.loader.exec_module(qam)


def test_qam_loss_matches_openpi_flow_direction():
    timestep = torch.tensor([0.5])
    sigma_sq = 2.0 * (1.0 - timestep) * timestep
    adjoint = torch.tensor([[[-1.0]]])
    v_beta = torch.zeros(1, 1, 1)

    # For OpenPI flow-ODE, x_next = x - dt * v.  A positive critic
    # action gradient gives a negative terminal adjoint, so the learned
    # velocity should move below the reference velocity to increase the
    # final action.
    v_good = v_beta + 0.5 * sigma_sq.view(1, 1, 1) * adjoint
    v_bad = -v_good

    good_loss, _, _ = qam.qam_vector_field_loss(v_good, v_beta, adjoint, timestep)
    bad_loss, _, _ = qam.qam_vector_field_loss(v_bad, v_beta, adjoint, timestep)

    assert good_loss < bad_loss
    assert qam.flow_ode_step(torch.zeros_like(v_good), v_good, 0, 2).item() > 0.0
