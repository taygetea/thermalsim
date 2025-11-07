"""Unit tests for component base class and implementations."""

import pytest
import numpy as np
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.core.port import MassFlowPort


def test_heater_creation():
    """Test heater component instantiation"""
    heater = ConstantPressureHeater('h1', P=1e6, Q=1e6)

    assert heater.name == 'h1'
    assert heater.P == 1e6
    assert heater.Q == 1e6
    assert 'inlet' in heater.ports
    assert 'outlet' in heater.ports


def test_heater_state_size():
    """Heater should have 2 state variables"""
    heater = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    assert heater.get_state_size() == 2


def test_heater_initial_state():
    """Initial state should be reasonable array"""
    heater = ConstantPressureHeater('h1', P=1e6, Q=1e6)
    y0 = heater.get_initial_state()

    assert isinstance(y0, np.ndarray)
    assert y0.shape == (2,)
    assert np.all(np.isfinite(y0))


def test_heater_residual_shape():
    """Residual should return correct shape"""
    heater = ConstantPressureHeater('h1', P=1e6, Q=1e6)

    state = np.array([1e6, 100.0])
    ports = {
        'inlet': MassFlowPort('in', 'in'),
        'outlet': MassFlowPort('out', 'out')
    }
    ports['inlet'].h = 5e5
    ports['inlet'].P = 1e6

    res = heater.residual(state, ports, t=0)

    assert isinstance(res, np.ndarray)
    assert res.shape == (2,)


def test_turbine_efficiency_validation():
    """Turbine should reject invalid efficiencies"""
    with pytest.raises(ValueError):
        Turbine('t1', efficiency=1.5, P_out=1e5)

    with pytest.raises(ValueError):
        Turbine('t1', efficiency=0.0, P_out=1e5)
