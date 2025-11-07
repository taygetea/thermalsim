"""Integration test for full Rankine cycle."""

import pytest
import numpy as np
from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.components.pump import Pump


def build_rankine_cycle():
    """Helper to construct standard Rankine cycle"""
    graph = ThermalGraph()

    P_high = 10e6
    P_low = 10e3

    boiler = ConstantPressureHeater('boiler', P=P_high, Q=100e6)
    turbine = Turbine('turbine', efficiency=0.85, P_out=P_low)
    condenser = ConstantPressureHeater('condenser', P=P_low, Q=-80e6)
    pump = Pump('pump', efficiency=0.80, P_out=P_high)

    for comp in [boiler, turbine, condenser, pump]:
        graph.add_component(comp)

    graph.connect(boiler.outlet, turbine.inlet)
    graph.connect(turbine.outlet, condenser.inlet)
    graph.connect(condenser.outlet, pump.inlet)
    graph.connect(pump.outlet, boiler.inlet)

    return graph


def test_rankine_cycle_converges():
    """Rankine cycle should reach steady state"""
    graph = build_rankine_cycle()

    result = graph.solve(t_span=(0, 1000), rtol=1e-6, atol=1e-8)

    assert result.success, f"Solver failed: {result.message}"
    assert len(result.t) > 10, "Should take multiple time steps"


def test_rankine_cycle_efficiency():
    """Thermal efficiency should be in expected range"""
    graph = build_rankine_cycle()

    result = graph.solve(t_span=(0, 1000))
    assert result.success

    # Extract final power values
    turbine_state = graph.get_component_state(result, 'turbine')
    pump_state = graph.get_component_state(result, 'pump')

    W_turbine = turbine_state[1, -1]  # turbine power
    W_pump = pump_state[1, -1]        # pump power

    Q_in = 100e6  # from boiler
    W_net = W_turbine - W_pump
    efficiency = W_net / Q_in

    # Rankine cycle at these conditions should be 30-40% efficient
    assert 0.25 < efficiency < 0.42, \
        f"Efficiency {efficiency:.1%} outside expected range (25-42%)"


def test_rankine_energy_balance():
    """Energy should be conserved: Q_in = W_net + Q_out"""
    graph = build_rankine_cycle()

    result = graph.solve(t_span=(0, 1000))
    assert result.success

    turbine_state = graph.get_component_state(result, 'turbine')
    pump_state = graph.get_component_state(result, 'pump')

    W_turbine = turbine_state[1, -1]
    W_pump = pump_state[1, -1]

    Q_in = 100e6
    Q_out = 80e6
    W_net = W_turbine - W_pump

    energy_balance = Q_in - W_net - Q_out

    # Should be close to zero (within 1%)
    assert abs(energy_balance) / Q_in < 0.01, \
        f"Energy imbalance: {energy_balance/1e6:.2f} MW"


def test_rankine_mass_conservation():
    """Mass flow should be conserved around the loop"""
    graph = build_rankine_cycle()

    result = graph.solve(t_span=(0, 1000))
    assert result.success

    # Extract mass flows from each component
    boiler_state = graph.get_component_state(result, 'boiler')
    turbine_state = graph.get_component_state(result, 'turbine')

    mdot_boiler = boiler_state[1, -1]
    mdot_turbine = turbine_state[2, -1]

    # Should all be equal
    np.testing.assert_allclose(mdot_boiler, mdot_turbine, rtol=1e-4)
