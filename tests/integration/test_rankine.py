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


# ============================================================================
# Tests for Temporary Sequential Solver (solve_sequential)
# These tests validate the temporary sequential solver implementation
# ============================================================================

def build_rankine_cycle_sequential():
    """Helper to construct Rankine cycle for sequential solver (with matched heat values)"""
    graph = ThermalGraph()

    P_high = 10e6
    P_low = 10e3

    # Use heat values that are thermodynamically consistent with the sequential solver
    # These values are matched to achieve ~31% efficiency
    boiler = ConstantPressureHeater('boiler', P=P_high, Q=252e6)
    turbine = Turbine('turbine', efficiency=0.85, P_out=P_low)
    condenser = ConstantPressureHeater('condenser', P=P_low, Q=-173e6)
    pump = Pump('pump', efficiency=0.80, P_out=P_high)

    for comp in [boiler, turbine, condenser, pump]:
        graph.add_component(comp)

    graph.connect(boiler.outlet, turbine.inlet)
    graph.connect(turbine.outlet, condenser.inlet)
    graph.connect(condenser.outlet, pump.inlet)
    graph.connect(pump.outlet, boiler.inlet)

    return graph


def test_rankine_sequential_converges():
    """Sequential solver should converge for Rankine cycle"""
    graph = build_rankine_cycle_sequential()

    result = graph.solve_sequential()

    assert result.success, f"Sequential solver failed: {result.message}"
    assert result.x is not None, "Solution state should be populated"
    assert len(result.x) > 0, "Solution should have non-zero length"


def test_rankine_sequential_efficiency():
    """Sequential solver should produce correct thermal efficiency"""
    graph = build_rankine_cycle_sequential()

    result = graph.solve_sequential()
    assert result.success, f"Sequential solver failed: {result.message}"

    # Extract power values from final state
    # State vector layout: [boiler_h_out, boiler_mdot, turbine_h_out, turbine_W, turbine_mdot, ...]
    turbine_state = graph.get_component_state(result, 'turbine')
    pump_state = graph.get_component_state(result, 'pump')

    # Extract power values (index 1 for both turbine and pump)
    W_turbine = turbine_state[1, 0]  # Sequential result has single time point
    W_pump = pump_state[1, 0]

    Q_in = 252e6  # from boiler
    W_net = W_turbine - W_pump
    efficiency = W_net / Q_in

    # Rankine cycle at these conditions should be 30-40% efficient
    # Sequential solver achieves ~31% based on examples/rankine_cycle.py
    assert 0.28 < efficiency < 0.35, \
        f"Efficiency {efficiency:.1%} outside expected range (28-35%)"


def test_rankine_sequential_energy_balance():
    """Sequential solver should conserve energy"""
    graph = build_rankine_cycle_sequential()

    result = graph.solve_sequential()
    assert result.success

    turbine_state = graph.get_component_state(result, 'turbine')
    pump_state = graph.get_component_state(result, 'pump')

    W_turbine = turbine_state[1, 0]
    W_pump = pump_state[1, 0]

    Q_in = 252e6
    Q_out = 173e6
    W_net = W_turbine - W_pump

    energy_balance = Q_in - W_net - Q_out

    # Should be close to zero (within 1%)
    assert abs(energy_balance) / Q_in < 0.01, \
        f"Energy imbalance: {energy_balance/1e6:.2f} MW ({energy_balance/Q_in*100:.2f}%)"


def test_rankine_sequential_mass_conservation():
    """Sequential solver should produce consistent mass flow"""
    graph = build_rankine_cycle_sequential()

    result = graph.solve_sequential()
    assert result.success

    # All components should have the same mass flow rate
    boiler_state = graph.get_component_state(result, 'boiler')
    turbine_state = graph.get_component_state(result, 'turbine')
    condenser_state = graph.get_component_state(result, 'condenser')
    pump_state = graph.get_component_state(result, 'pump')

    # Mass flow is last variable for each component (index varies by component)
    mdot_boiler = boiler_state[1, 0]      # boiler: [h_out, mdot]
    mdot_turbine = turbine_state[2, 0]    # turbine: [h_out, W, mdot]
    mdot_condenser = condenser_state[1, 0] # condenser: [h_out, mdot]
    mdot_pump = pump_state[2, 0]          # pump: [h_out, W, mdot]

    # All should be equal
    np.testing.assert_allclose(mdot_boiler, mdot_turbine, rtol=1e-6)
    np.testing.assert_allclose(mdot_turbine, mdot_condenser, rtol=1e-6)
    np.testing.assert_allclose(mdot_condenser, mdot_pump, rtol=1e-6)


def test_sequential_solver_only_supports_4_components():
    """Sequential solver should raise error if graph has != 4 components"""
    graph = ThermalGraph()

    # Add only 2 components
    boiler = ConstantPressureHeater('boiler', P=10e6, Q=100e6)
    turbine = Turbine('turbine', efficiency=0.85, P_out=10e3)

    graph.add_component(boiler)
    graph.add_component(turbine)
    graph.connect(boiler.outlet, turbine.inlet)

    # Should raise ValueError about 4-component limitation
    with pytest.raises(ValueError, match="Sequential solver requires components named"):
        result = graph.solve_sequential()
