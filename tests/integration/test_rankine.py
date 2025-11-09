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
# ============================================================================
# Tests for Steady-State Solver (solve_steady_state)
# These tests validate the simultaneous steady-state solver with multiple backends
# ============================================================================

def build_rankine_cycle_steady():
    """Helper to construct Rankine cycle for steady-state solver"""
    graph = ThermalGraph()

    P_high = 10e6
    P_low = 10e3

    # Heat values matched for ~31% efficiency
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


def test_rankine_steady_scipy_converges():
    """Scipy backend should converge for Rankine cycle (with sequential init)"""
    graph = build_rankine_cycle_steady()

    # With sequential initialization enabled (default), this should converge
    result = graph.solve_steady_state(backend='scipy')

    assert result.success, f"Scipy solver failed: {result.message}"
    assert result.x is not None, "Solution state should be populated"
    assert len(result.x) > 0, "Solution should have non-zero length"
    # Should use sequential init since direct scipy fails
    assert 'sequential' in result.solver_used.lower(), \
        f"Expected sequential initialization, got: {result.solver_used}"


def test_rankine_steady_scipy_efficiency():
    """Scipy backend should produce correct thermal efficiency"""
    graph = build_rankine_cycle_steady()

    result = graph.solve_steady_state(backend='scipy')
    assert result.success, f"Scipy solver failed: {result.message}"

    # Extract power values from final state
    turbine_state = graph.get_component_state(result, 'turbine')
    pump_state = graph.get_component_state(result, 'pump')

    # Extract power values (index 1 for both turbine and pump)
    W_turbine = turbine_state[1, 0]
    W_pump = pump_state[1, 0]

    Q_in = 252e6  # from boiler
    W_net = W_turbine - W_pump
    efficiency = W_net / Q_in

    # Rankine cycle at these conditions should be 30-40% efficient
    assert 0.28 < efficiency < 0.35, \
        f"Efficiency {efficiency:.1%} outside expected range (28-35%)"


def test_rankine_steady_scipy_energy_balance():
    """Scipy backend should conserve energy"""
    graph = build_rankine_cycle_steady()

    result = graph.solve_steady_state(backend='scipy')
    assert result.success, f"Scipy solver failed: {result.message}"

    # Extract power and heat transfer values
    turbine_state = graph.get_component_state(result, 'turbine')
    pump_state = graph.get_component_state(result, 'pump')

    W_turbine = turbine_state[1, 0]
    W_pump = pump_state[1, 0]
    W_net = W_turbine - W_pump

    Q_in = 252e6
    Q_out = -173e6  # Negative (heat rejection)
    Q_net = Q_in + Q_out

    # Energy balance: Q_net should equal W_net (within tolerance)
    rel_error = abs(Q_net - W_net) / Q_in
    assert rel_error < 0.01, \
        f"Energy balance error {rel_error:.1%} exceeds 1% (Q_net={Q_net/1e6:.1f} MW, W_net={W_net/1e6:.1f} MW)"


def test_rankine_steady_scipy_mass_conservation():
    """Scipy backend should conserve mass flow"""
    graph = build_rankine_cycle_steady()

    result = graph.solve_steady_state(backend='scipy')
    assert result.success, f"Scipy solver failed: {result.message}"

    # All components should have same mass flow rate in closed loop
    boiler_state = graph.get_component_state(result, 'boiler')
    turbine_state = graph.get_component_state(result, 'turbine')
    condenser_state = graph.get_component_state(result, 'condenser')
    pump_state = graph.get_component_state(result, 'pump')

    # Mass flow is last state variable for each component
    mdot_boiler = boiler_state[-1, 0]
    mdot_turbine = turbine_state[-1, 0]
    mdot_condenser = condenser_state[-1, 0]
    mdot_pump = pump_state[-1, 0]

    # All should be equal within numerical tolerance
    np.testing.assert_allclose(mdot_boiler, mdot_turbine, rtol=1e-6)
    np.testing.assert_allclose(mdot_turbine, mdot_condenser, rtol=1e-6)
    np.testing.assert_allclose(mdot_condenser, mdot_pump, rtol=1e-6)


def test_rankine_steady_sequential_backend():
    """Sequential backend should work directly"""
    graph = build_rankine_cycle_steady()

    result = graph.solve_steady_state(backend='sequential')

    assert result.success, f"Sequential solver failed: {result.message}"
    assert result.x is not None, "Solution state should be populated"
    assert len(result.x) > 0, "Solution should have non-zero length"
    assert result.solver_used == 'sequential', f"Expected 'sequential', got {result.solver_used}"


def test_rankine_steady_diffeqpy_converges():
    """Diffeqpy backend should converge for Rankine cycle (with sequential init)"""
    pytest.importorskip("diffeqpy", reason="diffeqpy not installed")

    graph = build_rankine_cycle_steady()
    result = graph.solve_steady_state(backend='diffeqpy')

    assert result.success, f"Diffeqpy solver failed: {result.message}"
    assert result.x is not None, "Solution state should be populated"
    assert len(result.x) > 0, "Solution should have non-zero length"
    # Should use sequential init or fallback
    assert 'sequential' in result.solver_used.lower() or result.solver_used == 'diffeqpy', \
        f"Unexpected solver: {result.solver_used}"
