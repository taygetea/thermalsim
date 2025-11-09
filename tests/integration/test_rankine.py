"""Integration test for full Rankine cycle."""

import pytest
import numpy as np
from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.components.pump import Pump


# ============================================================================
# Tests for Steady-State Solver (solve_steady_state)
# These tests validate the simultaneous steady-state solver with multiple backends
#
# Note: The MVP system has only algebraic variables (no differential equations).
# The transient ODE solver (solve()) is not applicable until Phase 3 when
# differential variables are added. All steady-state testing uses solve_steady_state().
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
