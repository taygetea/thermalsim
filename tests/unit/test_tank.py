"""Unit tests for SimpleTank component (Phase 3 Milestone 2)."""

import pytest
import numpy as np
from thermal_sim.components.tank import SimpleTank
from thermal_sim.core.graph import ThermalGraph


class TestSimpleTankCreation:
    """Test SimpleTank initialization"""

    def test_creation(self):
        """SimpleTank should initialize with required parameters"""
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)

        assert tank.name == 'tank'
        assert tank.A == 10.0
        assert tank.P == 2e5
        assert tank.P_downstream == 1e5
        assert tank.C_v == 0.01
        assert tank.level_initial == 2.0  # Default

    def test_custom_initial_level(self):
        """SimpleTank should accept custom initial level"""
        tank = SimpleTank('tank', A=5.0, P=2e5, P_downstream=1e5,
                          C_v=0.01, level_initial=5.0)

        assert tank.level_initial == 5.0

    def test_has_ports(self):
        """SimpleTank should have inlet and outlet MassFlowPorts"""
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)

        assert 'inlet' in tank.ports
        assert 'outlet' in tank.ports
        assert tank.ports['inlet'].direction == 'in'
        assert tank.ports['outlet'].direction == 'out'

    def test_invalid_area(self):
        """SimpleTank should reject non-positive area"""
        with pytest.raises(ValueError, match="Area must be positive"):
            SimpleTank('tank', A=-1.0, P=2e5, P_downstream=1e5, C_v=0.01)

        with pytest.raises(ValueError, match="Area must be positive"):
            SimpleTank('tank', A=0.0, P=2e5, P_downstream=1e5, C_v=0.01)

    def test_invalid_initial_level(self):
        """SimpleTank should reject non-positive initial level"""
        with pytest.raises(ValueError, match="Initial level must be positive"):
            SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01, level_initial=0.0)


class TestSimpleTankInterface:
    """Test Component interface implementation"""

    @pytest.fixture
    def tank(self):
        """Standard tank for tests"""
        return SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)

    def test_get_variables(self, tank):
        """get_variables() should return 1 differential + 1 algebraic variable"""
        variables = tank.get_variables()

        assert len(variables) == 2
        assert variables[0].name == 'level'
        assert variables[0].kind == 'differential'  # NEW: First differential variable!
        assert variables[1].name == 'mdot_out'
        assert variables[1].kind == 'algebraic'

    def test_get_initial_state(self, tank):
        """get_initial_state() should return array of length 2"""
        state = tank.get_initial_state()

        assert isinstance(state, np.ndarray)
        assert len(state) == 2
        assert state[0] == tank.level_initial  # level
        assert state[1] == 100.0  # mdot_out initial guess

    def test_residual_shape(self, tank):
        """residual() should return array matching state length"""
        state = np.array([2.0, 100.0])  # level, mdot_out
        state_dot = np.array([0.0, 0.0])  # derivatives
        ports = tank.ports

        # Set inlet conditions
        ports['inlet'].mdot = 100.0

        residual = tank.residual(state, ports, t=0.0, state_dot=state_dot)

        assert isinstance(residual, np.ndarray)
        assert len(residual) == 2


class TestSimpleTankPhysics:
    """Test physical correctness of SimpleTank"""

    def test_steady_state_mass_balance(self):
        """At steady state (dlevel/dt=0), mdot_in should equal mdot_out"""
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)

        # Compute what mdot_out should be for this pressure drop
        from thermal_sim.properties.coolprop_wrapper import FluidProperties
        water = FluidProperties('Water')
        rho = water.density(2e5, 300.0)
        dP = 2e5 - 1e5  # 1 bar drop
        mdot_out_expected = tank.C_v * np.sqrt(rho * dP)

        # For steady state, set mdot_in = mdot_out
        mdot_in = mdot_out_expected
        level = 2.0

        state = np.array([level, mdot_out_expected])
        tank.ports['inlet'].mdot = mdot_in

        # For steady-state, pass state_dot=None
        residual = tank.residual(state, tank.ports, t=0.0, state_dot=None)

        # At steady state with matching flows, mass balance residual should be zero
        assert abs(residual[0]) < 1e-6, f"Steady-state mass balance: {residual[0]}"
        assert abs(residual[1]) < 1e-6, f"Outlet flow residual: {residual[1]}"

    def test_transient_level_rising(self):
        """If mdot_in > mdot_out, level should be rising (dlevel/dt > 0)"""
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)

        mdot_in = 150.0  # More flow in
        mdot_out = 100.0  # Less flow out
        level = 2.0

        state = np.array([level, mdot_out])
        state_dot = np.zeros(2)  # Will be computed
        tank.ports['inlet'].mdot = mdot_in

        residual = tank.residual(state, tank.ports, t=0.0, state_dot=state_dot)

        # Residual form: dlevel/dt - state_dot[0] = 0
        # So: dlevel/dt = residual[0] + state_dot[0]
        # Since state_dot[0] = 0, dlevel/dt = residual[0]

        # Actually, let me recalculate correctly:
        # eq_mass_balance = dlevel_dt - state_dot[0]
        # So: dlevel_dt = eq_mass_balance + state_dot[0] = residual[0] + 0

        from thermal_sim.properties.coolprop_wrapper import FluidProperties
        water = FluidProperties('Water')
        rho = water.density(2e5, 300.0)
        dlevel_dt_expected = (mdot_in - mdot_out) / (rho * tank.A)

        # The residual should be dlevel_dt_expected - 0 = dlevel_dt_expected
        assert residual[0] > 0, f"Level should be rising: dlevel/dt = {residual[0]}"

    def test_outlet_flow_equation(self):
        """Outlet flow should follow mdot = C_v * sqrt(rho * dP)"""
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)

        # Calculate expected outlet flow
        from thermal_sim.properties.coolprop_wrapper import FluidProperties
        water = FluidProperties('Water')
        rho = water.density(2e5, 300.0)
        dP = 2e5 - 1e5
        mdot_expected = tank.C_v * np.sqrt(rho * dP)

        state = np.array([2.0, mdot_expected])
        state_dot = np.zeros(2)
        tank.ports['inlet'].mdot = mdot_expected

        residual = tank.residual(state, tank.ports, t=0.0, state_dot=state_dot)

        # Algebraic equation residual should be near zero
        assert abs(residual[1]) < 1e-10, f"Outlet flow residual: {residual[1]}"


class TestSimpleTankDAEAssembly:
    """Test that SimpleTank works with ThermalGraph.assemble_dae()"""

    def test_assemble_dae_identifies_differential_variable(self):
        """assemble_dae() should correctly identify level as differential"""
        graph = ThermalGraph()
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)
        graph.add_component(tank)

        residual_func, y0, ydot0, algebraic_vars = graph.assemble_dae()

        # Should have 2 variables: 1 differential (level), 1 algebraic (mdot_out)
        assert len(algebraic_vars) == 2
        assert algebraic_vars[0] == False, "level should be differential (False)"
        assert algebraic_vars[1] == True, "mdot_out should be algebraic (True)"

    def test_assemble_dae_initial_conditions(self):
        """assemble_dae() should return correct initial conditions"""
        graph = ThermalGraph()
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01, level_initial=3.5)
        graph.add_component(tank)

        residual_func, y0, ydot0, algebraic_vars = graph.assemble_dae()

        assert len(y0) == 2
        assert y0[0] == 3.5, "Initial level should be 3.5"
        assert y0[1] == 100.0, "Initial mdot_out guess should be 100.0"

        # Initial derivatives should be zeros
        assert np.all(ydot0 == 0.0)

    def test_dae_residual_function_callable(self):
        """DAE residual function should be callable with (t, y, ydot)"""
        graph = ThermalGraph()
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)
        graph.add_component(tank)

        residual_func, y0, ydot0, algebraic_vars = graph.assemble_dae()

        # Should be able to call with 3 arguments
        ydot_test = np.array([0.1, 0.0])  # Some derivative
        residual = residual_func(0.0, y0, ydot_test)

        assert isinstance(residual, np.ndarray)
        assert len(residual) == 2


class TestSimpleTankBackwardCompatibility:
    """Test that SimpleTank still works with old API"""

    def test_residual_works_without_state_dot(self):
        """residual() should work when called without state_dot (backward compat)"""
        tank = SimpleTank('tank', A=10.0, P=2e5, P_downstream=1e5, C_v=0.01)

        state = np.array([2.0, 100.0])
        tank.ports['inlet'].mdot = 100.0

        # Call without state_dot (old API)
        residual = tank.residual(state, tank.ports, t=0.0)

        assert isinstance(residual, np.ndarray)
        assert len(residual) == 2

        # Should behave as steady-state (dlevel/dt = 0 → mdot_in = mdot_out)
