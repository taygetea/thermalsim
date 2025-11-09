"""Unit tests for ControlValve component (Phase 3 Milestone 4)."""

import pytest
import numpy as np
from thermal_sim.components.control_valve import ControlValve
from thermal_sim.core.graph import ThermalGraph


class TestControlValveCreation:
    """Test ControlValve initialization"""

    def test_creation_with_required_parameters(self):
        """ControlValve should initialize with required Cv_max"""
        valve = ControlValve('valve', Cv_max=0.02)

        assert valve.name == 'valve'
        assert valve.Cv_max == 0.02
        assert valve.tau == 5.0  # Default
        assert valve.position_initial == 0.5  # Default

    def test_creation_with_custom_parameters(self):
        """ControlValve should accept custom parameters"""
        valve = ControlValve('valve', Cv_max=0.01, tau=3.0, position_initial=0.75)

        assert valve.Cv_max == 0.01
        assert valve.tau == 3.0
        assert valve.position_initial == 0.75

    def test_has_mass_flow_and_scalar_ports(self):
        """ControlValve should have inlet, outlet (MassFlow) and command (Scalar)"""
        valve = ControlValve('valve', Cv_max=0.02)

        assert 'inlet' in valve.ports
        assert 'outlet' in valve.ports
        assert 'command' in valve.ports
        assert valve.ports['inlet'].direction == 'in'
        assert valve.ports['outlet'].direction == 'out'
        assert valve.ports['command'].direction == 'in'

    def test_invalid_cv_max(self):
        """ControlValve should reject non-positive Cv_max"""
        with pytest.raises(ValueError, match="Cv_max must be positive"):
            ControlValve('valve', Cv_max=0.0)

        with pytest.raises(ValueError, match="Cv_max must be positive"):
            ControlValve('valve', Cv_max=-0.01)

    def test_invalid_tau(self):
        """ControlValve should reject non-positive time constant"""
        with pytest.raises(ValueError, match="tau must be positive"):
            ControlValve('valve', Cv_max=0.02, tau=0.0)

        with pytest.raises(ValueError, match="tau must be positive"):
            ControlValve('valve', Cv_max=0.02, tau=-1.0)

    def test_invalid_initial_position(self):
        """ControlValve should reject position outside [0, 1]"""
        with pytest.raises(ValueError, match="Initial position must be in"):
            ControlValve('valve', Cv_max=0.02, position_initial=-0.1)

        with pytest.raises(ValueError, match="Initial position must be in"):
            ControlValve('valve', Cv_max=0.02, position_initial=1.5)


class TestControlValveInterface:
    """Test Component interface implementation"""

    @pytest.fixture
    def valve(self):
        """Standard valve for tests"""
        return ControlValve('valve', Cv_max=0.02, tau=5.0)

    def test_get_variables(self, valve):
        """get_variables() should return 1 differential + 1 algebraic variable"""
        variables = valve.get_variables()

        assert len(variables) == 2
        assert variables[0].name == 'position'
        assert variables[0].kind == 'differential'
        assert variables[1].name == 'mdot'
        assert variables[1].kind == 'algebraic'

    def test_get_initial_state(self, valve):
        """get_initial_state() should return [position_initial, mdot_guess]"""
        state = valve.get_initial_state()

        assert isinstance(state, np.ndarray)
        assert len(state) == 2
        assert state[0] == valve.position_initial
        assert state[1] == 100.0  # Initial mdot guess

    def test_residual_shape(self, valve):
        """residual() should return array matching state length"""
        state = np.array([0.5, 100.0])  # position, mdot
        state_dot = np.array([0.0, 0.0])
        ports = valve.ports

        # Set inlet conditions
        ports['inlet'].h = 500e3
        ports['inlet'].P = 2e5
        ports['command'].value = 0.5

        residual = valve.residual(state, ports, t=0.0, state_dot=state_dot)

        assert isinstance(residual, np.ndarray)
        assert len(residual) == 2


class TestControlValvePhysics:
    """Test physical correctness of control valve"""

    def test_steady_state_position_equals_command(self):
        """At steady state, position should equal command"""
        valve = ControlValve('valve', Cv_max=0.02, tau=5.0)

        # Command = 0.7, position = 0.7 (steady state)
        state = np.array([0.7, 100.0])
        valve.ports['command'].value = 0.7
        valve.ports['inlet'].h = 500e3
        valve.ports['inlet'].P = 2e5

        # Steady-state call (state_dot = None)
        residual = valve.residual(state, valve.ports, t=0.0, state_dot=None)

        # eq_position = command - position = 0.7 - 0.7 = 0
        assert abs(residual[0]) < 1e-10

    def test_transient_position_dynamics(self):
        """Position should respond with first-order lag"""
        valve = ControlValve('valve', Cv_max=0.02, tau=5.0)

        # Command = 1.0, current position = 0.5
        state = np.array([0.5, 100.0])
        state_dot = np.zeros(2)
        valve.ports['command'].value = 1.0
        valve.ports['inlet'].h = 500e3
        valve.ports['inlet'].P = 2e5

        residual = valve.residual(state, valve.ports, t=0.0, state_dot=state_dot)

        # d(position)/dt = (command - position) / tau
        #                = (1.0 - 0.5) / 5.0 = 0.1
        # eq_position = 0.1 - state_dot[0] = 0.1 - 0.0 = 0.1
        assert abs(residual[0] - 0.1) < 1e-10

    def test_flow_modulated_by_position(self):
        """Flow should be proportional to position"""
        valve = ControlValve('valve', Cv_max=0.02, tau=5.0)

        # Test with position = 0.5 (half open)
        state = np.array([0.5, 0.0])  # position=0.5, mdot will be computed
        state_dot = np.zeros(2)
        valve.ports['command'].value = 0.5
        valve.ports['inlet'].h = 500e3
        valve.ports['inlet'].P = 2e5
        valve.ports['outlet'].P = 1e5

        residual = valve.residual(state, valve.ports, t=0.0, state_dot=state_dot)

        # Cv_effective = Cv_max * position = 0.02 * 0.5 = 0.01
        # dP = 2e5 - 1e5 = 1e5
        # rho ≈ 1000 kg/m³ (water)
        # mdot = 0.01 * sqrt(1000 * 1e5) = 0.01 * sqrt(1e8) = 0.01 * 10000 = 100
        # eq_mdot = mdot - 100 = 0 - 100 = -100
        # So residual[1] should be approximately -100

        # Allow some tolerance due to CoolProp density calculation
        assert abs(residual[1] - (-100)) < 10  # Within 10%

    def test_flow_zero_when_closed(self):
        """Flow should be zero when valve is fully closed"""
        valve = ControlValve('valve', Cv_max=0.02)

        # Fully closed: position = 0
        state = np.array([0.0, 0.0])
        state_dot = np.zeros(2)
        valve.ports['command'].value = 0.0
        valve.ports['inlet'].h = 500e3
        valve.ports['inlet'].P = 2e5
        valve.ports['outlet'].P = 1e5

        residual = valve.residual(state, valve.ports, t=0.0, state_dot=state_dot)

        # Cv_effective = 0.02 * 0.0 = 0
        # mdot = 0
        # eq_mdot = 0 - 0 = 0
        assert abs(residual[1]) < 1e-10

    def test_isenthalpic_expansion(self):
        """Outlet enthalpy should equal inlet enthalpy"""
        valve = ControlValve('valve', Cv_max=0.02)

        state = np.array([0.5, 100.0])
        state_dot = np.zeros(2)
        valve.ports['command'].value = 0.5
        valve.ports['inlet'].h = 750e3  # High enthalpy
        valve.ports['inlet'].P = 3e5

        valve.residual(state, valve.ports, t=0.0, state_dot=state_dot)

        # Outlet enthalpy should match inlet (isenthalpic)
        assert valve.ports['outlet'].h == valve.ports['inlet'].h

    def test_command_saturation(self):
        """Command should be saturated to [0, 1] range"""
        valve = ControlValve('valve', Cv_max=0.02, tau=5.0)

        # Command > 1.0 should be clipped
        state = np.array([0.5, 100.0])
        state_dot = np.zeros(2)
        valve.ports['command'].value = 1.5  # Out of range
        valve.ports['inlet'].h = 500e3
        valve.ports['inlet'].P = 2e5

        residual = valve.residual(state, valve.ports, t=0.0, state_dot=state_dot)

        # Command is clipped to 1.0
        # d(position)/dt = (1.0 - 0.5) / 5.0 = 0.1
        assert abs(residual[0] - 0.1) < 1e-10


class TestControlValveDAEAssembly:
    """Test that ControlValve works with ThermalGraph.assemble_dae()"""

    def test_assemble_dae_identifies_differential_variable(self):
        """assemble_dae() should correctly identify position as differential"""
        graph = ThermalGraph()
        valve = ControlValve('valve', Cv_max=0.02)
        graph.add_component(valve)

        residual_func, y0, ydot0, algebraic_vars = graph.assemble_dae()

        # Should have 2 variables: 1 differential (position), 1 algebraic (mdot)
        assert len(algebraic_vars) == 2
        assert algebraic_vars[0] == False  # position is differential
        assert algebraic_vars[1] == True   # mdot is algebraic

    def test_assemble_dae_initial_conditions(self):
        """assemble_dae() should return correct initial conditions"""
        graph = ThermalGraph()
        valve = ControlValve('valve', Cv_max=0.02, position_initial=0.75)
        graph.add_component(valve)

        residual_func, y0, ydot0, algebraic_vars = graph.assemble_dae()

        assert len(y0) == 2
        assert y0[0] == 0.75  # Initial position
        assert y0[1] == 100.0  # Initial mdot guess

    def test_dae_residual_function_callable(self):
        """DAE residual function should be callable with (t, y, ydot)"""
        graph = ThermalGraph()
        valve = ControlValve('valve', Cv_max=0.02)
        graph.add_component(valve)

        residual_func, y0, ydot0, algebraic_vars = graph.assemble_dae()

        # Set inlet conditions and command
        valve.ports['inlet'].h = 500e3
        valve.ports['inlet'].P = 2e5
        valve.ports['command'].value = 0.5

        # Should be able to call with 3 arguments
        ydot_test = np.array([0.1, 0.0])  # Some derivative
        residual = residual_func(0.0, y0, ydot_test)

        assert isinstance(residual, np.ndarray)
        assert len(residual) == 2


class TestControlValveDiagnostics:
    """Test diagnostic methods"""

    def test_get_position(self):
        """get_position() should return initial position"""
        valve = ControlValve('valve', Cv_max=0.02, position_initial=0.6)

        position = valve.get_position()

        assert position == 0.6

    def test_get_flow(self):
        """get_flow() should return outlet mass flow"""
        valve = ControlValve('valve', Cv_max=0.02)
        valve.ports['outlet'].mdot = 125.0

        flow = valve.get_flow()

        assert flow == 125.0

    def test_repr(self):
        """__repr__ should show configuration"""
        valve = ControlValve('inlet_valve', Cv_max=0.015, tau=3.0, position_initial=0.8)

        repr_str = repr(valve)

        assert 'inlet_valve' in repr_str
        assert '0.015' in repr_str  # Cv_max
        assert '3.0' in repr_str    # tau
        assert '0.80' in repr_str   # position


class TestControlValveBackwardCompatibility:
    """Test backward compatibility with old API"""

    def test_residual_works_without_state_dot(self):
        """residual() should work when called without state_dot"""
        valve = ControlValve('valve', Cv_max=0.02)

        state = np.array([0.5, 100.0])
        valve.ports['command'].value = 0.5
        valve.ports['inlet'].h = 500e3
        valve.ports['inlet'].P = 2e5

        # Call without state_dot (old API for steady-state)
        residual = valve.residual(state, valve.ports, t=0.0)

        assert isinstance(residual, np.ndarray)
        assert len(residual) == 2


class TestControlValveEdgeCases:
    """Test edge cases and error handling"""

    def test_negative_pressure_drop(self):
        """Valve should handle negative pressure drop gracefully"""
        valve = ControlValve('valve', Cv_max=0.02)

        state = np.array([0.5, 0.0])
        state_dot = np.zeros(2)
        valve.ports['command'].value = 0.5
        valve.ports['inlet'].h = 500e3
        valve.ports['inlet'].P = 1e5
        valve.ports['outlet'].P = 2e5  # Higher than inlet!

        # Should not raise error, flow should be zero
        residual = valve.residual(state, valve.ports, t=0.0, state_dot=state_dot)

        # dP = max(1e5 - 2e5, 0) = 0
        # mdot = 0
        assert abs(residual[1]) < 1e-10

    def test_negative_position_prevented(self):
        """Negative position should be treated as zero flow"""
        valve = ControlValve('valve', Cv_max=0.02)

        # Artificially set negative position (shouldn't happen in practice)
        state = np.array([-0.1, 0.0])
        state_dot = np.zeros(2)
        valve.ports['command'].value = 0.0
        valve.ports['inlet'].h = 500e3
        valve.ports['inlet'].P = 2e5
        valve.ports['outlet'].P = 1e5

        # Should handle gracefully
        residual = valve.residual(state, valve.ports, t=0.0, state_dot=state_dot)

        # Flow should be zero (negative position clamped to 0)
        assert abs(residual[1]) < 1e-10
