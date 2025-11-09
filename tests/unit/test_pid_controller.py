"""Unit tests for PIDController component (Phase 3 Milestone 4)."""

import pytest
import numpy as np
from thermal_sim.components.pid_controller import PIDController
from thermal_sim.core.graph import ThermalGraph


class TestPIDControllerCreation:
    """Test PIDController initialization"""

    def test_creation_with_defaults(self):
        """PIDController should initialize with default parameters"""
        pid = PIDController('pid')

        assert pid.name == 'pid'
        assert pid.Kp == 1.0
        assert pid.Ki == 0.1
        assert pid.Kd == 0.01
        assert pid.setpoint == 0.0

    def test_creation_with_custom_parameters(self):
        """PIDController should accept custom parameters"""
        pid = PIDController('pid', Kp=2.0, Ki=0.5, Kd=0.1, setpoint=5.0)

        assert pid.Kp == 2.0
        assert pid.Ki == 0.5
        assert pid.Kd == 0.1
        assert pid.setpoint == 5.0

    def test_has_scalar_ports(self):
        """PIDController should have measurement and output ScalarPorts"""
        pid = PIDController('pid')

        assert 'measurement' in pid.ports
        assert 'output' in pid.ports
        assert pid.ports['measurement'].direction == 'in'
        assert pid.ports['output'].direction == 'out'

    def test_invalid_negative_gains(self):
        """PIDController should reject negative gains"""
        with pytest.raises(ValueError, match="gains must be non-negative"):
            PIDController('pid', Kp=-1.0)

        with pytest.raises(ValueError, match="gains must be non-negative"):
            PIDController('pid', Ki=-0.1)

        with pytest.raises(ValueError, match="gains must be non-negative"):
            PIDController('pid', Kd=-0.01)


class TestPIDControllerInterface:
    """Test Component interface implementation"""

    @pytest.fixture
    def pid(self):
        """Standard PID for tests"""
        return PIDController('pid', Kp=1.0, Ki=0.1, Kd=0.01, setpoint=2.0)

    def test_get_variables(self, pid):
        """get_variables() should return 1 differential + 2 algebraic variables"""
        variables = pid.get_variables()

        assert len(variables) == 3
        assert variables[0].name == 'integral'
        assert variables[0].kind == 'differential'
        assert variables[1].name == 'derivative'
        assert variables[1].kind == 'algebraic'
        assert variables[2].name == 'output_signal'
        assert variables[2].kind == 'algebraic'

    def test_get_initial_state(self, pid):
        """get_initial_state() should return zeros"""
        state = pid.get_initial_state()

        assert isinstance(state, np.ndarray)
        assert len(state) == 3
        assert np.all(state == 0.0)

    def test_residual_shape(self, pid):
        """residual() should return array matching state length"""
        state = np.array([0.0, 0.0, 0.5])  # integral, derivative, output
        state_dot = np.array([0.0, 0.0, 0.0])
        ports = pid.ports

        # Set measurement
        ports['measurement'].value = 1.5

        residual = pid.residual(state, ports, t=0.0, state_dot=state_dot)

        assert isinstance(residual, np.ndarray)
        assert len(residual) == 3


class TestPIDControllerPhysics:
    """Test physical correctness of PID controller"""

    def test_proportional_term(self):
        """Output should include Kp * error"""
        pid = PIDController('pid', Kp=2.0, Ki=0.0, Kd=0.0, setpoint=5.0)

        state = np.array([0.0, 0.0, 0.0])  # No integral, no derivative
        state_dot = np.zeros(3)
        pid.ports['measurement'].value = 3.0  # Measurement

        residual = pid.residual(state, pid.ports, t=0.0, state_dot=state_dot)

        # Error = setpoint - measurement = 5.0 - 3.0 = 2.0
        # Expected output = Kp * error = 2.0 * 2.0 = 4.0
        # eq_output = output_signal - (Kp*error + integral + derivative)
        # 0 = output_signal - 4.0
        # So residual[2] should be -4.0
        assert abs(residual[2] - (-4.0)) < 1e-10

    def test_integral_accumulation(self):
        """Integral should accumulate error over time"""
        pid = PIDController('pid', Kp=0.0, Ki=0.5, Kd=0.0, setpoint=10.0)

        # Start with some integral
        state = np.array([2.0, 0.0, 0.0])  # integral=2.0
        state_dot = np.zeros(3)
        pid.ports['measurement'].value = 8.0

        residual = pid.residual(state, pid.ports, t=0.0, state_dot=state_dot)

        # Error = 10.0 - 8.0 = 2.0
        # d(integral)/dt = Ki * error = 0.5 * 2.0 = 1.0
        # eq_integral = 1.0 - state_dot[0] = 1.0 - 0.0 = 1.0
        assert abs(residual[0] - 1.0) < 1e-10

    def test_derivative_term(self):
        """Derivative term should equal Kd * error (simplified)"""
        pid = PIDController('pid', Kp=0.0, Ki=0.0, Kd=0.2, setpoint=5.0)

        state = np.array([0.0, 0.0, 0.0])
        state_dot = np.zeros(3)
        pid.ports['measurement'].value = 3.0

        residual = pid.residual(state, pid.ports, t=0.0, state_dot=state_dot)

        # Error = 5.0 - 3.0 = 2.0
        # derivative = Kd * error = 0.2 * 2.0 = 0.4
        # eq_derivative = derivative - 0.4 = 0.0 - 0.4 = -0.4
        assert abs(residual[1] - (-0.4)) < 1e-10

    def test_full_pid_output(self):
        """Full PID output should be Kp*e + integral + derivative"""
        pid = PIDController('pid', Kp=1.0, Ki=0.5, Kd=0.1, setpoint=10.0)

        # State: integral=2.0, derivative=0.3 (would be computed by solver)
        state = np.array([2.0, 0.3, 0.0])
        state_dot = np.zeros(3)
        pid.ports['measurement'].value = 7.0

        residual = pid.residual(state, pid.ports, t=0.0, state_dot=state_dot)

        # Error = 10.0 - 7.0 = 3.0
        # derivative state = 0.3 (given in state vector)
        # Expected output = Kp*e + integral + derivative
        #                 = 1.0*3.0 + 2.0 + 0.3 = 5.3
        # eq_output = output_signal - 5.3
        # 0 = output_signal - 5.3
        # residual[2] should be -5.3
        assert abs(residual[2] - (-5.3)) < 1e-10

    def test_anti_windup_clipping(self):
        """Output should be clipped to [0, 1] range"""
        pid = PIDController('pid', Kp=10.0, Ki=0.0, Kd=0.0, setpoint=10.0)

        state = np.array([0.0, 0.0, 5.0])  # Large output signal
        state_dot = np.zeros(3)
        pid.ports['measurement'].value = 0.0

        pid.residual(state, pid.ports, t=0.0, state_dot=state_dot)

        # Output should be clipped to 1.0
        assert pid.ports['output'].value == 1.0

    def test_negative_error_handling(self):
        """Controller should handle negative errors correctly"""
        pid = PIDController('pid', Kp=1.0, Ki=0.0, Kd=0.0, setpoint=5.0)

        state = np.array([0.0, 0.0, 0.0])
        state_dot = np.zeros(3)
        pid.ports['measurement'].value = 7.0  # Above setpoint

        residual = pid.residual(state, pid.ports, t=0.0, state_dot=state_dot)

        # Error = 5.0 - 7.0 = -2.0
        # Output = Kp * error = 1.0 * (-2.0) = -2.0
        # eq_output = output_signal - (-2.0) = 0.0 - (-2.0) = 2.0
        assert abs(residual[2] - 2.0) < 1e-10

        # But actual output port should be clipped to [0, 1]
        assert pid.ports['output'].value == 0.0


class TestPIDControllerDAEAssembly:
    """Test that PIDController works with ThermalGraph.assemble_dae()"""

    def test_assemble_dae_identifies_differential_variable(self):
        """assemble_dae() should correctly identify integral as differential"""
        graph = ThermalGraph()
        pid = PIDController('pid', Kp=1.0, Ki=0.1, Kd=0.01, setpoint=2.0)
        graph.add_component(pid)

        residual_func, y0, ydot0, algebraic_vars = graph.assemble_dae()

        # Should have 3 variables: 1 differential (integral), 2 algebraic
        assert len(algebraic_vars) == 3
        assert algebraic_vars[0] == False  # integral is differential
        assert algebraic_vars[1] == True   # derivative is algebraic
        assert algebraic_vars[2] == True   # output_signal is algebraic

    def test_assemble_dae_initial_conditions(self):
        """assemble_dae() should return correct initial conditions"""
        graph = ThermalGraph()
        pid = PIDController('pid')
        graph.add_component(pid)

        residual_func, y0, ydot0, algebraic_vars = graph.assemble_dae()

        assert len(y0) == 3
        assert np.all(y0 == 0.0)  # All initial values are zero
        assert np.all(ydot0 == 0.0)

    def test_dae_residual_function_callable(self):
        """DAE residual function should be callable with (t, y, ydot)"""
        graph = ThermalGraph()
        pid = PIDController('pid', Kp=1.0, Ki=0.1, Kd=0.01, setpoint=2.0)
        graph.add_component(pid)

        # Set measurement value
        pid.ports['measurement'].value = 1.5

        residual_func, y0, ydot0, algebraic_vars = graph.assemble_dae()

        # Should be able to call with 3 arguments
        ydot_test = np.array([0.1, 0.0, 0.0])  # Some derivative
        residual = residual_func(0.0, y0, ydot_test)

        assert isinstance(residual, np.ndarray)
        assert len(residual) == 3


class TestPIDControllerDiagnostics:
    """Test diagnostic methods"""

    def test_get_error(self):
        """get_error() should return setpoint - measurement"""
        pid = PIDController('pid', setpoint=5.0)
        pid.ports['measurement'].value = 3.0

        error = pid.get_error()

        assert error == 2.0

    def test_get_output(self):
        """get_output() should return current output value"""
        pid = PIDController('pid')
        pid.ports['output'].value = 0.75

        output = pid.get_output()

        assert output == 0.75

    def test_repr(self):
        """__repr__ should show configuration"""
        pid = PIDController('level_pid', Kp=2.0, Ki=0.5, Kd=0.1, setpoint=3.5)

        repr_str = repr(pid)

        assert 'level_pid' in repr_str
        assert '2.00' in repr_str  # Kp
        assert '0.500' in repr_str  # Ki
        assert '0.100' in repr_str  # Kd
        assert '3.50' in repr_str  # setpoint


class TestPIDControllerBackwardCompatibility:
    """Test backward compatibility with old API"""

    def test_residual_works_without_state_dot(self):
        """residual() should work when called without state_dot"""
        pid = PIDController('pid', Kp=1.0, Ki=0.1, Kd=0.01, setpoint=2.0)

        state = np.array([0.0, 0.0, 0.0])
        pid.ports['measurement'].value = 1.5

        # Call without state_dot (old API for steady-state)
        residual = pid.residual(state, pid.ports, t=0.0)

        assert isinstance(residual, np.ndarray)
        assert len(residual) == 3
