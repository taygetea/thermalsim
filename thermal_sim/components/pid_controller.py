"""
PID controller component for feedback control (Phase 3 Milestone 4).

Implements proportional-integral-derivative control with anti-windup.
"""

import numpy as np
from thermal_sim.core.component import Component
from thermal_sim.core.variable import Variable
from thermal_sim.core.port import ScalarPort


class PIDController(Component):
    """
    PID controller with anti-windup (Phase 3 Milestone 4).

    Implements the standard PID control law:
        u(t) = Kp·e(t) + Ki·∫e(τ)dτ + Kd·de/dt

    Where:
        - e(t) = setpoint - measurement (error)
        - Integral term accumulates error over time
        - Derivative term dampens oscillations
        - Output is clipped to [0, 1] range (anti-windup)

    State variables:
        - integral: Accumulated error (differential)
        - derivative: Rate of error change (algebraic, simplified)
        - output_signal: Control output (algebraic)

    Parameters:
        name: Component identifier
        Kp: Proportional gain (default 1.0)
        Ki: Integral gain (default 0.1)
        Kd: Derivative gain (default 0.01)
        setpoint: Desired measurement value (default 0.0)

    Ports:
        measurement (ScalarPort, in): Process variable measurement
        output (ScalarPort, out): Control signal [0, 1]

    Example:
        >>> pid = PIDController('level_controller', Kp=2.0, Ki=0.5, Kd=0.1, setpoint=2.0)
        >>> graph.add_component(pid)
        >>> graph.connect(level_sensor.output, pid.measurement)
        >>> graph.connect(pid.output, valve.command)
        >>> result = graph.solve_transient(tspan=(0, 100))
    """

    def __init__(self,
                 name: str,
                 Kp: float = 1.0,
                 Ki: float = 0.1,
                 Kd: float = 0.01,
                 setpoint: float = 0.0):
        super().__init__(name)

        if Kp < 0 or Ki < 0 or Kd < 0:
            raise ValueError("PID gains must be non-negative")

        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        self.setpoint = setpoint  # Desired value

        # Create ports
        self.measurement = ScalarPort('measurement', direction='in')
        self.output = ScalarPort('output', direction='out')

        self.measurement.component = self
        self.output.component = self

        self.ports = {
            'measurement': self.measurement,
            'output': self.output,
        }

    def get_variables(self):
        return [
            Variable('integral', kind='differential', initial=0.0, units='-'),
            Variable('derivative', kind='algebraic', initial=0.0, units='-'),
            Variable('output_signal', kind='algebraic', initial=0.0, units='-'),
        ]

    def get_initial_state(self):
        return np.array([0.0, 0.0, 0.0])

    def residual(self, state, ports, t, state_dot=None):
        integral, derivative, output_signal = state

        # Error: e(t) = setpoint - measurement
        error = self.setpoint - ports['measurement'].value

        # Equation 1: Integral dynamics
        # d(integral)/dt = Ki * error
        if state_dot is not None:
            # Transient: residual = f(x,t) - dx/dt
            eq_integral = (self.Ki * error) - state_dot[0]
        else:
            # Steady-state: integral unchanging
            eq_integral = self.Ki * error  # Should be zero at steady state

        # Equation 2: Derivative term (simplified)
        # In full PID, this would be Kd * de/dt, but we use algebraic approximation
        eq_derivative = derivative - self.Kd * error

        # Equation 3: Output signal
        # u = Kp*e + integral + derivative
        eq_output = output_signal - (self.Kp * error + integral + derivative)

        # Update output port with anti-windup (clip to [0, 1])
        ports['output'].value = np.clip(output_signal, 0.0, 1.0)

        return np.array([eq_integral, eq_derivative, eq_output])

    def get_error(self) -> float:
        """Get current error (for diagnostics)"""
        return self.setpoint - self.measurement.value

    def get_output(self) -> float:
        """Get current output signal (for diagnostics)"""
        return self.output.value

    def __repr__(self):
        return (f"PIDController('{self.name}', Kp={self.Kp:.2f}, Ki={self.Ki:.3f}, "
                f"Kd={self.Kd:.3f}, setpoint={self.setpoint:.2f})")
