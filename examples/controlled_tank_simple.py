"""
Simple controlled tank example (Phase 3 Milestone 5 demonstration).

This example demonstrates a working control loop with manual connection:
- Tank with level dynamics
- Control valve regulating flow
- PID controller maintaining level at setpoint

This is a simplified version showing the core Phase 3 capabilities before
full ScalarPort auto-connection is implemented.
"""

import numpy as np
import time

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.tank import SimpleTank
from thermal_sim.components.control_valve import ControlValve
from thermal_sim.components.pid_controller import PIDController
from thermal_sim.properties.coolprop_wrapper import FluidProperties


def main():
    print("=" * 70)
    print("CONTROLLED TANK LEVEL SYSTEM - TRANSIENT SIMULATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # System Parameters
    # -------------------------------------------------------------------------

    tank_area = 20.0         # m² cross-sectional area
    tank_P = 2e5             # 2 bar tank pressure
    tank_P_downstream = 1e5  # 1 bar downstream
    tank_Cv = 0.015          # Tank outlet valve coefficient
    level_initial = 1.5      # m (start below setpoint)
    level_setpoint = 2.5     # m (target)

    valve_Cv_max = 0.025     # Inlet valve sizing (larger than tank outlet)
    valve_tau = 3.0          # seconds (valve response time)

    pid_Kp = 0.5             # Proportional gain
    pid_Ki = 0.05            # Integral gain
    pid_Kd = 1.0             # Derivative gain

    print(f"\nSystem Configuration:")
    print(f"  Tank area: {tank_area:.1f} m²")
    print(f"  Tank pressure: {tank_P/1e5:.1f} bar")
    print(f"  Initial level: {level_initial:.1f} m")
    print(f"  Setpoint: {level_setpoint:.1f} m")
    print(f"  Valve Cv_max: {valve_Cv_max:.3f}, τ={valve_tau:.1f}s")
    print(f"  PID gains: Kp={pid_Kp}, Ki={pid_Ki}, Kd={pid_Kd}")

    # -------------------------------------------------------------------------
    # Build System
    # -------------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("Building components...")
    print("-" * 70)

    graph = ThermalGraph()

    # Tank with level dynamics
    tank = SimpleTank(
        'tank',
        A=tank_area,
        P=tank_P,
        P_downstream=tank_P_downstream,
        C_v=tank_Cv,
        level_initial=level_initial
    )
    print(f"✓ Tank: A={tank_area} m², P={tank_P/1e5:.1f} bar")

    # Control valve on tank inlet
    valve = ControlValve(
        'inlet_valve',
        Cv_max=valve_Cv_max,
        tau=valve_tau,
        position_initial=0.5  # Start half-open
    )
    print(f"✓ Valve: Cv_max={valve_Cv_max}, τ={valve_tau}s")

    # PID controller
    pid = PIDController(
        'level_controller',
        Kp=pid_Kp,
        Ki=pid_Ki,
        Kd=pid_Kd,
        setpoint=level_setpoint
    )
    print(f"✓ PID: setpoint={level_setpoint} m")

    graph.add_component(tank)
    graph.add_component(valve)
    graph.add_component(pid)

    print(f"\n✓ Total: {len(graph.components)} components")

    # -------------------------------------------------------------------------
    # Initial Conditions
    # -------------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("Setting initial conditions...")
    print("-" * 70)

    # Compute steady-state flows
    water = FluidProperties('Water')
    rho = water.density(tank_P, 300.0)
    dP = tank_P - tank_P_downstream
    mdot_out_ss = tank_Cv * np.sqrt(rho * dP)

    # Start with slightly lower inlet flow to create initial imbalance
    # This will cause level to drop, triggering controller response
    mdot_in_initial = mdot_out_ss * 0.9  # 10% less than outlet

    # Set port conditions
    valve.ports['inlet'].P = tank_P * 1.1  # Slight upstream pressure
    valve.ports['inlet'].h = 419e3  # Liquid water enthalpy
    valve.ports['outlet'].P = tank_P
    valve.ports['command'].value = 0.5  # Initial command

    tank.ports['inlet'].mdot = mdot_in_initial
    tank.ports['inlet'].P = tank_P
    tank.ports['inlet'].h = 419e3

    print(f"Initial steady-state flow: {mdot_out_ss:.2f} kg/s")

    # -------------------------------------------------------------------------
    # Assemble DAE and Prepare State
    # -------------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("Assembling DAE system...")
    print("-" * 70)

    residual_func, y0, ydot0, algebraic_vars = graph.assemble_dae()

    n_diff = sum(not alg for alg in algebraic_vars)
    n_alg = sum(algebraic_vars)
    print(f"State vector: {len(y0)} variables ({n_diff} differential, {n_alg} algebraic)")

    # Get component offsets
    tank_offset = graph._component_offsets['tank']
    valve_offset = graph._component_offsets['inlet_valve']
    pid_offset = graph._component_offsets['level_controller']

    # Override initial states
    y0[tank_offset] = level_initial  # Tank level
    y0[tank_offset + 1] = mdot_out_ss  # Tank outlet flow

    y0[valve_offset] = 0.5  # Valve position (half open)
    y0[valve_offset + 1] = mdot_in_initial  # Valve flow

    y0[pid_offset] = 0.0  # PID integral
    y0[pid_offset + 1] = 0.0  # PID derivative
    y0[pid_offset + 2] = 0.5  # PID output (mid-range)

    print(f"Initial state set:")
    print(f"  Tank level: {y0[tank_offset]:.2f} m")
    print(f"  Valve position: {y0[valve_offset]:.2f}")
    print(f"  PID output: {y0[pid_offset + 2]:.2f}")

    # -------------------------------------------------------------------------
    # Manual Control Loop Closure
    # -------------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("Creating closed-loop residual function...")
    print("-" * 70)

    # Wrap the residual function to close the control loop manually
    def closed_loop_residual(t, y, ydot):
        # Extract states for each component
        tank_level = y[tank_offset]
        tank_mdot_out = y[tank_offset + 1]

        valve_position = y[valve_offset]
        valve_mdot = y[valve_offset + 1]

        pid_integral = y[pid_offset]
        pid_derivative = y[pid_offset + 1]
        pid_output_signal = y[pid_offset + 2]

        # Step 1: Set PID measurement from tank level
        pid.ports['measurement'].value = tank_level

        # Step 2: Evaluate PID residual (this updates pid.ports['output'])
        pid_state = np.array([pid_integral, pid_derivative, pid_output_signal])
        pid_state_dot = ydot[pid_offset:pid_offset+3]
        pid_residual = pid.residual(pid_state, pid.ports, t, pid_state_dot)

        # Step 3: Transfer PID output to valve command
        valve.ports['command'].value = pid.ports['output'].value

        # Step 4: Evaluate valve residual (this updates valve.ports['outlet'].mdot)
        valve_state = np.array([valve_position, valve_mdot])
        valve_state_dot = ydot[valve_offset:valve_offset+2]
        valve_residual = valve.residual(valve_state, valve.ports, t, valve_state_dot)

        # Step 5: Transfer valve outlet flow to tank inlet
        tank.ports['inlet'].mdot = valve.ports['outlet'].mdot

        # Step 6: Evaluate tank residual
        tank_state = np.array([tank_level, tank_mdot_out])
        tank_state_dot = ydot[tank_offset:tank_offset+2]
        tank_residual = tank.residual(tank_state, tank.ports, t, tank_state_dot)

        # Assemble full residual vector in correct order
        residuals = []
        for comp in [tank, valve, pid]:
            offset = graph._component_offsets[comp.name]
            size = len(comp.get_variables())
            if comp.name == 'tank':
                residuals.append(tank_residual)
            elif comp.name == 'inlet_valve':
                residuals.append(valve_residual)
            elif comp.name == 'level_controller':
                residuals.append(pid_residual)

        return np.concatenate(residuals)

    print(f"✓ Control loop closed:")
    print(f"  Tank level → PID measurement")
    print(f"  PID output → Valve command")

    # -------------------------------------------------------------------------
    # Transient Simulation
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING TRANSIENT SIMULATION")
    print("=" * 70)

    t_end = 200.0  # 200 seconds
    n_points = 201
    t_points = np.linspace(0, t_end, n_points)

    print(f"Time span: 0 to {t_end:.0f} seconds")
    print(f"Output points: {n_points}")

    print(f"\nStarting integration...")
    start_time = time.time()

    # We need to use scipy's solve_ivp directly with our closed-loop residual
    # since solve_transient uses the graph's residual function
    from scipy.integrate import solve_ivp

    # Convert DAE to ODE form (same as in graph.py)
    diff_indices = [i for i, is_alg in enumerate(algebraic_vars) if not is_alg]
    alg_indices = [i for i, is_alg in enumerate(algebraic_vars) if is_alg]

    def ode_func(t, y):
        n_vars = len(y)
        ydot = np.zeros(n_vars)

        # Solve algebraic variables
        if alg_indices:
            def alg_residual(y_alg_vals):
                y_temp = y.copy()
                y_temp[alg_indices] = y_alg_vals
                res = closed_loop_residual(t, y_temp, np.zeros(n_vars))
                return res[alg_indices]

            from scipy.optimize import fsolve
            y_alg_solution = fsolve(alg_residual, y[alg_indices])
            y_new = y.copy()
            y_new[alg_indices] = y_alg_solution
        else:
            y_new = y

        # Extract derivatives for differential variables
        res_zero = closed_loop_residual(t, y_new, np.zeros(n_vars))
        ydot[diff_indices] = res_zero[diff_indices]

        return ydot

    result = solve_ivp(
        ode_func,
        (0, t_end),
        y0,
        method='BDF',
        t_eval=t_points,
        rtol=1e-4,
        atol=1e-6
    )

    elapsed_time = time.time() - start_time

    if result.success:
        print(f"✓ SUCCESS")
        print(f"  Wall time: {elapsed_time:.2f} s")
        print(f"  Simulation time: {t_end:.0f} s")
        print(f"  Speed: {t_end/elapsed_time:.1f}× real-time")
        print(f"  Time steps: {len(result.t)}")
    else:
        print(f"✗ FAILED: {result.message}")
        return

    # -------------------------------------------------------------------------
    # Extract Results
    # -------------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("Extracting results...")
    print("-" * 70)

    # Extract state histories
    level_history = result.y[tank_offset, :]
    tank_mdot_out = result.y[tank_offset + 1, :]

    valve_position = result.y[valve_offset, :]
    valve_mdot = result.y[valve_offset + 1, :]

    pid_integral = result.y[pid_offset, :]
    pid_output = result.y[pid_offset + 2, :]

    print(f"✓ Results extracted")

    # -------------------------------------------------------------------------
    # Results Summary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nInitial state (t=0s):")
    print(f"  Tank level: {level_history[0]:.2f} m")
    print(f"  Valve position: {valve_position[0]:.2%}")
    print(f"  PID output: {pid_output[0]:.2%}")

    print(f"\nFinal state (t={result.t[-1]:.0f}s):")
    print(f"  Tank level: {level_history[-1]:.2f} m")
    print(f"  Valve position: {valve_position[-1]:.2%}")
    print(f"  PID output: {pid_output[-1]:.2%}")

    # Check steady state
    dlevel_dt = (level_history[-1] - level_history[-10]) / (result.t[-1] - result.t[-10])
    print(f"\nFinal level rate: {dlevel_dt:.5f} m/s")

    if abs(dlevel_dt) < 1e-3:
        print(f"✓ Steady state reached")
    else:
        print(f"⚠ Still transient")

    # Control performance
    error_final = level_setpoint - level_history[-1]
    error_percent = abs(error_final / level_setpoint) * 100

    print(f"\nControl Performance:")
    print(f"  Setpoint: {level_setpoint:.2f} m")
    print(f"  Final level: {level_history[-1]:.2f} m")
    print(f"  Error: {error_final:.3f} m ({error_percent:.1f}%)")

    if error_percent < 2.0:
        print(f"✓ Control objective met (< 2%)")
    else:
        print(f"⚠ Error exceeds 2% target")

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    if HAS_MATPLOTLIB:
        print("\n" + "-" * 70)
        print("Generating plots...")
        print("-" * 70)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

        # Tank level
        ax1.plot(result.t, level_history, 'b-', linewidth=2, label='Actual')
        ax1.axhline(level_setpoint, color='r', linestyle='--', label='Setpoint')
        ax1.axhline(level_initial, color='g', linestyle=':', label='Initial')
        ax1.set_ylabel('Level [m]')
        ax1.set_title('Tank Level Control Response')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Valve position and PID output
        ax2.plot(result.t, valve_position * 100, 'b-', linewidth=2, label='Valve position')
        ax2.plot(result.t, pid_output * 100, 'r--', linewidth=1.5, label='PID output')
        ax2.set_ylabel('Position / Output [%]')
        ax2.set_title('Control Action')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Mass flows
        ax3.plot(result.t, valve_mdot, 'b-', linewidth=2, label='Inlet (valve)')
        ax3.plot(result.t, tank_mdot_out, 'r--', linewidth=1.5, label='Outlet (tank)')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Mass Flow [kg/s]')
        ax3.set_title('System Mass Flows')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        filename = 'controlled_tank_simple.png'
        plt.savefig(filename, dpi=150)
        print(f"✓ Saved: {filename}")
    else:
        print("\n(Matplotlib not available)")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    print(f"\n✓ Phase 3 Capabilities Demonstrated:")
    print(f"  • Tank level dynamics (differential equation)")
    print(f"  • Control valve with lag (differential equation)")
    print(f"  • PID feedback control (integral differential)")
    print(f"  • Closed-loop transient simulation")
    print(f"  • Performance: {t_end/elapsed_time:.1f}× real-time")

    if result.success and error_percent < 2.0:
        print(f"\n🎉 ALL OBJECTIVES MET!")


if __name__ == '__main__':
    main()
