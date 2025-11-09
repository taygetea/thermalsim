"""
Tank filling example - demonstrates transient dynamics (Phase 3 Milestone 3).

A simple storage tank with:
- Constant inlet flow
- Outlet flow governed by pressure drop
- Level rises over time until steady-state is reached

This example shows the first use of solve_transient() with differential equations!
"""

import numpy as np
from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.tank import SimpleTank

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def main():
    print("=" * 60)
    print("TANK FILLING SIMULATION")
    print("=" * 60)

    # Create system
    graph = ThermalGraph()

    # Tank parameters
    A = 10.0  # m² cross-sectional area
    P_tank = 2e5  # Pa (2 bar)
    P_downstream = 1e5  # Pa (1 bar)
    C_v = 0.01  # Valve coefficient
    level_initial = 1.0  # m

    tank = SimpleTank(
        'tank',
        A=A,
        P=P_tank,
        P_downstream=P_downstream,
        C_v=C_v,
        level_initial=level_initial
    )

    graph.add_component(tank)

    print(f"\nTank Configuration:")
    print(f"  Area: {A} m²")
    print(f"  Tank pressure: {P_tank/1e5:.1f} bar")
    print(f"  Downstream pressure: {P_downstream/1e5:.1f} bar")
    print(f"  Valve coefficient: {C_v} kg/(s·√Pa)")
    print(f"  Initial level: {level_initial} m")

    # Compute steady-state outlet flow first
    from thermal_sim.properties.coolprop_wrapper import FluidProperties
    water = FluidProperties('Water')
    rho = water.density(P_tank, 300.0)
    dP = P_tank - P_downstream
    mdot_out_ss = C_v * np.sqrt(rho * dP)

    # Set inlet flow higher than steady-state outlet to fill the tank
    mdot_in = mdot_out_ss * 1.2  # 20% more than outlet
    tank.ports['inlet'].mdot = mdot_in

    print(f"\nInlet flow: {mdot_in:.2f} kg/s (constant)")
    print(f"Steady-state outlet flow: {mdot_out_ss:.2f} kg/s")
    print(f"Net accumulation: {mdot_in - mdot_out_ss:.2f} kg/s")

    if mdot_in > mdot_out_ss:
        print("→ Tank will fill (level rises)")
    elif mdot_in < mdot_out_ss:
        print("→ Tank will drain (level falls)")
    else:
        print("→ Tank at equilibrium (no level change)")

    # Solve transient
    print("\n" + "-" * 60)
    print("Running transient simulation...")
    print("-" * 60)

    t_end = 200.0  # seconds
    t_points = np.linspace(0, t_end, 201)

    # Need to provide custom y0 since we don't have full steady-state
    y0 = tank.get_initial_state()
    y0[1] = mdot_out_ss  # Set mdot_out to steady-state value

    result = graph.solve_transient(
        tspan=t_points,
        y0=y0,
        rtol=1e-6,
        atol=1e-8
    )

    print(f"Solver status: {'SUCCESS' if result.success else 'FAILED'}")
    if not result.success:
        print(f"Message: {result.message}")
        return

    print(f"Time steps: {len(result.t)}")
    print(f"Integration time: 0 to {result.t[-1]:.1f} seconds")

    # Extract results
    tank_state = graph.get_component_state(result, 'tank')
    level_history = tank_state[0, :]  # Level is first state variable
    mdot_out_history = tank_state[1, :]  # Outlet flow is second

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nInitial state:")
    print(f"  Level: {level_history[0]:.2f} m")
    print(f"  Outlet flow: {mdot_out_history[0]:.2f} kg/s")

    print(f"\nFinal state (t={result.t[-1]:.1f}s):")
    print(f"  Level: {level_history[-1]:.2f} m")
    print(f"  Outlet flow: {mdot_out_history[-1]:.2f} kg/s")

    # Check if approaching steady state
    dlevel_dt = (level_history[-1] - level_history[-10]) / (result.t[-1] - result.t[-10])
    print(f"\nFinal rate of change: {dlevel_dt:.4f} m/s")

    if abs(dlevel_dt) < 1e-3:
        print("✓ System has reached steady state")
    else:
        print("⚠ System still transient (increase simulation time)")

    # Plot results (if matplotlib available)
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot level
        ax1.plot(result.t, level_history, 'b-', linewidth=2)
        ax1.axhline(level_initial, color='g', linestyle='--', label='Initial level')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Level [m]')
        ax1.set_title('Tank Level vs Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot flows
        ax2.plot(result.t, np.full_like(result.t, mdot_in), 'g-', linewidth=2, label='Inlet flow')
        ax2.plot(result.t, mdot_out_history, 'r-', linewidth=2, label='Outlet flow')
        ax2.axhline(mdot_out_ss, color='b', linestyle='--', label='SS outlet flow')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Mass flow [kg/s]')
        ax2.set_title('Mass Flows vs Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plot_filename = 'tank_filling.png'
        plt.savefig(plot_filename, dpi=150)
        print(f"✓ Plot saved to {plot_filename}")
    else:
        print("\n(Matplotlib not available - skipping plots)")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
