"""
Simple Rankine cycle demonstration.

This example creates a basic steam power cycle:
    Boiler → Turbine → Condenser → Pump → (back to Boiler)

Expected result: ~30-35% thermal efficiency
"""

from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.components.pump import Pump
import numpy as np


def main():
    # Create system graph
    graph = ThermalGraph()

    # Define operating conditions
    P_high = 10e6   # 10 MPa (high pressure)
    P_low = 10e3    # 10 kPa (condenser pressure)

    # For a saturated Rankine cycle at these pressures:
    # h_liquid_low = 0.192 MJ/kg, h_vapor_high = 2.725 MJ/kg
    # Delta_h = 2.52 MJ/kg per kg/s
    # For mdot = 100 kg/s, Q_in ≈ 252 MW

    # Create components with matched heat transfer
    boiler = ConstantPressureHeater(
        name='boiler',
        P=P_high,
        Q=252e6,     # 252 MW heat input (matched to cycle)
        fluid='Water'
    )

    turbine = Turbine(
        name='turbine',
        efficiency=0.85,
        P_out=P_low,
        fluid='Water'
    )

    condenser = ConstantPressureHeater(
        name='condenser',
        P=P_low,
        Q=-173e6,    # 173 MW heat rejection (matched to cycle)
        fluid='Water'
    )

    pump = Pump(
        name='pump',
        efficiency=0.80,
        P_out=P_high,
        fluid='Water'
    )

    # Add components to graph
    for comp in [boiler, turbine, condenser, pump]:
        graph.add_component(comp)

    # Connect in cycle
    graph.connect(boiler.outlet, turbine.inlet)
    graph.connect(turbine.outlet, condenser.inlet)
    graph.connect(condenser.outlet, pump.inlet)
    graph.connect(pump.outlet, boiler.inlet)

    # Check topology
    warnings = graph.validate_topology()
    if warnings:
        print("Topology warnings:")
        for w in warnings:
            print(f"  {w}")

    # Solve for steady state
    # TODO_IDA Phase 2: Replace solve_sequential() with solve_steady_state() once SUNDIALS IDA is integrated
    # The sequential solver is a temporary workaround for the coupled algebraic system
    print("Solving Rankine cycle (using temporary sequential solver)...")
    result = graph.solve_sequential()  # TEMPORARY - will be solve_steady_state() with IDA

    if result.success:
        print(f"✓ Solution converged (residual norm: {np.linalg.norm(result.fun):.2e})")

        # Extract results
        turbine_state = graph.get_component_state(result, 'turbine')
        pump_state = graph.get_component_state(result, 'pump')

        # Turbine state: [h_out, W_shaft, mdot]
        W_turbine = turbine_state[1, 0]  # Steady-state turbine power
        W_pump = pump_state[1, 0]        # Steady-state pump power

        Q_in = boiler.Q
        W_net = W_turbine - W_pump

        efficiency = W_net / Q_in

        print(f"\nResults:")
        print(f"  Turbine power: {W_turbine/1e6:.2f} MW")
        print(f"  Pump power: {W_pump/1e6:.2f} MW")
        print(f"  Net power: {W_net/1e6:.2f} MW")
        print(f"  Thermal efficiency: {efficiency*100:.1f}%")

        # Validate
        if 0.30 < efficiency < 0.40:
            print("✓ Efficiency in expected range (30-40%)")
        else:
            print(f"⚠ Efficiency {efficiency*100:.1f}% outside expected range")
    else:
        print(f"✗ Solver failed: {result.message}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
