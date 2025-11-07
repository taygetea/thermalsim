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

    # Create components
    boiler = ConstantPressureHeater(
        name='boiler',
        P=P_high,
        Q=100e6,     # 100 MW heat input
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
        Q=-80e6,     # 80 MW heat rejection (negative = cooling)
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

    # Solve
    print("Solving Rankine cycle...")
    result = graph.solve(
        t_span=(0, 1000),  # Simulate 1000 seconds
        rtol=1e-6,
        atol=1e-8
    )

    if result.success:
        print(f"✓ Solution converged in {len(result.t)} time steps")

        # Extract results
        turbine_state = graph.get_component_state(result, 'turbine')
        pump_state = graph.get_component_state(result, 'pump')

        # Turbine state: [h_out, W_shaft, mdot]
        W_turbine = turbine_state[1, -1]  # Final turbine power
        W_pump = pump_state[1, -1]        # Final pump power

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
