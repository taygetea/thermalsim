"""
Solve Rankine cycle using root finding (for pure algebraic systems).
"""

from thermal_sim.core.graph import ThermalGraph
from thermal_sim.components.heater import ConstantPressureHeater
from thermal_sim.components.turbine import Turbine
from thermal_sim.components.pump import Pump
from scipy.optimize import root
import numpy as np

# Create system graph
graph = ThermalGraph()

# Define operating conditions
P_high = 10e6
P_low = 10e3

# Create components
boiler = ConstantPressureHeater('boiler', P=P_high, Q=100e6)
turbine = Turbine('turbine', efficiency=0.85, P_out=P_low)
condenser = ConstantPressureHeater('condenser', P=P_low, Q=-80e6)
pump = Pump('pump', efficiency=0.80, P_out=P_high)

# Add components to graph
for comp in [boiler, turbine, condenser, pump]:
    graph.add_component(comp)

# Connect in cycle
graph.connect(boiler.outlet, turbine.inlet)
graph.connect(turbine.outlet, condenser.inlet)
graph.connect(condenser.outlet, pump.inlet)
graph.connect(pump.outlet, boiler.inlet)

# Assemble
residual_func, y0 = graph.assemble()

# Solve using root finding
print("Solving Rankine cycle using root finding...")
result = root(lambda y: residual_func(0, y), y0, method='hybr')

if result.success:
    print("✓ Solution converged!")

    # Extract component states
    boiler_state = result.x[graph._component_offsets['boiler']:graph._component_offsets['boiler']+2]
    turbine_state = result.x[graph._component_offsets['turbine']:graph._component_offsets['turbine']+3]
    condenser_state = result.x[graph._component_offsets['condenser']:graph._component_offsets['condenser']+2]
    pump_state = result.x[graph._component_offsets['pump']:graph._component_offsets['pump']+3]

    h_boil_out, mdot_boil = boiler_state
    h_turb_out, W_turbine, mdot_turb = turbine_state
    h_cond_out, mdot_cond = condenser_state
    h_pump_out, W_pump, mdot_pump = pump_state

    Q_in = boiler.Q
    W_net = W_turbine - W_pump
    efficiency = W_net / Q_in

    print(f"\nResults:")
    print(f"  Turbine power: {W_turbine/1e6:.2f} MW")
    print(f"  Pump power: {W_pump/1e6:.2f} MW")
    print(f"  Net power: {W_net/1e6:.2f} MW")
    print(f"  Mass flow rate: {mdot_boil:.2f} kg/s")
    print(f"  Thermal efficiency: {efficiency*100:.1f}%")

    # Validate
    if 0.30 < efficiency < 0.40:
        print("✓ Efficiency in expected range (30-40%)")
    else:
        print(f"⚠ Efficiency {efficiency*100:.1f}% outside expected range")

    # Check energy balance
    Q_out = abs(condenser.Q)
    energy_balance = Q_in - W_net - Q_out
    print(f"\n  Energy balance error: {energy_balance/1e6:.2f} MW ({abs(energy_balance)/Q_in*100:.2f}%)")

else:
    print(f"✗ Solution failed: {result.message}")
