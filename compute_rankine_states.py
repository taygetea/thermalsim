"""Compute good initial conditions for Rankine cycle using CoolProp."""

from CoolProp.CoolProp import PropsSI

# Rankine cycle operating points
P_high = 10e6  # 10 MPa
P_low = 10e3   # 10 kPa

print("Computing thermodynamic states for water Rankine cycle:")
print(f"  High pressure: {P_high/1e6:.1f} MPa")
print(f"  Low pressure: {P_low/1e3:.1f} kPa")
print()

# State 1: After condenser (before pump) - saturated liquid at low pressure
T1_sat = PropsSI('T', 'P', P_low, 'Q', 0, 'Water')
h1 = PropsSI('H', 'P', P_low, 'Q', 0, 'Water')
print(f"State 1 (condenser outlet / pump inlet):")
print(f"  P = {P_low/1e3:.1f} kPa, T = {T1_sat-273.15:.1f} °C")
print(f"  h = {h1/1e6:.3f} MJ/kg (saturated liquid)")
print()

# State 2: After pump - compressed liquid at high pressure
# Approximate: h2 ≈ h1 + v1 * (P2 - P1) / efficiency
# Or use isentropic: s2 = s1, then adjust for efficiency
s1 = PropsSI('S', 'P', P_low, 'H', h1, 'Water')
h2s = PropsSI('H', 'P', P_high, 'S', s1, 'Water')
eta_pump = 0.80
h2 = h1 + (h2s - h1) / eta_pump
print(f"State 2 (pump outlet / boiler inlet):")
print(f"  P = {P_high/1e6:.1f} MPa")
print(f"  h = {h2/1e6:.3f} MJ/kg (compressed liquid)")
print()

# State 3: After boiler - saturated vapor at high pressure
T3_sat = PropsSI('T', 'P', P_high, 'Q', 1, 'Water')
h3 = PropsSI('H', 'P', P_high, 'Q', 1, 'Water')
print(f"State 3 (boiler outlet / turbine inlet):")
print(f"  P = {P_high/1e6:.1f} MPa, T = {T3_sat-273.15:.1f} °C")
print(f"  h = {h3/1e6:.3f} MJ/kg (saturated vapor)")
print()

# State 4: After turbine - two-phase at low pressure
s3 = PropsSI('S', 'P', P_high, 'H', h3, 'Water')
h4s = PropsSI('H', 'P', P_low, 'S', s3, 'Water')
eta_turbine = 0.85
h4 = h3 - eta_turbine * (h3 - h4s)
print(f"State 4 (turbine outlet / condenser inlet):")
print(f"  P = {P_low/1e3:.1f} kPa")
print(f"  h = {h4/1e6:.3f} MJ/kg (two-phase)")
print()

# Compute cycle performance with these states
mdot_guess = 100.0  # kg/s
W_turbine = mdot_guess * (h3 - h4)
W_pump = mdot_guess * (h2 - h1)
Q_in = mdot_guess * (h3 - h2)
Q_out = mdot_guess * (h4 - h1)
W_net = W_turbine - W_pump
efficiency = W_net / Q_in

print(f"Cycle performance (mdot = {mdot_guess} kg/s):")
print(f"  Q_in = {Q_in/1e6:.2f} MW")
print(f"  Q_out = {Q_out/1e6:.2f} MW")
print(f"  W_turbine = {W_turbine/1e6:.2f} MW")
print(f"  W_pump = {W_pump/1e6:.2f} MW")
print(f"  W_net = {W_net/1e6:.2f} MW")
print(f"  Efficiency = {efficiency*100:.2f}%")
print()

print("Suggested initial conditions for components:")
print(f"  Boiler: h_out = {h3:.0f} J/kg, mdot = {mdot_guess} kg/s")
print(f"  Turbine: h_out = {h4:.0f} J/kg, W_shaft = {W_turbine:.0f} W, mdot = {mdot_guess} kg/s")
print(f"  Condenser: h_out = {h1:.0f} J/kg, mdot = {mdot_guess} kg/s")
print(f"  Pump: h_out = {h2:.0f} J/kg, W_shaft = {W_pump:.0f} W, mdot = {mdot_guess} kg/s")
