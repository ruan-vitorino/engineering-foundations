"""
Basic engineering calculations.

Author: Ruan Vitorino Souza
Description:
Simple electrical calculations used as part of engineering studies.
"""
def eletrical_power(voltage, current):
  return voltage * current

def ohms_law(voltage_ohm, current_ohm):
  return voltage / current

power = eletrical_power(220, 5)
resistance = ohms_law(12, 2)

print(f"Power: {power} W")
print(f"Resistance: {resistance} Ohms")
