"""
Basic engineering calculations.

Author: Ruan Vitorino Souza
Description:
Simple electrical calculations used as part of engineering studies.
"""

def series(resistances):
    total = 0
    for r in resistances:
        total += r
    return total


def parallel(resistances):
    inverse_total = 0
    for r in resistances:
        inverse_total += 1 / r
    return 1 / inverse_total


# Example
resistors = [10, 20, 30]

r_series = series(resistors)
r_parallel = parallel(resistors)

print(f"Resistors: {resistors}")
print(f"Series resistance: {r_series} Ohms")
print(f"Parallel resistance: {r_parallel:.2f} Ohms")
