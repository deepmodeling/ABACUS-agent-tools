RY_TO_EV = 13.60569253
THZ_TO_K = 47.9924

# Physical constants for effective mass calculation
HBAR_EV_S = 6.582119569e-16  # ℏ in eV·s
HBAR_J_S = 1.054571817e-34   # ℏ in J·s
ELECTRON_MASS_KG = 9.1093837015e-31  # m_e in kg
ANGSTROM_TO_M = 1e-10  # Å to m conversion
EV_TO_J = 1.602176634e-19  # eV to J conversion

# Derived constant: m*/m_e = EFFECTIVE_MASS_FACTOR / curvature
# where curvature is d²E/dk² in eV/Å⁻²
# Formula: m* = ℏ² / (m_e * d²E/dk²)
# Converting units: ℏ in J·s, m_e in kg, curvature in eV/Å⁻²
EFFECTIVE_MASS_FACTOR = (HBAR_J_S**2) / (ELECTRON_MASS_KG * EV_TO_J * ANGSTROM_TO_M**2)
