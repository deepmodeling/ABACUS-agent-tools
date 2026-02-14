"""
Unit tests for effective mass calculation functions.
"""
import os
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# Set test mode to avoid MCP server initialization
os.environ["ABACUSAGENT_MODEL"] = "test"

# Mock MPI-related imports to avoid MPI dependency in tests
sys.modules['mpi4py'] = MagicMock()
sys.modules['mpi4py.MPI'] = MagicMock()
sys.modules['pyatb'] = MagicMock()
sys.modules['pyatb.easy_use'] = MagicMock()
sys.modules['pyatb.easy_use.input_generator'] = MagicMock()
sys.modules['pyatb.easy_use.stru_analyzer'] = MagicMock()
sys.modules['pyatb.parallel'] = MagicMock()

from abacusagent.modules.submodules.band import (
    fit_parabola_1d,
    calculate_effective_mass_from_curvature,
    find_band_extrema,
)
from abacusagent.constant import EFFECTIVE_MASS_FACTOR


class TestParabolicFitting:
    """Test parabolic fitting function."""

    def test_fit_parabola_perfect_fit(self):
        """Test fitting with perfect parabolic data."""
        # Generate perfect parabola: E = 1.0 + 2.0*k^2
        k_distances = np.linspace(-0.5, 0.5, 11)
        energies = 1.0 + 2.0 * k_distances**2

        result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), 0.0)

        assert result['E0'] == pytest.approx(1.0, abs=1e-10)
        assert result['a'] == pytest.approx(2.0, abs=1e-10)
        assert result['curvature'] == pytest.approx(4.0, abs=1e-10)
        assert result['r_squared'] == pytest.approx(1.0, abs=1e-10)

    def test_fit_parabola_with_noise(self):
        """Test fitting with noisy data."""
        np.random.seed(42)
        k_distances = np.linspace(-0.5, 0.5, 11)
        energies = 1.0 + 2.0 * k_distances**2 + np.random.normal(0, 0.01, 11)

        result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), 0.0)

        # Should still be close to true values
        assert result['E0'] == pytest.approx(1.0, abs=0.1)
        assert result['a'] == pytest.approx(2.0, abs=0.1)
        assert result['r_squared'] > 0.95

    def test_fit_parabola_insufficient_points(self):
        """Test fitting with insufficient data points."""
        k_distances = [0.0, 0.1]
        energies = [1.0, 1.02]

        result = fit_parabola_1d(k_distances, energies, 0.0)

        assert result['curvature'] is None
        assert 'error' in result

    def test_fit_parabola_negative_curvature(self):
        """Test fitting with negative curvature (band maximum)."""
        k_distances = np.linspace(-0.5, 0.5, 11)
        energies = 2.0 - 3.0 * k_distances**2  # Negative curvature

        result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), 0.0)

        assert result['a'] == pytest.approx(-3.0, abs=1e-10)
        assert result['curvature'] == pytest.approx(-6.0, abs=1e-10)


class TestEffectiveMassCalculation:
    """Test effective mass calculation from curvature."""

    def test_calculate_effective_mass_positive_curvature(self):
        """Test effective mass calculation with positive curvature."""
        # Typical semiconductor electron effective mass
        # For Si, m* ≈ 0.26 m_e, curvature ≈ EFFECTIVE_MASS_FACTOR / 0.26
        curvature = EFFECTIVE_MASS_FACTOR / 0.26

        result = calculate_effective_mass_from_curvature(curvature, 'kx')

        assert result['m_star'] == pytest.approx(0.26, abs=1e-6)
        assert result['is_flat_band'] is False

    def test_calculate_effective_mass_negative_curvature(self):
        """Test effective mass calculation with negative curvature (hole)."""
        # Typical hole effective mass
        curvature = -EFFECTIVE_MASS_FACTOR / 0.5

        result = calculate_effective_mass_from_curvature(curvature, 'kx')

        assert result['m_star'] == pytest.approx(-0.5, abs=1e-6)
        assert result['is_flat_band'] is False

    def test_calculate_effective_mass_flat_band(self):
        """Test effective mass calculation for flat band."""
        curvature = 1e-12  # Nearly zero curvature

        result = calculate_effective_mass_from_curvature(curvature, 'kx')

        assert np.isinf(result['m_star'])
        assert result['is_flat_band'] is True

    def test_calculate_effective_mass_zero_curvature(self):
        """Test effective mass calculation with exactly zero curvature."""
        curvature = 0.0

        result = calculate_effective_mass_from_curvature(curvature, 'kx')

        assert np.isinf(result['m_star'])
        assert result['is_flat_band'] is True


class TestBandExtremaDetection:
    """Test band extrema detection function."""

    def test_find_vbm_cbm_simple(self):
        """Test VBM and CBM detection in simple band structure."""
        # Create simple band structure with clear VBM and CBM
        # 3 bands, 10 k-points
        bands = [
            [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1],  # Valence band
            [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],      # Band crossing Fermi
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]            # Conduction band
        ]
        kpoints = [[i*0.1, 0, 0] for i in range(10)]
        kline = [i*0.1 for i in range(10)]
        efermi = 0.0
        energy_range = [-3.0, 3.0]

        extrema = find_band_extrema(bands, kpoints, kline, efermi, energy_range)

        # Should find VBM and CBM
        vbm = [e for e in extrema if e['extrema_type'] == 'VBM']
        cbm = [e for e in extrema if e['extrema_type'] == 'CBM']

        assert len(vbm) == 1
        assert len(cbm) == 1
        assert vbm[0]['energy'] == pytest.approx(-0.1, abs=1e-6)
        assert cbm[0]['energy'] == pytest.approx(0.1, abs=1e-6)

    def test_find_local_extrema(self):
        """Test detection of local extrema."""
        # Create band with local maximum
        bands = [
            [-1.0, -0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5]  # Local max at k=2
        ]
        kpoints = [[i*0.1, 0, 0] for i in range(10)]
        kline = [i*0.1 for i in range(10)]
        efermi = 0.0
        energy_range = [-4.0, 1.0]

        extrema = find_band_extrema(bands, kpoints, kline, efermi, energy_range)

        # Should find VBM (which is also local max at k=2)
        # But VBM is the highest energy below Fermi, which is at k=1 (-0.5 eV)
        assert len(extrema) >= 1
        vbm = [e for e in extrema if e['extrema_type'] == 'VBM'][0]
        # VBM is at k=1 with energy -0.5 eV (highest below Fermi)
        assert vbm['kpoint_index'] == 1

    def test_find_extrema_with_band_indices(self):
        """Test extrema detection with specific band indices."""
        bands = [
            [-2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1],
            [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        ]
        kpoints = [[i*0.1, 0, 0] for i in range(10)]
        kline = [i*0.1 for i in range(10)]
        efermi = 0.0
        energy_range = [-3.0, 3.0]

        # Only analyze band 1
        extrema = find_band_extrema(bands, kpoints, kline, efermi, energy_range,
                                   band_indices=[1])

        # Should only find extrema in band 1
        for e in extrema:
            assert e['band_index'] == 1


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_effective_mass_workflow(self):
        """Test complete workflow from band data to effective mass."""
        # Generate parabolic band around k=0
        k_distances = np.linspace(-0.5, 0.5, 11)
        curvature_true = 4.0  # eV/Å^2
        energies = 1.0 + 0.5 * curvature_true * k_distances**2

        # Fit parabola
        fit_result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), 0.0)

        # Calculate effective mass
        mass_result = calculate_effective_mass_from_curvature(
            fit_result['curvature'],
            'kpath'
        )

        # Verify
        assert fit_result['curvature'] == pytest.approx(curvature_true, abs=1e-10)
        expected_mass = EFFECTIVE_MASS_FACTOR / curvature_true
        assert mass_result['m_star'] == pytest.approx(expected_mass, rel=1e-10)

    def test_effective_mass_for_silicon_like_band(self):
        """Test effective mass calculation for Si-like band structure."""
        # Si electron effective mass at Γ point: m* ≈ 0.26 m_e
        # This corresponds to curvature ≈ EFFECTIVE_MASS_FACTOR / 0.26
        expected_mass = 0.26
        curvature = EFFECTIVE_MASS_FACTOR / expected_mass

        # Generate band data
        k_distances = np.linspace(-0.2, 0.2, 21)
        energies = 0.5 + 0.5 * curvature * k_distances**2

        # Fit and calculate
        fit_result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), 0.0)
        mass_result = calculate_effective_mass_from_curvature(
            fit_result['curvature'],
            'kpath'
        )

        assert mass_result['m_star'] == pytest.approx(expected_mass, abs=1e-3)


class TestParabolicFittingEdgeCases:
    """Additional edge case tests for parabolic fitting."""

    def test_fit_parabola_asymmetric_window(self):
        """Test fitting with asymmetric data around extremum."""
        # More points on one side
        k_distances = np.concatenate([
            np.linspace(-0.5, 0.0, 3),
            np.linspace(0.0, 0.5, 8)
        ])
        energies = 1.0 + 2.0 * k_distances**2

        result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), 0.0)

        # Should still fit reasonably well
        assert result['a'] == pytest.approx(2.0, abs=0.1)
        assert result['r_squared'] > 0.99

    def test_fit_parabola_off_center(self):
        """Test fitting with extremum not at k=0."""
        k0 = 0.3
        k_distances = np.linspace(0.0, 0.6, 11)
        energies = 1.5 + 3.0 * (k_distances - k0)**2

        result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), k0)

        assert result['E0'] == pytest.approx(1.5, abs=1e-2)
        assert result['a'] == pytest.approx(3.0, abs=1e-2)

    def test_fit_parabola_very_flat_band(self):
        """Test fitting with very small curvature."""
        k_distances = np.linspace(-0.5, 0.5, 11)
        energies = 1.0 + 1e-6 * k_distances**2

        result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), 0.0)

        assert result['curvature'] == pytest.approx(2e-6, abs=1e-8)
        assert result['r_squared'] > 0.5

    def test_fit_parabola_linear_component(self):
        """Test fitting with linear component (non-extremum point)."""
        k_distances = np.linspace(-0.5, 0.5, 11)
        # E = 1.0 + 0.5*k + 2.0*k^2 (has linear term)
        energies = 1.0 + 0.5 * k_distances + 2.0 * k_distances**2

        result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), 0.0)

        # Should still extract curvature correctly
        assert result['curvature'] == pytest.approx(4.0, abs=1e-10)
        assert result['b'] == pytest.approx(0.5, abs=1e-10)


class TestEffectiveMassEdgeCases:
    """Additional edge case tests for effective mass calculation."""

    def test_very_light_effective_mass(self):
        """Test calculation with very light effective mass (large curvature)."""
        # m* = 0.01 m_e (very light)
        curvature = EFFECTIVE_MASS_FACTOR / 0.01

        result = calculate_effective_mass_from_curvature(curvature, 'kx')

        assert result['m_star'] == pytest.approx(0.01, abs=1e-6)
        assert result['is_flat_band'] is False

    def test_very_heavy_effective_mass(self):
        """Test calculation with very heavy effective mass (small curvature)."""
        # m* = 10.0 m_e (very heavy)
        curvature = EFFECTIVE_MASS_FACTOR / 10.0

        result = calculate_effective_mass_from_curvature(curvature, 'kx')

        assert result['m_star'] == pytest.approx(10.0, abs=1e-6)
        assert result['is_flat_band'] is False

    def test_effective_mass_sign_preservation(self):
        """Test that sign of effective mass is preserved."""
        # Positive curvature (electron-like)
        curvature_pos = EFFECTIVE_MASS_FACTOR / 0.5
        result_pos = calculate_effective_mass_from_curvature(curvature_pos, 'kx')
        assert result_pos['m_star'] > 0

        # Negative curvature (hole-like)
        curvature_neg = -EFFECTIVE_MASS_FACTOR / 0.5
        result_neg = calculate_effective_mass_from_curvature(curvature_neg, 'kx')
        assert result_neg['m_star'] < 0

    def test_effective_mass_near_zero_curvature(self):
        """Test behavior near flat band threshold."""
        # Just above threshold (threshold is 1e-10)
        curvature = 1e-11
        result = calculate_effective_mass_from_curvature(curvature, 'kx')
        assert result['is_flat_band'] is True
        assert np.isinf(result['m_star'])


class TestBandExtremaEdgeCases:
    """Additional edge case tests for band extrema detection."""

    def test_find_extrema_no_gap(self):
        """Test extrema detection in metallic system (no gap)."""
        # Band crossing Fermi level
        bands = [
            [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        ]
        kpoints = [[i*0.1, 0, 0] for i in range(10)]
        kline = [i*0.1 for i in range(10)]
        efermi = 0.0
        energy_range = [-2.0, 4.0]

        extrema = find_band_extrema(bands, kpoints, kline, efermi, energy_range)

        # Should find VBM and CBM at adjacent points
        vbm = [e for e in extrema if e['extrema_type'] == 'VBM']
        cbm = [e for e in extrema if e['extrema_type'] == 'CBM']

        assert len(vbm) == 1
        assert len(cbm) == 1

    def test_find_extrema_multiple_bands(self):
        """Test extrema detection with multiple bands."""
        bands = [
            [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.4, -0.3, -0.2, -0.1],  # VB
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],            # CB1
            [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]             # CB2
        ]
        kpoints = [[i*0.1, 0, 0] for i in range(10)]
        kline = [i*0.1 for i in range(10)]
        efermi = 0.0
        energy_range = [-4.0, 3.0]

        extrema = find_band_extrema(bands, kpoints, kline, efermi, energy_range)

        # Should find VBM in band 0 and CBM in band 1
        vbm = [e for e in extrema if e['extrema_type'] == 'VBM']
        cbm = [e for e in extrema if e['extrema_type'] == 'CBM']

        assert len(vbm) == 1
        assert len(cbm) == 1
        assert vbm[0]['band_index'] == 0
        assert cbm[0]['band_index'] == 1

    def test_find_extrema_at_boundary(self):
        """Test extrema detection when extremum is at k-path boundary."""
        # Band with maximum at first point
        bands = [
            [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]
        ]
        kpoints = [[i*0.1, 0, 0] for i in range(10)]
        kline = [i*0.1 for i in range(10)]
        efermi = 0.0
        energy_range = [-1.0, 0.5]

        extrema = find_band_extrema(bands, kpoints, kline, efermi, energy_range)

        # Should find VBM - but it's at the boundary (k=0)
        # The algorithm looks for local extrema (comparing with neighbors)
        # So boundary points may not be detected as local extrema
        # VBM will be the highest energy below Fermi, which is at k=0 (0.0 eV)
        # But since 0.0 is not < 0, it won't be VBM. Next is k=1 (-0.1 eV)
        vbm = [e for e in extrema if e['extrema_type'] == 'VBM']
        assert len(vbm) == 1
        assert vbm[0]['kpoint_index'] == 1  # -0.1 eV is the highest below Fermi

    def test_find_extrema_with_plateau(self):
        """Test extrema detection with flat region (plateau)."""
        bands = [
            [-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0]
        ]
        kpoints = [[i*0.1, 0, 0] for i in range(10)]
        kline = [i*0.1 for i in range(10)]
        efermi = 0.0
        energy_range = [-2.0, 3.0]

        extrema = find_band_extrema(bands, kpoints, kline, efermi, energy_range)

        # Should find CBM (one of the plateau points)
        cbm = [e for e in extrema if e['extrema_type'] == 'CBM']
        assert len(cbm) >= 1


class TestFittingQuality:
    """Tests for fitting quality assessment."""

    def test_good_fit_high_r_squared(self):
        """Test that good parabolic data gives high R²."""
        k_distances = np.linspace(-0.5, 0.5, 21)
        energies = 1.0 + 2.0 * k_distances**2

        result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), 0.0)

        assert result['r_squared'] > 0.999

    def test_poor_fit_low_r_squared(self):
        """Test that non-parabolic data gives lower R²."""
        k_distances = np.linspace(-0.5, 0.5, 21)
        # Quartic function (not parabolic)
        energies = 1.0 + 2.0 * k_distances**4

        result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), 0.0)

        # R² should be lower for non-parabolic data
        assert result['r_squared'] < 0.99

    def test_residuals_calculation(self):
        """Test that residuals are calculated correctly."""
        k_distances = np.linspace(-0.5, 0.5, 11)
        energies = 1.0 + 2.0 * k_distances**2

        result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), 0.0)

        # For perfect fit, residuals should be near zero
        assert all(abs(r) < 1e-10 for r in result['residuals'])

    def test_num_points_in_result(self):
        """Test that number of points is correctly reported."""
        k_distances = np.linspace(-0.5, 0.5, 15)
        energies = 1.0 + 2.0 * k_distances**2

        result = fit_parabola_1d(k_distances.tolist(), energies.tolist(), 0.0)

        assert result['num_points'] == 15


class TestPhysicalConstants:
    """Tests to verify physical constants are correct."""

    def test_effective_mass_factor_value(self):
        """Test that EFFECTIVE_MASS_FACTOR has reasonable value."""
        # Should be around 7.62 for the given units
        assert EFFECTIVE_MASS_FACTOR > 7.0
        assert EFFECTIVE_MASS_FACTOR < 8.0

    def test_effective_mass_units_consistency(self):
        """Test unit consistency in effective mass calculation."""
        # For GaAs: m* ≈ 0.067 m_e
        # Typical curvature for GaAs CBM
        m_star_expected = 0.067
        curvature = EFFECTIVE_MASS_FACTOR / m_star_expected

        result = calculate_effective_mass_from_curvature(curvature, 'kx')

        assert result['m_star'] == pytest.approx(m_star_expected, abs=1e-6)


class TestRobustness:
    """Tests for robustness and error handling."""

    def test_fit_with_nan_values(self):
        """Test fitting behavior with NaN values."""
        k_distances = [0.0, 0.1, 0.2, 0.3, 0.4]
        energies = [1.0, 1.02, np.nan, 1.08, 1.16]

        result = fit_parabola_1d(k_distances, energies, 0.0)

        # Should handle NaN gracefully
        assert 'error' in result or result['curvature'] is None

    def test_fit_with_inf_values(self):
        """Test fitting behavior with infinite values."""
        k_distances = [0.0, 0.1, 0.2, 0.3, 0.4]
        energies = [1.0, 1.02, np.inf, 1.08, 1.16]

        result = fit_parabola_1d(k_distances, energies, 0.0)

        # Should handle inf gracefully - polyfit returns NaN
        assert 'error' in result or result['curvature'] is None or np.isnan(result['curvature'])

    def test_empty_band_list(self):
        """Test extrema detection with empty band list."""
        bands = []
        kpoints = []
        kline = []
        efermi = 0.0
        energy_range = [-1.0, 1.0]

        extrema = find_band_extrema(bands, kpoints, kline, efermi, energy_range)

        assert len(extrema) == 0

    def test_single_kpoint(self):
        """Test extrema detection with single k-point."""
        bands = [[0.5]]
        kpoints = [[0.0, 0.0, 0.0]]
        kline = [0.0]
        efermi = 0.0
        energy_range = [-1.0, 1.0]

        extrema = find_band_extrema(bands, kpoints, kline, efermi, energy_range)

        # Cannot find local extrema with single point
        # But should find VBM or CBM if energy is appropriate
        assert len(extrema) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
