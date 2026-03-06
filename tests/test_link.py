"""Tests for missiontools.comm.Link."""

import numpy as np
import pytest

from missiontools import Spacecraft, GroundStation, IsotropicAntenna, Link
from missiontools.comm import SymmetricAntenna
from missiontools.orbit.propagation import propagate_analytical
from missiontools.orbit.frames import geodetic_to_ecef, ecef_to_eci, eci_to_ecef


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPOCH = np.datetime64('2025-01-01T00:00:00', 'us')
_C = 299_792_458.0
_K_DB = 10.0 * np.log10(1.380649e-23)  # ≈ -228.6 dBW/K/Hz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sc():
    """Simple circular equatorial orbit at 600 km."""
    return Spacecraft(
        a=6_971_000.0, e=0.0, i=0.0, raan=0.0, arg_p=0.0, ma=0.0,
        epoch=_EPOCH,
    )


def _subsat_latlon_at_epoch(sc):
    """Return (lat_deg, lon_deg) of the sub-satellite point at _EPOCH."""
    r_eci, _ = propagate_analytical(
        np.atleast_1d(_EPOCH), **sc.keplerian_params, type=sc.propagator_type
    )
    r_ecef = eci_to_ecef(r_eci, np.atleast_1d(_EPOCH))
    x, y, z = r_ecef[0]
    lon_deg = float(np.degrees(np.arctan2(y, x)))
    lat_deg = float(np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2))))
    return lat_deg, lon_deg


def _aligned_sc_gs():
    """SC and GS such that SC is directly overhead GS at _EPOCH."""
    sc = _make_sc()
    lat, lon = _subsat_latlon_at_epoch(sc)
    gs = GroundStation(lat=lat, lon=lon)
    return sc, gs


def _compute_range_at_epoch(sc, gs):
    """Range from SC to GS at _EPOCH (m)."""
    r_sc, _ = propagate_analytical(
        np.atleast_1d(_EPOCH), **sc.keplerian_params, type=sc.propagator_type
    )
    r_gs_ecef = geodetic_to_ecef(np.radians(gs.lat), np.radians(gs.lon), gs.alt)
    r_gs_eci = ecef_to_eci(np.atleast_2d(r_gs_ecef), np.atleast_1d(_EPOCH))
    return float(np.linalg.norm(r_sc[0] - r_gs_eci[0]))


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestLinkConstruction:
    def setup_method(self):
        sc, gs = _aligned_sc_gs()
        self.tx = IsotropicAntenna()
        self.rx = IsotropicAntenna()
        sc.add_antenna(self.tx)
        gs.add_antenna(self.rx)
        self.default_kwargs = dict(
            tx=self.tx, rx=self.rx,
            tx_power_dbw=0.0, frequency_hz=8e9, data_rate_bps=1e6,
            rx_gt_db_k=0.0, required_eb_n0_db=6.0,
        )

    def test_valid_construction(self):
        link = Link(**self.default_kwargs)
        assert link.tx is self.tx
        assert link.rx is self.rx

    def test_tx_wrong_type(self):
        with pytest.raises(TypeError, match="tx must be an AbstractAntenna"):
            Link(**{**self.default_kwargs, 'tx': object()})

    def test_rx_wrong_type(self):
        with pytest.raises(TypeError, match="rx must be an AbstractAntenna"):
            Link(**{**self.default_kwargs, 'rx': "not an antenna"})

    def test_tx_not_attached(self):
        unattached = IsotropicAntenna()
        with pytest.raises(ValueError, match="tx antenna must be attached"):
            Link(**{**self.default_kwargs, 'tx': unattached})

    def test_rx_not_attached(self):
        unattached = IsotropicAntenna()
        with pytest.raises(ValueError, match="rx antenna must be attached"):
            Link(**{**self.default_kwargs, 'rx': unattached})

    def test_frequency_zero(self):
        with pytest.raises(ValueError, match="frequency_hz must be positive"):
            Link(**{**self.default_kwargs, 'frequency_hz': 0.0})

    def test_frequency_negative(self):
        with pytest.raises(ValueError, match="frequency_hz must be positive"):
            Link(**{**self.default_kwargs, 'frequency_hz': -1.0})

    def test_data_rate_zero(self):
        with pytest.raises(ValueError, match="data_rate_bps must be positive"):
            Link(**{**self.default_kwargs, 'data_rate_bps': 0.0})

    def test_defaults(self):
        link = Link(**self.default_kwargs)
        assert link.implementation_loss_db == 2.0
        assert link.misc_losses_db == 0.0
        assert link.use_p618 is True


# ---------------------------------------------------------------------------
# peak_gain_dbi
# ---------------------------------------------------------------------------

class TestPeakGainDbi:
    def test_isotropic_default(self):
        assert IsotropicAntenna().peak_gain_dbi == 0.0

    def test_isotropic_custom(self):
        assert IsotropicAntenna(gain_dbi=5.0).peak_gain_dbi == 5.0

    def test_symmetric_at_boresight(self):
        sc = _make_sc()
        ant = SymmetricAntenna(
            angles_deg=[0.0, 90.0], gains_dbi=[10.0, -3.0],
            body_vector=[0, 0, 1],
        )
        sc.add_antenna(ant)
        assert ant.peak_gain_dbi == pytest.approx(10.0)

    def test_symmetric_nonzero_start(self):
        sc = _make_sc()
        ant = SymmetricAntenna(
            angles_deg=[30.0, 90.0], gains_dbi=[6.0, -6.0],
            body_vector=[0, 0, 1],
        )
        sc.add_antenna(ant)
        # np.interp clamps to left value at 0° → 6.0
        assert ant.peak_gain_dbi == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Link margin — analytical verification (use_p618=False)
# ---------------------------------------------------------------------------

class TestLinkMarginAnalytical:
    """Verify link budget formula against a manual calculation.

    SC is directly above GS at _EPOCH (equatorial orbit, GS at subsat point).
    Both antennas are isotropic → G_tx = const, pointing_loss = 0.
    """

    def setup_method(self):
        self.sc, self.gs = _aligned_sc_gs()
        self.tx = IsotropicAntenna(gain_dbi=3.0)
        self.rx = IsotropicAntenna(gain_dbi=0.0)
        self.sc.add_antenna(self.tx)
        self.gs.add_antenna(self.rx)
        self.freq = 8.4e9
        self.rate = 1e6
        self.gt = 10.0
        self.req_ebn0 = 6.0
        self.impl_loss = 2.0

    def test_single_timestep(self):
        link = Link(
            tx=self.tx, rx=self.rx,
            tx_power_dbw=0.0, frequency_hz=self.freq, data_rate_bps=self.rate,
            rx_gt_db_k=self.gt, required_eb_n0_db=self.req_ebn0,
            implementation_loss_db=self.impl_loss, misc_losses_db=0.0,
            use_p618=False,
        )
        margin = link.link_margin(_EPOCH)

        # Manual calculation
        r = _compute_range_at_epoch(self.sc, self.gs)
        fspl_db = 20.0 * np.log10(4.0 * np.pi * r * self.freq / _C)
        # G_tx=3, G/T=10, pointing_loss=0 (isotropic)
        c_n0 = 0.0 + 3.0 - fspl_db + self.gt + 0.0 - _K_DB
        eb_n0 = c_n0 - 10.0 * np.log10(self.rate)
        expected = eb_n0 - self.req_ebn0 - self.impl_loss

        assert margin == pytest.approx(expected, abs=0.01)

    def test_array_timestep(self):
        link = Link(
            tx=self.tx, rx=self.rx,
            tx_power_dbw=0.0, frequency_hz=self.freq, data_rate_bps=self.rate,
            rx_gt_db_k=self.gt, required_eb_n0_db=self.req_ebn0,
            implementation_loss_db=self.impl_loss, use_p618=False,
        )
        t = _EPOCH + np.arange(3) * np.timedelta64(10, 's')
        margin = link.link_margin(t)
        assert margin.shape == (3,)

    def test_scalar_t_returns_scalar(self):
        link = Link(
            tx=self.tx, rx=self.rx,
            tx_power_dbw=0.0, frequency_hz=self.freq, data_rate_bps=self.rate,
            rx_gt_db_k=self.gt, required_eb_n0_db=self.req_ebn0, use_p618=False,
        )
        result = link.link_margin(_EPOCH)
        assert np.ndim(result) == 0

    def test_misc_losses_reduce_margin(self):
        kw = dict(
            tx=self.tx, rx=self.rx,
            tx_power_dbw=0.0, frequency_hz=self.freq, data_rate_bps=self.rate,
            rx_gt_db_k=self.gt, required_eb_n0_db=self.req_ebn0, use_p618=False,
        )
        link_no_misc = Link(**kw, misc_losses_db=0.0)
        link_with_misc = Link(**kw, misc_losses_db=3.0)
        diff = link_no_misc.link_margin(_EPOCH) - link_with_misc.link_margin(_EPOCH)
        assert diff == pytest.approx(3.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Pointing loss
# ---------------------------------------------------------------------------

class TestPointingLoss:
    """Off-boresight rx antenna changes pointing loss."""

    def test_on_boresight_no_pointing_loss(self):
        """Directional rx pointing at zenith, SC overhead → 0 pointing loss."""
        sc, gs = _aligned_sc_gs()

        tx = IsotropicAntenna()
        sc.add_antenna(tx)

        # Directional rx pointing at zenith (elevation 90°)
        rx_dir = SymmetricAntenna(
            angles_deg=[0.0, 90.0],
            gains_dbi=[10.0, -3.0],
            azimuth_deg=0.0,
            elevation_deg=90.0,
        )
        gs.add_antenna(rx_dir)

        # Isotropic rx (G/T=0, same as directional boresight minus peak=10)
        # But here G/T parameter stays 0, so directional adds 0 pointing loss
        # compared to itself (G_rx - G_rx_peak = 10 - 10 = 0)
        kw = dict(
            tx_power_dbw=0.0, frequency_hz=8e9, data_rate_bps=1e6,
            rx_gt_db_k=0.0, required_eb_n0_db=0.0, use_p618=False,
            implementation_loss_db=0.0,
        )
        link = Link(tx=tx, rx=rx_dir, **kw)
        margin = link.link_margin(_EPOCH)
        assert np.isfinite(margin)

    def test_pointing_loss_reduces_margin(self):
        """Off-boresight rx → margin decreases vs boresight."""
        sc1, gs = _aligned_sc_gs()
        sc2, _ = _aligned_sc_gs()  # same geometry

        tx1 = IsotropicAntenna()
        sc1.add_antenna(tx1)

        tx2 = IsotropicAntenna()
        sc2.add_antenna(tx2)

        # rx1: directional, pointing at zenith → boresight at SC
        gs2 = GroundStation(lat=gs.lat, lon=gs.lon)
        rx_zenith = SymmetricAntenna(
            angles_deg=[0.0, 90.0], gains_dbi=[10.0, -3.0],
            azimuth_deg=0.0, elevation_deg=90.0,
        )
        gs.add_antenna(rx_zenith)

        # rx2: directional, pointing at horizon → off-boresight at SC
        rx_horizon = SymmetricAntenna(
            angles_deg=[0.0, 90.0], gains_dbi=[10.0, -3.0],
            azimuth_deg=0.0, elevation_deg=0.0,
        )
        gs2.add_antenna(rx_horizon)

        kw = dict(
            tx_power_dbw=0.0, frequency_hz=8e9, data_rate_bps=1e6,
            rx_gt_db_k=0.0, required_eb_n0_db=0.0, use_p618=False,
            implementation_loss_db=0.0,
        )
        margin_boresight = Link(tx=tx1, rx=rx_zenith, **kw).link_margin(_EPOCH)
        margin_offaxis = Link(tx=tx2, rx=rx_horizon, **kw).link_margin(_EPOCH)

        # SC is overhead → off-axis antenna sees SC at ~90° off-boresight → large pointing loss
        assert margin_boresight > margin_offaxis


# ---------------------------------------------------------------------------
# Obstruction / line-of-sight
# ---------------------------------------------------------------------------

class TestObstruction:
    def _make_link(self, sc, gs, **extra):
        tx = IsotropicAntenna()
        rx = IsotropicAntenna()
        sc.add_antenna(tx)
        gs.add_antenna(rx)
        return Link(
            tx=tx, rx=rx,
            tx_power_dbw=0.0, frequency_hz=8e9, data_rate_bps=1e6,
            rx_gt_db_k=0.0, required_eb_n0_db=0.0, use_p618=False,
            implementation_loss_db=0.0, **extra
        )

    def test_obstructed_is_nan(self):
        """SC on far side of Earth from GS → LOS blocked → NaN."""
        # SC at ma=π → roughly antipodal from ma=0 position
        # GS at the subsat point of ma=0 (opposite side from SC at epoch)
        sc_0 = _make_sc()
        lat, lon = _subsat_latlon_at_epoch(sc_0)
        gs = GroundStation(lat=lat, lon=lon)

        sc_anti = Spacecraft(
            a=6_971_000.0, e=0.0, i=0.0, raan=0.0, arg_p=0.0,
            ma=np.pi, epoch=_EPOCH,
        )
        link = self._make_link(sc_anti, gs)
        margin = link.link_margin(_EPOCH)
        assert np.isnan(margin)

    def test_unobstructed_is_finite(self):
        """SC directly above GS → LOS clear → finite margin."""
        sc, gs = _aligned_sc_gs()
        link = self._make_link(sc, gs)
        margin = link.link_margin(_EPOCH)
        assert np.isfinite(margin)

    def test_array_mixed_obstruction(self):
        """Array of times: SC starts above, then blocked on far side."""
        sc, gs = _aligned_sc_gs()
        link = self._make_link(sc, gs)

        # Orbital period (s)
        period_s = int(2 * np.pi * np.sqrt(sc.a**3 / sc.central_body_mu))
        t0 = _EPOCH
        t_half = _EPOCH + np.timedelta64(period_s // 2, 's')
        t_arr = np.array([t0, t_half], dtype='datetime64[us]')
        margins = link.link_margin(t_arr)
        assert margins.shape == (2,)
        assert np.isfinite(margins[0])   # above GS at t=0
        assert np.isnan(margins[1])      # antipodal at t=T/2


# ---------------------------------------------------------------------------
# SC–SC link with use_p618=True
# ---------------------------------------------------------------------------

class TestScScLink:
    def test_p618_silently_ignored_for_sc_sc(self):
        sc1 = _make_sc()
        sc2 = Spacecraft(
            a=7_371_000.0, e=0.0, i=np.radians(45.0),
            raan=0.0, arg_p=0.0, ma=0.0, epoch=_EPOCH,
        )
        tx = IsotropicAntenna()
        rx = IsotropicAntenna()
        sc1.add_antenna(tx)
        sc2.add_antenna(rx)

        kw = dict(
            tx=tx, rx=rx,
            tx_power_dbw=0.0, frequency_hz=8e9, data_rate_bps=1e6,
            rx_gt_db_k=0.0, required_eb_n0_db=0.0, implementation_loss_db=0.0,
        )
        link_p618 = Link(**kw, use_p618=True)
        link_no_p618 = Link(**kw, use_p618=False)

        t = _EPOCH + np.arange(3) * np.timedelta64(60, 's')
        m_p618 = link_p618.link_margin(t)
        m_no = link_no_p618.link_margin(t)

        np.testing.assert_array_equal(np.isnan(m_p618), np.isnan(m_no))
        finite = ~np.isnan(m_p618)
        if np.any(finite):
            np.testing.assert_allclose(m_p618[finite], m_no[finite], rtol=1e-10)


# ---------------------------------------------------------------------------
# P.618 smoke test
# ---------------------------------------------------------------------------

class TestP618Smoke:
    """Verify P.618 integration doesn't raise and produces sensible output."""

    def test_p618_sc_gs_downlink(self):
        pytest.importorskip('itur')

        sc, gs = _aligned_sc_gs()
        tx = IsotropicAntenna()
        rx = IsotropicAntenna()
        sc.add_antenna(tx)
        gs.add_antenna(rx)

        link = Link(
            tx=tx, rx=rx,
            tx_power_dbw=10.0, frequency_hz=8.4e9, data_rate_bps=1e6,
            rx_gt_db_k=10.0, required_eb_n0_db=6.0, use_p618=True,
            implementation_loss_db=0.0,
        )
        # 99.9% availability (p=0.1%) is in the valid range for ITU-R P.618
        margin = link.link_margin(_EPOCH, availability_pct=99.9)
        assert np.isfinite(margin)

    def test_p618_gs_sc_uplink(self):
        pytest.importorskip('itur')

        sc, gs = _aligned_sc_gs()
        tx = IsotropicAntenna()
        rx = IsotropicAntenna()
        gs.add_antenna(tx)
        sc.add_antenna(rx)

        link = Link(
            tx=tx, rx=rx,
            tx_power_dbw=10.0, frequency_hz=2.0e9, data_rate_bps=1e6,
            rx_gt_db_k=5.0, required_eb_n0_db=6.0, use_p618=True,
            implementation_loss_db=0.0,
        )
        margin = link.link_margin(_EPOCH, availability_pct=99.9)
        assert np.isfinite(margin)

    def test_p618_margin_lower_than_no_p618(self):
        """P618 attenuation should reduce link margin vs no P618."""
        pytest.importorskip('itur')

        sc, gs = _aligned_sc_gs()
        tx = IsotropicAntenna()
        rx = IsotropicAntenna()
        sc.add_antenna(tx)
        gs.add_antenna(rx)

        kw = dict(
            tx=tx, rx=rx,
            tx_power_dbw=10.0, frequency_hz=8.4e9, data_rate_bps=1e6,
            rx_gt_db_k=10.0, required_eb_n0_db=6.0, implementation_loss_db=0.0,
        )
        link_with = Link(**kw, use_p618=True)
        link_without = Link(**kw, use_p618=False)

        m_with = link_with.link_margin(_EPOCH, availability_pct=99.9)
        m_without = link_without.link_margin(_EPOCH)
        assert m_with < m_without


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_link_importable_from_top_level():
    from missiontools import Link as L
    assert L is Link
