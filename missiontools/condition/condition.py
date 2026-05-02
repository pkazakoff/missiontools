"""
Conditions
==========
Boolean time-domain predicates with built-in caching.

A :class:`AbstractCondition` is a callable with one method,
:meth:`~AbstractCondition.at`, that returns a boolean array indicating
whether the condition holds at each requested time.  Conditions capture
any required external state (spacecraft, ground stations, ...) at
construction time, so the public API depends only on time.

Hierarchy
---------
:class:`AbstractCondition` (ABC)
├── :class:`SpaceGroundAccessCondition`
├── :class:`SunlightCondition`
├── :class:`SubSatelliteRegionCondition`
├── :class:`VisibilityCondition`
├── :class:`AndCondition`
├── :class:`OrCondition`
├── :class:`NotCondition`
└── :class:`XorCondition`

Boolean composition operators (``&``, ``|``, ``^``, ``~``) are available
on every :class:`AbstractCondition` instance, so that::

    condition1 & (condition2 | condition3)

is equivalent to::

    AndCondition(condition1, OrCondition(condition2, condition3))
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
import shapely

from ..cache import cached_propagate_analytical
from ..orbit.access import earth_access, space_to_space_access
from ..orbit.frames import eci_to_ecef, ecef_to_eci, geodetic_to_ecef, ecef_to_geodetic
from ..orbit.shadow import in_sunlight


class AbstractCondition(ABC):
    """Base class for boolean time-domain conditions.

    Subclasses implement :meth:`_compute` and :meth:`__repr__`.  The base
    class handles input coercion, scalar/array shape tracking, and a
    small per-instance count-based LRU cache keyed on the SHA-256 digest
    of the requested time array.

    Parameters
    ----------
    cache_size : int, optional
        Maximum number of distinct time arrays whose results are cached.
        Default 16.  Set to 0 to disable caching.

    Notes
    -----
    Subclasses can bypass caching entirely by overriding :meth:`at` rather
    than :meth:`_compute`.

    Boolean operators ``&``, ``|``, ``^``, ``~`` return new composite
    conditions:

    * ``a & b`` → :class:`AndCondition`
    * ``a | b`` → :class:`OrCondition`
    * ``a ^ b`` → :class:`XorCondition`
    * ``~a``    → :class:`NotCondition`
    """

    def __init__(self, cache_size: int = 16) -> None:
        if cache_size < 0:
            raise ValueError(f"cache_size must be non-negative, got {cache_size}")
        self._cache_size = cache_size
        self._cache: OrderedDict[bytes, npt.NDArray[np.bool_]] = OrderedDict()

    @abstractmethod
    def _compute(self, t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
        """Evaluate the condition at the given times.

        Parameters
        ----------
        t : ndarray of datetime64[us], shape (N,)
            Times at which to evaluate the condition.

        Returns
        -------
        ndarray of bool, shape (N,)
        """

    @abstractmethod
    def __repr__(self) -> str: ...

    def at(self, t: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Evaluate the condition at one or more times.

        Parameters
        ----------
        t : array_like of datetime64, shape (N,) or scalar
            Time(s) at which to evaluate.

        Returns
        -------
        ndarray of bool, shape (N,) or scalar bool
            True where the condition holds.
        """
        t_in = np.asarray(t, dtype="datetime64[us]")
        scalar = t_in.ndim == 0
        t_arr = np.atleast_1d(t_in)

        if self._cache_size > 0:
            key = hashlib.sha256(t_arr.tobytes()).digest()
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return bool(cached[0]) if scalar else cached
            result = np.asarray(self._compute(t_arr), dtype=bool)
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            result = np.asarray(self._compute(t_arr), dtype=bool)

        return bool(result[0]) if scalar else result

    def __and__(self, other: object) -> AndCondition:
        if not isinstance(other, AbstractCondition):
            return NotImplemented
        return AndCondition(self, other)

    def __or__(self, other: object) -> OrCondition:
        if not isinstance(other, AbstractCondition):
            return NotImplemented
        return OrCondition(self, other)

    def __xor__(self, other: object) -> XorCondition:
        if not isinstance(other, AbstractCondition):
            return NotImplemented
        return XorCondition(self, other)

    def __invert__(self) -> NotCondition:
        return NotCondition(self)

    def intervals(
        self,
        t_start: np.datetime64,
        t_end: np.datetime64,
        *,
        max_step: np.timedelta64 = np.timedelta64(10, "s"),
        tolerance: np.timedelta64 = np.timedelta64(1, "s"),
    ) -> list[tuple[np.datetime64, np.datetime64]]:
        """Return edge-refined intervals where the condition is True.

        Scans ``[t_start, t_end]`` at ``max_step`` resolution, detects
        rising/falling edges, then bisects each edge to ``tolerance``
        precision.

        Parameters
        ----------
        t_start : np.datetime64
            Start of the time window.
        t_end : np.datetime64
            End of the time window (inclusive).
        max_step : np.timedelta64, optional
            Scan step for the initial coarse pass (default 10 s).
        tolerance : np.timedelta64, optional
            Bisection refinement tolerance (default 1 s).

        Returns
        -------
        list[tuple[np.datetime64, np.datetime64]]
            Sorted, non-overlapping ``[(t0, t1), ...]`` intervals where
            the condition is True.  Empty when the condition is never True.
        """
        t_start = np.asarray(t_start, dtype="datetime64[us]")
        t_end = np.asarray(t_end, dtype="datetime64[us]")
        max_step = np.asarray(max_step, dtype="timedelta64[us]")
        tolerance = np.asarray(tolerance, dtype="timedelta64[us]")

        total_us = int((t_end - t_start) / np.timedelta64(1, "us"))
        step_us = int(max_step / np.timedelta64(1, "us"))
        tol_us = int(tolerance / np.timedelta64(1, "us"))

        if total_us <= 0 or step_us <= 0:
            return []

        offs = np.arange(0, total_us + 1, step_us, dtype=np.int64)
        if offs[-1] != total_us:
            offs = np.append(offs, np.int64(total_us))
        t_grid = t_start + offs.astype("timedelta64[us]")

        flags = self.at(t_grid)

        if flags.all():
            return [(t_start, t_end)]
        if not flags.any():
            return []

        padded = np.concatenate([[False], flags, [False]])
        rises = np.where(np.diff(padded.astype(np.int8)) == 1)[0]
        falls = np.where(np.diff(padded.astype(np.int8)) == -1)[0]

        def _bisect(
            lo: np.datetime64, hi: np.datetime64, target: bool
        ) -> np.datetime64:
            lo_us = int((lo - t_start) / np.timedelta64(1, "us"))
            hi_us = int((hi - t_start) / np.timedelta64(1, "us"))
            while (hi_us - lo_us) > tol_us:
                mid_us = (lo_us + hi_us) // 2
                mid_t = t_start + np.timedelta64(mid_us, "us")
                if bool(self.at(mid_t)) == target:
                    hi_us = mid_us
                else:
                    lo_us = mid_us
            return t_start + np.timedelta64(hi_us, "us")

        result = []
        for ri, fi in zip(rises, falls):
            t0 = t_grid[ri] if ri == 0 else _bisect(t_grid[ri - 1], t_grid[ri], True)
            t1 = (
                t_grid[fi - 1]
                if fi == len(t_grid)
                else _bisect(t_grid[fi - 1], t_grid[fi], False)
            )
            result.append((t0, t1))

        return result


class SpaceGroundAccessCondition(AbstractCondition):
    """True when a spacecraft is visible from a ground station.

    Visibility is the standard above-horizon test: the elevation angle
    from the geodetic up-direction at the ground station to the
    spacecraft must meet or exceed ``el_min_deg``.  Earth blockage is implicit
    for ``el_min_deg >= 0``.

    Parameters
    ----------
    spacecraft : Spacecraft
        The spacecraft whose visibility is being tested.
    ground_station : GroundStation
        The observing ground station.
    el_min_deg : float, optional
        Minimum elevation angle (degrees).  Default 5.0.

    Raises
    ------
    TypeError
        If ``spacecraft`` is not a :class:`~missiontools.Spacecraft` or
        ``ground_station`` is not a :class:`~missiontools.GroundStation`.

    Examples
    --------
    ::

        from missiontools import Spacecraft, GroundStation
        from missiontools.condition import SpaceGroundAccessCondition

        sc = Spacecraft(...)
        gs = GroundStation(lat=51.5, lon=-0.1)
        cond = SpaceGroundAccessCondition(sc, gs, el_min_deg=5.0)
        cond.at(np.datetime64('2025-01-01', 'us'))   # -> bool
    """

    def __init__(self, spacecraft, ground_station, el_min_deg: float = 5.0) -> None:
        from ..spacecraft import Spacecraft
        from ..ground_station import GroundStation

        if not isinstance(spacecraft, Spacecraft):
            raise TypeError(
                f"spacecraft must be a Spacecraft instance, "
                f"got {type(spacecraft).__name__!r}"
            )
        if not isinstance(ground_station, GroundStation):
            raise TypeError(
                f"ground_station must be a GroundStation instance, "
                f"got {type(ground_station).__name__!r}"
            )
        if not np.isfinite(el_min_deg):
            raise ValueError(f"el_min_deg must be finite, got {el_min_deg}")
        super().__init__()
        self._sc = spacecraft
        self._gs = ground_station
        self._el_min_deg = float(el_min_deg)
        self._el_min_rad = np.radians(self._el_min_deg)

    def __repr__(self) -> str:
        return (
            f"SpaceGroundAccessCondition("
            f"spacecraft={self._sc!r}, ground_station={self._gs!r}, "
            f"el_min_deg={self._el_min_deg})"
        )

    def _compute(self, t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
        r, _ = cached_propagate_analytical(
            t,
            **self._sc.keplerian_params,
            propagator_type=self._sc.propagator_type,
        )
        return earth_access(
            r,
            lat=np.radians(self._gs.lat),
            lon=np.radians(self._gs.lon),
            alt=self._gs.alt,
            el_min=self._el_min_rad,
            frame="eci",
            t=t,
        )


class SunlightCondition(AbstractCondition):
    """True when the object is in sunlight.

    Uses a cylindrical shadow model centred on the central body.  For a
    spacecraft the position is propagated analytically; for a ground
    station the fixed ECEF position is converted to ECI.

    Parameters
    ----------
    obj : Spacecraft | GroundStation
        The object whose sunlight status is being tested.

    Raises
    ------
    TypeError
        If *obj* is not a :class:`~missiontools.Spacecraft` or
        :class:`~missiontools.GroundStation`.
    """

    def __init__(self, obj) -> None:
        from ..spacecraft import Spacecraft
        from ..ground_station import GroundStation

        if isinstance(obj, Spacecraft):
            self._is_sc = True
        elif isinstance(obj, GroundStation):
            self._is_sc = False
        else:
            raise TypeError(
                f"obj must be a Spacecraft or GroundStation instance, "
                f"got {type(obj).__name__!r}"
            )
        super().__init__()
        self._obj = obj
        if not self._is_sc:
            self._gs_ecef = geodetic_to_ecef(
                np.radians(obj.lat),
                np.radians(obj.lon),
                obj.alt,
            )

    def __repr__(self) -> str:
        return f"SunlightCondition(obj={self._obj!r})"

    def _compute(self, t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
        if self._is_sc:
            r, _ = cached_propagate_analytical(
                t,
                **self._obj.keplerian_params,
                propagator_type=self._obj.propagator_type,
            )
            return in_sunlight(r, t, body_radius=self._obj.central_body_radius)
        else:
            r_eci = ecef_to_eci(
                np.broadcast_to(self._gs_ecef, (len(t), 3)),
                t,
            )
            return in_sunlight(r_eci, t)


class SubSatelliteRegionCondition(AbstractCondition):
    """True when the spacecraft's sub-satellite point falls inside an AoI.

    Parameters
    ----------
    spacecraft : Spacecraft
        The spacecraft whose sub-satellite point is tested.
    aoi : AoI
        Area of interest **with a geometry** (``aoi.geometry is not None``).

    Raises
    ------
    TypeError
        If *spacecraft* is not a :class:`~missiontools.Spacecraft`.
    ValueError
        If *aoi* does not have a geometry defined.
    """

    def __init__(self, spacecraft, aoi) -> None:
        from ..spacecraft import Spacecraft
        from ..aoi import AoI

        if not isinstance(spacecraft, Spacecraft):
            raise TypeError(
                f"spacecraft must be a Spacecraft instance, "
                f"got {type(spacecraft).__name__!r}"
            )
        if not isinstance(aoi, AoI):
            raise TypeError(f"aoi must be an AoI instance, got {type(aoi).__name__!r}")
        if aoi.geometry is None:
            raise ValueError(
                "aoi must have a geometry defined; construct with "
                "AoI.from_region, AoI.from_shapefile, or AoI.from_geography"
            )
        super().__init__()
        self._sc = spacecraft
        self._aoi = aoi

    def __repr__(self) -> str:
        return (
            f"SubSatelliteRegionCondition(spacecraft={self._sc!r}, aoi={self._aoi!r})"
        )

    def _compute(self, t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
        r, _ = cached_propagate_analytical(
            t,
            **self._sc.keplerian_params,
            propagator_type=self._sc.propagator_type,
        )
        r_ecef = eci_to_ecef(r, t)
        lat_rad, lon_rad, _ = ecef_to_geodetic(r_ecef)
        lat_deg = np.degrees(lat_rad)
        lon_deg = np.degrees(lon_rad)
        inside = shapely.contains_xy(self._aoi.geometry, lon_deg, lat_deg)
        crosses_am = (lon_deg.max() - lon_deg.min()) > 180
        if crosses_am:
            inside |= shapely.contains_xy(self._aoi.geometry, lon_deg + 360.0, lat_deg)
            inside |= shapely.contains_xy(self._aoi.geometry, lon_deg - 360.0, lat_deg)
        return inside


class VisibilityCondition(AbstractCondition):
    """True when two objects have unobstructed line-of-sight.

    Earth blockage is modelled as a sphere with radius equal to the
    spacecraft's ``central_body_radius`` (or the default mean Earth radius
    when neither object is a spacecraft).

    Parameters
    ----------
    obj1, obj2 : Spacecraft | GroundStation
        The two objects whose mutual visibility is tested.  At least one
        should be a :class:`~missiontools.Spacecraft` for meaningful
        results (ground-station-to-ground-station visibility is almost
        always blocked by Earth).

    Raises
    ------
    TypeError
        If either argument is not a :class:`~missiontools.Spacecraft` or
        :class:`~missiontools.GroundStation`.
    """

    def __init__(self, obj1, obj2) -> None:
        from ..spacecraft import Spacecraft
        from ..ground_station import GroundStation
        from ..orbit.constants import EARTH_MEAN_RADIUS

        for name, val in (("obj1", obj1), ("obj2", obj2)):
            if not isinstance(val, (Spacecraft, GroundStation)):
                raise TypeError(
                    f"{name} must be a Spacecraft or GroundStation instance, "
                    f"got {type(val).__name__!r}"
                )
        super().__init__()
        self._obj1 = obj1
        self._obj2 = obj2
        self._is_sc1 = isinstance(obj1, Spacecraft)
        self._is_sc2 = isinstance(obj2, Spacecraft)
        if not self._is_sc1:
            self._gs1_ecef = geodetic_to_ecef(
                np.radians(obj1.lat),
                np.radians(obj1.lon),
                obj1.alt,
            )
        if not self._is_sc2:
            self._gs2_ecef = geodetic_to_ecef(
                np.radians(obj2.lat),
                np.radians(obj2.lon),
                obj2.alt,
            )

        self._body_radius = EARTH_MEAN_RADIUS
        for obj in (obj1, obj2):
            if isinstance(obj, Spacecraft):
                self._body_radius = obj.central_body_radius
                break

    def __repr__(self) -> str:
        return f"VisibilityCondition(obj1={self._obj1!r}, obj2={self._obj2!r})"

    def _compute(self, t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
        if self._is_sc1:
            r1, _ = cached_propagate_analytical(
                t,
                **self._obj1.keplerian_params,
                propagator_type=self._obj1.propagator_type,
            )
        else:
            r1 = ecef_to_eci(
                np.broadcast_to(self._gs1_ecef, (len(t), 3)),
                t,
            )

        if self._is_sc2:
            r2, _ = cached_propagate_analytical(
                t,
                **self._obj2.keplerian_params,
                propagator_type=self._obj2.propagator_type,
            )
        else:
            r2 = ecef_to_eci(
                np.broadcast_to(self._gs2_ecef, (len(t), 3)),
                t,
            )

        return space_to_space_access(r1, r2, body_radius=self._body_radius)


class AndCondition(AbstractCondition):
    """True when both child conditions are true (logical AND).

    Parameters
    ----------
    condition1, condition2 : AbstractCondition
        Child conditions.

    Raises
    ------
    TypeError
        If either argument is not an :class:`AbstractCondition`.
    """

    def __init__(
        self, condition1: AbstractCondition, condition2: AbstractCondition
    ) -> None:
        for name, val in (("condition1", condition1), ("condition2", condition2)):
            if not isinstance(val, AbstractCondition):
                raise TypeError(
                    f"{name} must be an AbstractCondition instance, "
                    f"got {type(val).__name__!r}"
                )
        super().__init__(cache_size=0)
        self._c1 = condition1
        self._c2 = condition2

    def __repr__(self) -> str:
        return f"AndCondition({self._c1!r}, {self._c2!r})"

    def _compute(self, t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
        return self._c1.at(t) & self._c2.at(t)


class OrCondition(AbstractCondition):
    """True when either child condition is true (logical OR).

    Parameters
    ----------
    condition1, condition2 : AbstractCondition
        Child conditions.

    Raises
    ------
    TypeError
        If either argument is not an :class:`AbstractCondition`.
    """

    def __init__(
        self, condition1: AbstractCondition, condition2: AbstractCondition
    ) -> None:
        for name, val in (("condition1", condition1), ("condition2", condition2)):
            if not isinstance(val, AbstractCondition):
                raise TypeError(
                    f"{name} must be an AbstractCondition instance, "
                    f"got {type(val).__name__!r}"
                )
        super().__init__(cache_size=0)
        self._c1 = condition1
        self._c2 = condition2

    def __repr__(self) -> str:
        return f"OrCondition({self._c1!r}, {self._c2!r})"

    def _compute(self, t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
        return self._c1.at(t) | self._c2.at(t)


class NotCondition(AbstractCondition):
    """True when the child condition is false (logical NOT).

    Parameters
    ----------
    condition : AbstractCondition
        Child condition to invert.

    Raises
    ------
    TypeError
        If *condition* is not an :class:`AbstractCondition`.
    """

    def __init__(self, condition: AbstractCondition) -> None:
        if not isinstance(condition, AbstractCondition):
            raise TypeError(
                f"condition must be an AbstractCondition instance, "
                f"got {type(condition).__name__!r}"
            )
        super().__init__(cache_size=0)
        self._c = condition

    def __repr__(self) -> str:
        return f"NotCondition({self._c!r})"

    def _compute(self, t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
        return ~self._c.at(t)


class XorCondition(AbstractCondition):
    """True when exactly one child condition is true (logical XOR).

    Parameters
    ----------
    condition1, condition2 : AbstractCondition
        Child conditions.

    Raises
    ------
    TypeError
        If either argument is not an :class:`AbstractCondition`.
    """

    def __init__(
        self, condition1: AbstractCondition, condition2: AbstractCondition
    ) -> None:
        for name, val in (("condition1", condition1), ("condition2", condition2)):
            if not isinstance(val, AbstractCondition):
                raise TypeError(
                    f"{name} must be an AbstractCondition instance, "
                    f"got {type(val).__name__!r}"
                )
        super().__init__(cache_size=0)
        self._c1 = condition1
        self._c2 = condition2

    def __repr__(self) -> str:
        return f"XorCondition({self._c1!r}, {self._c2!r})"

    def _compute(self, t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.bool_]:
        return self._c1.at(t) ^ self._c2.at(t)
