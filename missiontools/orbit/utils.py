from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .frames import geodetic_to_ecef, ecef_to_eci
from .propagation import propagate_analytical


def host_eci_state(
    host,
    t_arr: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    from ..spacecraft import Spacecraft
    from ..ground_station import GroundStation

    if isinstance(host, Spacecraft):
        r, v = propagate_analytical(
            t_arr, **host.keplerian_params, propagator_type=host.propagator_type
        )
        return r, v
    elif isinstance(host, GroundStation):
        r_ecef = geodetic_to_ecef(np.radians(host.lat), np.radians(host.lon), host.alt)
        n = len(t_arr)
        r_ecef_tiled = np.tile(r_ecef, (n, 1))
        r_eci = ecef_to_eci(r_ecef_tiled, t_arr)
        v_eci = np.zeros((n, 3))
        return r_eci, v_eci
    else:
        raise TypeError(f"Unsupported host type: {type(host).__name__!r}")
