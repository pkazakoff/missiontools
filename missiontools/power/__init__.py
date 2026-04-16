"""
missiontools.power
==================
Solar power generation modelling.

Classes
-------
:class:`AbstractSolarConfig`
    Abstract base class for solar power configurations.  Subclass this and
    implement :meth:`~AbstractSolarConfig.generation`,
    :meth:`~AbstractSolarConfig.optimal_angle`, and
    :meth:`~AbstractSolarConfig.oap` to add custom panel geometries.
:class:`NormalVectorSolarConfig`
    Concrete solar config defined by panel outward-normal vectors and
    areas.  Instantaneous power for each panel is

    .. math::

        P_k = I \\cdot A_k \\cdot \\eta \\cdot
              \\max\\!\\left(\\hat{n}_{k,\\mathrm{ECI}} \\cdot
                          \\hat{s}_{\\mathrm{ECI}},\\; 0\\right)

    where :math:`I` is the solar irradiance (W m⁻²), :math:`A_k` is the
    panel area (m²), :math:`\\eta` is the conversion efficiency,
    :math:`\\hat{n}_{k,\\mathrm{ECI}}` is the panel normal rotated into ECI
    via the spacecraft attitude law, and :math:`\\hat{s}_{\\mathrm{ECI}}` is
    the unit vector toward the Sun.  Power is zero in eclipse.

Usage
-----
Solar configs are attached to a :class:`~missiontools.Spacecraft` via
:meth:`~missiontools.Spacecraft.add_solar_config`.  They can also drive
yaw steering via :meth:`~missiontools.AbstractAttitudeLaw.yaw_steering`::

    import numpy as np
    from missiontools import Spacecraft, NormalVectorSolarConfig

    sc = Spacecraft.sunsync(altitude_km=550, node_solar_time='10:30')

    # Two body panels facing ±Y in the body frame, 0.3 m² each, 30% efficiency
    solar = NormalVectorSolarConfig(
        normal_vecs=[[0, 1, 0], [0, -1, 0]],
        areas=[0.3, 0.3],
        efficiency=0.30,
    )
    sc.add_solar_config(solar)

    result = solar.generation(
        np.datetime64('2025-01-01', 'us'),
        np.datetime64('2025-01-02', 'us'),
        np.timedelta64(60, 's'),
    )
    # result['power'] — (N,) instantaneous total power (W)
    print(f"Orbit-average power: {solar.oap():.1f} W")

Planned functionality
---------------------
- Battery sizing and depth of discharge
- Eclipse duration and energy balance
- End-of-life power degradation models
- Power mode and duty cycle analysis
"""

from .solar_config import AbstractSolarConfig, NormalVectorSolarConfig

__all__ = ['AbstractSolarConfig', 'NormalVectorSolarConfig']
