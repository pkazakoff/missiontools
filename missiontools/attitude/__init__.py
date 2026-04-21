"""
missiontools.attitude
=====================
Spacecraft and sensor pointing laws.

The base class is :class:`AbstractAttitudeLaw`, with concrete subclasses
for each pointing mode:

- :class:`FixedAttitudeLaw` — constant body orientation in a chosen
  reference frame (LVLH, ECI, or ECEF).  The convenience classmethod
  :meth:`~FixedAttitudeLaw.nadir` creates the most common configuration
  (body-z toward the Earth centre).
- :class:`TrackAttitudeLaw` — boresight pointing toward a target
  :class:`~missiontools.Spacecraft` at every timestep.
- :class:`CustomAttitudeLaw` — full 3-DOF control via a user-supplied
  quaternion callback.
- :class:`LimbAttitudeLaw` — body-frame vector aligned with the ray
  grazing an offset ellipsoid (limb pointing).

All pointing methods (:meth:`~AbstractAttitudeLaw.pointing_eci`,
:meth:`~AbstractAttitudeLaw.pointing_lvlh`,
:meth:`~AbstractAttitudeLaw.pointing_ecef`)
return the **body-z** unit vector expressed in the requested frame.

Optional yaw steering can be enabled via
:meth:`~AbstractAttitudeLaw.yaw_steering` on :class:`FixedAttitudeLaw`
and :class:`TrackAttitudeLaw` to maximise solar power generation by
rotating the spacecraft about the boresight axis at each timestep.

Planned functionality
---------------------
- Environmental disturbance torques
- Actuator sizing (reaction wheels, magnetorquers, thrusters)
- Pointing budget and error analysis
- Sensor modelling (star tracker, sun sensor, magnetometer)
"""

from .attitude_law import (AbstractAttitudeLaw,
                           FixedAttitudeLaw,
                           TrackAttitudeLaw,
                           CustomAttitudeLaw,
                           LimbAttitudeLaw)

__all__ = ['AbstractAttitudeLaw',
           'FixedAttitudeLaw',
           'TrackAttitudeLaw',
           'CustomAttitudeLaw',
           'LimbAttitudeLaw']
