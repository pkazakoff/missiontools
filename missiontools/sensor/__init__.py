"""
missiontools.sensor
===================
Instrument sensor classes for spacecraft field-of-view modelling.

The base class is :class:`AbstractSensor`, with concrete subclasses for
each sensor geometry:

- :class:`ConicSensor` — sensor with a conical field of view, defined by a
  half-angle.  The boresight can be driven by an independent
  :class:`~missiontools.AbstractAttitudeLaw` or fixed in the spacecraft body
  frame.
- :class:`RectangularSensor` — sensor with a rectangular field of view,
  defined by two independent half-angles (theta1, theta2).  Supports the
  same pointing modes as :class:`ConicSensor` plus an optional roll
  constraint vector for body-mounted sensors.
"""

from .sensor_law import AbstractSensor, ConicSensor, RectangularSensor

__all__ = ['AbstractSensor', 'ConicSensor', 'RectangularSensor']
