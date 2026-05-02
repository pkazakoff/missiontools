"""
missiontools.coverage
=====================
Coverage and access analysis, and geographic area sampling.
"""

from .coverage import (
    sample_aoi,
    sample_region,
    sample_shapefile,
    sample_geography,
    coverage_fraction,
    revisit_time,
    pointwise_coverage,
    access_pointwise,
    revisit_pointwise,
    coverage_fraction_multi,
    pointwise_coverage_multi,
)
from .sampling import (
    sample_from_geometry,
    load_shapefile_geometry,
    geography_geometry,
)
from .visibility import (
    SensorSpec,
    make_sensor_spec,
    make_sensor_spec_from_fov,
    CoverageConstraints,
    collect_access_intervals_multi,
)

__all__ = [
    "sample_aoi",
    "sample_region",
    "sample_shapefile",
    "sample_geography",
    "coverage_fraction",
    "revisit_time",
    "pointwise_coverage",
    "access_pointwise",
    "revisit_pointwise",
    "SensorSpec",
    "make_sensor_spec",
    "make_sensor_spec_from_fov",
    "CoverageConstraints",
    "coverage_fraction_multi",
    "pointwise_coverage_multi",
    "collect_access_intervals_multi",
    "load_shapefile_geometry",
    "sample_from_geometry",
    "geography_geometry",
]
