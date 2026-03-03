"""
missiontools.plotting.coverage_map
===================================
Interpolated heatmap visualisation of per-point coverage values.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._map import _try_cartopy, _new_map_ax, _set_extent


def plot_coverage_map(
        aoi,
        values: npt.ArrayLike,
        *,
        ax=None,
        projection=None,
        auto_window:    bool        = False,
        cmap:           str         = 'viridis',
        vmin:           float | None = None,
        vmax:           float | None = None,
        colorbar:       bool         = True,
        colorbar_label: str          = '',
        title:          str          = '',
        grid_resolution: int         = 200,
):
    """Interpolated heatmap of per-point values from an AoI on an Earth map.

    Parameters
    ----------
    aoi : AoI
        Area of interest (provides lat/lon of ground sample points).
    values : array_like, shape (M,)
        Per-point scalar values — e.g. coverage fraction, revisit time,
        or any quantity aligned with the AoI sample points.
    ax : GeoAxes, optional
        Existing Cartopy GeoAxes.  A new figure is created if ``None``.
    projection : cartopy CRS, optional
        Map projection for the new axes.  Default
        ``ccrs.PlateCarree()`` (WGS-84 equirectangular).  Ignored if *ax*
        is provided.
    auto_window : bool
        If ``True``, set the axes extent to 1.5× the AoI lat/lon bounding
        box.
    cmap : str
        Matplotlib colormap name (default ``'viridis'``).
    vmin, vmax : float, optional
        Colormap limits.  Defaults to the data min/max (ignoring NaN).
    colorbar : bool
        Add a colorbar to the figure (default ``True``).
    colorbar_label : str
        Label for the colorbar axis.
    title : str
        Axes title.
    grid_resolution : int
        Number of grid points per axis for the interpolation grid
        (default 200).

    Returns
    -------
    GeoAxes
        The axes on which the map was drawn.

    Raises
    ------
    ValueError
        If ``len(values) != len(aoi)``.

    Examples
    --------
    ::

        import numpy as np
        from missiontools import Spacecraft, Sensor, AoI, Coverage
        from missiontools.plotting import plot_coverage_map

        sc     = Spacecraft(a=6_771_000., e=0., i=np.radians(51.6),
                            raan=0., arg_p=0., ma=0.,
                            epoch=np.datetime64('2025-01-01', 'us'))
        sensor = Sensor(30.0, body_vector=[0, 0, 1])
        sc.add_sensor(sensor)

        aoi = AoI.from_region(-60, 60, -180, 180)
        cov = Coverage(aoi, [sensor])

        result = cov.coverage_fraction(
            np.datetime64('2025-01-01', 'us'),
            np.datetime64('2025-01-02', 'us'),
        )
        ax = plot_coverage_map(aoi, result['final_cumulative'],
                               colorbar_label='Coverage fraction',
                               title='24-hour coverage')
    """
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt

    ccrs, _ = _try_cartopy()
    ax = _new_map_ax(ax, projection)

    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(aoi):
        raise ValueError(
            f"len(values) = {len(values)} does not match len(aoi) = {len(aoi)}"
        )

    lat_deg = aoi.lat   # degrees
    lon_deg = aoi.lon   # degrees

    if auto_window:
        _set_extent(ax, lat_deg, lon_deg)

    # Transform AoI sample points into the axes' native projected CRS so that
    # the interpolation grid is rectangular in display space.  A lat/lon grid
    # maps to a curved trapezoid in conic projections, leaving corners empty.
    proj = ax.projection
    pts  = proj.transform_points(ccrs.PlateCarree(), lon_deg, lat_deg)
    x_pts, y_pts = pts[:, 0], pts[:, 1]
    valid    = np.isfinite(x_pts) & np.isfinite(y_pts)
    x_pts    = x_pts[valid]
    y_pts    = y_pts[valid]
    vals_valid = values[valid]

    x_min, x_max = float(x_pts.min()), float(x_pts.max())
    y_min, y_max = float(y_pts.min()), float(y_pts.max())

    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    x_mg, y_mg = np.meshgrid(x_grid, y_grid)

    # Interpolate in projected coordinates (NaN outside convex hull)
    z = griddata(
        (x_pts, y_pts),
        vals_valid,
        (x_mg, y_mg),
        method='linear',
    )

    # Data is already in the projection's coordinate system
    mesh = ax.pcolormesh(
        x_mg, y_mg, z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=proj,
    )

    if colorbar:
        plt.colorbar(mesh, ax=ax, label=colorbar_label, shrink=0.7)

    if title:
        ax.set_title(title)

    return ax
