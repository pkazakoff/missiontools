"""
missiontools.plotting._map
==========================
Shared Cartopy map-setup helpers.
"""
from __future__ import annotations

import numpy as np


def _try_cartopy():
    """Import cartopy or raise a clear ImportError."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except ImportError:
        raise ImportError(
            "missiontools.plotting requires cartopy. "
            "Install it with:  pip install missiontools[plot]"
        ) from None


def _new_map_ax(ax=None, projection=None):
    """Return a GeoAxes decorated with coastlines, borders, and gridlines.

    If *ax* is ``None``, a new figure and GeoAxes are created using
    *projection* (default: ``ccrs.PlateCarree()`` — WGS-84 equirectangular).
    The axes extent is set to the full Earth.

    Parameters
    ----------
    ax : GeoAxes, optional
        Existing axes to decorate.  If provided, *projection* is ignored.
    projection : cartopy CRS, optional
        Map projection for the new axes.  Default ``ccrs.PlateCarree()``.

    Returns
    -------
    GeoAxes
    """
    ccrs, cfeature = _try_cartopy()
    import matplotlib.pyplot as plt

    if projection is None:
        projection = ccrs.PlateCarree()

    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': projection})

    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    return ax


def _set_extent(ax, lat, lon, factor: float = 1.5) -> None:
    """Auto-window the axes to *factor* × the data bounding box.

    The bounding box is computed in the axes' native projected coordinate
    system, so padding is symmetric in projected units (metres for conic /
    cylindrical projections, degrees for PlateCarree).  This ensures the
    window is not artificially clipped or skewed for non-equirectangular
    projections.

    Parameters
    ----------
    ax : GeoAxes
    lat : ndarray, degrees
    lon : ndarray, degrees
    factor : float
        Window size as a multiple of the data range (default 1.5).
    """
    ccrs, _ = _try_cartopy()

    # Project the scattered points into the axes' native CRS
    proj = ax.projection
    pts  = proj.transform_points(
        ccrs.PlateCarree(),
        np.asarray(lon, dtype=float),
        np.asarray(lat, dtype=float),
    )
    x, y = pts[:, 0], pts[:, 1]

    # Discard any points that fell outside the projection's valid domain
    valid = np.isfinite(x) & np.isfinite(y)
    x, y  = x[valid], y[valid]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    pad_x = (factor - 1) / 2 * max(x_max - x_min, 1.0)
    pad_y = (factor - 1) / 2 * max(y_max - y_min, 1.0)

    # Clamp to the projection's renderable limits.  Cartopy draws nothing
    # (no coastlines, gridlines, or features) outside these bounds, so any
    # viewport that extends beyond them only adds blank white space.
    x_lo, x_hi = ax.projection.x_limits
    y_lo, y_hi = ax.projection.y_limits

    ax.set_xlim(max(x_min - pad_x, x_lo), min(x_max + pad_x, x_hi))
    ax.set_ylim(max(y_min - pad_y, y_lo), min(y_max + pad_y, y_hi))
