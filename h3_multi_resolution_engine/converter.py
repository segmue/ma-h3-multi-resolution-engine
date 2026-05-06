"""
Vector to DGGS (H3) conversion functions.

Supports conversion of points, lines, and polygons to H3 cells with
automatic coordinate transformation.
"""

from typing import Set, List, Union, Optional, Any
from enum import Enum
import h3
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely.ops import transform
from pyproj import Transformer, CRS, Geod


# =============================================================================
# CONSTANTS
# =============================================================================

class ContainmentMode(Enum):
    """
    H3 containment modes for polygon_to_cells operations.

    The experimental API (h3shape_to_cells_experimental) supports all 4 modes.
    The standard API (polygon_to_cells) only supports CENTER.
    """
    CENTER = "center"              # Cell center must be inside polygon (default classic)
    FULL = "full"                  # Cell must be fully contained in polygon
    OVERLAPPING = "overlap"        # Cell overlaps polygon at any point (recommended)
    OVERLAPPING_BBOX = "overlap_bbox"  # Cell bounding box overlaps polygon


# Average hexagon areas in m² per H3 resolution level
# Source: https://h3geo.org/docs/core-library/restable/
H3_AVG_HEXAGON_AREA_M2: dict[int, float] = {
    0:  4_357_449_416_078.392,
    1:    609_788_441_794.134,
    2:     86_801_780_398.997,
    3:     12_393_434_655.088,
    4:      1_770_347_654.491,
    5:        252_903_858.182,
    6:         36_129_062.164,
    7:          5_161_293.360,
    8:            737_327.598,
    9:            105_332.513,
    10:            15_047.502,
    11:             2_149.643,
    12:               307.092,
    13:                43.870,
    14:                 6.267,
    15:                 0.895,
}

# WGS84 ellipsoid for geodesic area calculations
_WGS84_GEOD = Geod(ellps='WGS84')

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


def _ensure_wgs84(geometry, source_crs: Optional[Union[str, int]] = None):
    """
    Transform geometry to WGS84 if needed.

    Args:
        geometry: Shapely geometry object
        source_crs: Source CRS (EPSG code as int, string like 'EPSG:2056', or None for WGS84)

    Returns:
        Transformed geometry in WGS84
    """
    if source_crs is None:
        return geometry

    # Handle different CRS input formats
    if isinstance(source_crs, int):
        source_crs = f"EPSG:{source_crs}"

    # Create transformer
    transformer = Transformer.from_crs(
        CRS.from_user_input(source_crs),
        CRS.from_epsg(4326),  # WGS84
        always_xy=True
    )

    return transform(transformer.transform, geometry)


def _polygon_to_h3_cells(
    polygon: Polygon,
    resolution: int,
    containment_mode: ContainmentMode = ContainmentMode.OVERLAPPING
) -> Set[str]:
    """
    Helper function to convert a single polygon to H3 cells at given resolution.

    Uses the experimental H3 API (h3shape_to_cells_experimental) which supports
    different containment modes. Falls back to standard API if experimental fails.
    """
    exterior_coords = [(coord[1], coord[0]) for coord in polygon.exterior.coords]

    holes = []
    for interior in polygon.interiors:
        hole_coords = [(coord[1], coord[0]) for coord in interior.coords]
        holes.append(hole_coords)

    if holes:
        h3_poly = h3.LatLngPoly(exterior_coords, *holes)
    else:
        h3_poly = h3.LatLngPoly(exterior_coords)

    # Try experimental API first (supports all containment modes)
    try:
        return h3.h3shape_to_cells_experimental(h3_poly, resolution, contain=containment_mode.value)
    except (AttributeError, TypeError, Exception) as e:
        if containment_mode != ContainmentMode.CENTER:
            print(f"[_polygon_to_h3_cells] Warning: h3shape_to_cells_experimental failed ({e}). "
                  f"Falling back to standard polygon_to_cells (CENTER mode only).")
        return h3.polygon_to_cells(h3_poly, resolution)


def _calculate_geodesic_area_m2(polygon: Union[Polygon, MultiPolygon]) -> float:
    """
    Calculate the geodesic area of a polygon in square meters using WGS84 ellipsoid.
    """
    area, _ = _WGS84_GEOD.geometry_area_perimeter(polygon)
    return abs(area)


def _estimate_resolution_from_area(
    polygon_area_m2: float,
    target_cells: int,
    min_resolution: int,
    max_resolution: int
) -> int:
    """
    Estimate optimal H3 resolution based on polygon area and target cell count.
    """
    target_cell_area = polygon_area_m2 / target_cells

    for res in range(min_resolution, max_resolution + 1):
        if H3_AVG_HEXAGON_AREA_M2[res] <= target_cell_area:
            return res

    return max_resolution


def _calculate_optimal_resolution(
    polygon: Union[Polygon, MultiPolygon],
    target_cells: int,
    min_resolution: int,
    max_resolution: int,
    containment_mode: ContainmentMode
) -> Optional[int]:
    """
    Calculate optimal H3 resolution for a polygon (already in WGS84).

    Returns None if polygon too small (use centroid fallback).
    """
    polygon_area_m2 = _calculate_geodesic_area_m2(polygon)

    estimated_resolution = _estimate_resolution_from_area(
        polygon_area_m2, target_cells, min_resolution, max_resolution
    )

    if isinstance(polygon, MultiPolygon):
        h3_cells = set()
        for poly in polygon.geoms:
            h3_cells.update(_polygon_to_h3_cells(poly, estimated_resolution, containment_mode))
    else:
        h3_cells = _polygon_to_h3_cells(polygon, estimated_resolution, containment_mode)

    num_cells = len(h3_cells)

    if num_cells < target_cells and estimated_resolution < max_resolution:
        finer_resolution = estimated_resolution + 1

        if isinstance(polygon, MultiPolygon):
            finer_cells = set()
            for poly in polygon.geoms:
                finer_cells.update(_polygon_to_h3_cells(poly, finer_resolution, containment_mode))
        else:
            finer_cells = _polygon_to_h3_cells(polygon, finer_resolution, containment_mode)

        return finer_resolution

    if num_cells == 0:
        return None

    return estimated_resolution


def _point_to_h3(point: Point, resolution: int) -> str:
    """Convert a single Point (already in WGS84) to H3 cell ID."""
    lng, lat = point.x, point.y
    return h3.latlng_to_cell(lat, lng, resolution)


def _line_to_h3(line: LineString, resolution: int) -> Set[str]:
    """Convert a single LineString (already in WGS84) to H3 cells."""
    h3_cells = set()
    coords = list(line.coords)

    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]

        cell_start = h3.latlng_to_cell(p1[1], p1[0], resolution)
        cell_end = h3.latlng_to_cell(p2[1], p2[0], resolution)

        try:
            path = h3.grid_path_cells(cell_start, cell_end)
            h3_cells.update(path)
        except Exception:
            h3_cells.add(cell_start)
            h3_cells.add(cell_end)

    return h3_cells


def _polygon_to_h3(
    polygon: Polygon,
    resolution: int,
    containment_mode: ContainmentMode = ContainmentMode.OVERLAPPING
) -> Set[str]:
    """Convert a single Polygon (already in WGS84) to H3 cells."""
    h3_cells = _polygon_to_h3_cells(polygon, resolution, containment_mode)

    if not h3_cells:
        centroid = polygon.centroid
        lng, lat = centroid.x, centroid.y
        cell_id = h3.latlng_to_cell(lat, lng, resolution)
        return {cell_id}

    return set(h3_cells)


def _polygon_to_h3_adaptive(
    polygon: Union[Polygon, MultiPolygon],
    target_cells: int,
    min_resolution: int,
    max_resolution: int,
    containment_mode: ContainmentMode
) -> tuple[Set[str], int]:
    """Convert Polygon or MultiPolygon (already in WGS84) to H3 cells with adaptive resolution."""
    optimal_res = _calculate_optimal_resolution(
        polygon, target_cells, min_resolution, max_resolution,
        containment_mode=containment_mode
    )

    if optimal_res is None:
        centroid = polygon.centroid
        lng, lat = centroid.x, centroid.y
        cell_id = h3.latlng_to_cell(lat, lng, max_resolution)
        return {cell_id}, max_resolution

    if isinstance(polygon, MultiPolygon):
        all_cells = set()
        for poly in polygon.geoms:
            all_cells.update(_polygon_to_h3(poly, optimal_res, containment_mode))
        return all_cells, optimal_res
    else:
        return _polygon_to_h3(polygon, optimal_res, containment_mode), optimal_res


def convert_geometry_to_h3(
    geometry: Union[Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon],
    target_cells: int = 1000,
    min_resolution: int = 5,
    max_resolution: int = 12,
    source_crs: Optional[Union[str, int]] = None,
    containment_mode: ContainmentMode = ContainmentMode.OVERLAPPING
) -> tuple[Set[str], int]:
    """
    Convert any geometry type to H3 cell IDs with adaptive resolution for polygons.

    - Points/Lines: Use max_resolution (fixed)
    - Polygons/MultiPolygons: Use adaptive resolution based on area

    Args:
        geometry: Shapely geometry object
        target_cells: Target number of H3 cells for polygons (default: 1000)
        min_resolution: Minimum resolution for polygons (default: 5)
        max_resolution: Resolution for points/lines, max for polygons (default: 12)
        source_crs: Source CRS (e.g., 2056 for LV95, None for WGS84)
        containment_mode: ContainmentMode for polygons (default: OVERLAPPING)

    Returns:
        Tuple of (set of H3 cell IDs, resolution used)

    Raises:
        ValueError: If geometry type is not supported
    """
    geometry_wgs84 = _ensure_wgs84(geometry, source_crs)

    if isinstance(geometry_wgs84, Point):
        return {_point_to_h3(geometry_wgs84, max_resolution)}, max_resolution

    elif isinstance(geometry_wgs84, LineString):
        return _line_to_h3(geometry_wgs84, max_resolution), max_resolution

    elif isinstance(geometry_wgs84, (Polygon, MultiPolygon)):
        return _polygon_to_h3_adaptive(
            geometry_wgs84, target_cells, min_resolution, max_resolution, containment_mode
        )

    elif isinstance(geometry_wgs84, MultiPoint):
        cells = set()
        for point in geometry_wgs84.geoms:
            cells.add(_point_to_h3(point, max_resolution))
        return cells, max_resolution

    elif isinstance(geometry_wgs84, MultiLineString):
        cells = set()
        for line in geometry_wgs84.geoms:
            cells.update(_line_to_h3(line, max_resolution))
        return cells, max_resolution

    else:
        raise ValueError(f"Unsupported geometry type: {type(geometry)}")


def convert_geodataframe_to_h3(
    gdf: Any,
    target_cells: int = 1000,
    min_resolution: int = 5,
    max_resolution: int = 12,
    geometry_column: str = 'geometry',
    containment_mode: ContainmentMode = ContainmentMode.OVERLAPPING
) -> tuple[List[Set[str]], List[int]]:
    """
    Batch conversion of GeoDataFrame geometries to H3 cells.

    Each geometry gets its own optimal resolution based on size:
    - Polygons: Adaptive resolution to achieve ~target_cells
    - Points/Lines: Fixed max_resolution

    Args:
        gdf: GeoDataFrame with geometries to convert
        target_cells: Target number of cells per polygon (default: 1000)
        min_resolution: Minimum resolution for polygons (default: 5)
        max_resolution: Resolution for points/lines, max for polygons (default: 12)
        geometry_column: Name of the geometry column (default: 'geometry')
        containment_mode: ContainmentMode for polygons (default: OVERLAPPING)

    Returns:
        Tuple of (list of H3 cell sets, list of resolutions used)
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for convert_geodataframe_to_h3")

    # Batch transform to WGS84 once (performance optimization)
    gdf_wgs84 = gdf.to_crs(epsg=4326)

    h3_cells_list = []
    resolutions_list = []

    for geom in gdf_wgs84[geometry_column]:
        cells, resolution = convert_geometry_to_h3(
            geom,
            target_cells=target_cells,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
            source_crs=None,  # Already WGS84
            containment_mode=containment_mode
        )
        h3_cells_list.append(cells)
        resolutions_list.append(resolution)

    return h3_cells_list, resolutions_list
