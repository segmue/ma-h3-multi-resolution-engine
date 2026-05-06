# h3-multi-resolution-index

Converts vector geometries to H3 cells and provides spatial predicates across resolution levels. DuckDB backend.

## Conversion

Geometries → H3 cells, with adaptive resolution for polygons (chosen by area to hit a target cell count). Points/lines get a fixed resolution.

```python
from h3_multi_resolution_engine import convert_geometry_to_h3

# Point → single cell at max resolution
cells, res = convert_geometry_to_h3(Point(8.54, 47.37))

# Polygon → adaptive resolution based on area
cells, res = convert_geometry_to_h3(polygon, source_crs=2056, target_cells=1000)
# small polygon → fine resolution, large polygon → coarse resolution

# Batch from GeoDataFrame
cells_list, res_list = convert_geodataframe_to_h3(gdf, target_cells=1000)
```

## Querying

Once stored in DuckDB, `H3Engine` provides lazy set operations and eager predicates. Cross-resolution comparison via `h3_cell_to_parent()`.

```python
from h3_multi_resolution_engine import H3Engine

db = H3Engine("spatial.duckdb")
wald = db.union("OBJEKTART = 'Wald'")
seen = db.union("OBJEKTART = 'See'")

db.intersects(wald, seen)                    # bool
db.area(db.intersection(wald, seen))         # float (km²)
db.find_overlapping_features(feature_id=42, objektart="Gemeindegebiet")
```

## DuckDB Schema

```sql
features (feature_id PK, UUID, NAME, OBJEKTART, source, h3_cells UBIGINT[], h3_resolution, h3_cell_count)
h3_lookup (feature_id, cell UBIGINT, cell_res TINYINT)
```

## Dependencies

h3, duckdb, shapely, geopandas, pyproj
