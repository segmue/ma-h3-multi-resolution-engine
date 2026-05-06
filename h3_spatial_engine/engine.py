"""
H3Engine - DuckDB-basierte Spatial Predicates fuer H3 Cells.

Alle Predicates nutzen DuckDB's H3 Extension fuer Resolution-Normalisierung
via h3_cell_to_parent(). H3 Cells werden als UBIGINT[] Arrays in der
features Tabelle gespeichert und bei Bedarf via UNNEST expandiert.

Geometrien werden als GEOMETRY-Typ gespeichert (via DuckDB Spatial Extension),
was native Spatial Indexing und ST_* Funktionen ermoeglicht.

Architektur (Lazy Evaluation):
    - FeatureSet: SQL WHERE-String oder DuckDBPyRelation mit h3_cells Arrays
    - CellSet: Lazy Query-Plan mit SQL-String und Resolution

    Workflow: FeatureSet -> union() -> CellSet -> intersection/intersects/... -> CellSet
    Ausfuehrung: area(CellSet) oder CellSet.run() materialisiert die Query

Verwendung:
    from h3_spatial_engine import H3Engine

    db = H3Engine("data.duckdb")

    # DuckDB Relational API fuer allgemeine Queries
    wald = db.features.filter("OBJEKTART = 'Wald'")
    wald.aggregate("count(*)").df()

    # FeatureSet -> CellSet via union() (LAZY - keine Ausfuehrung)
    cells_wald = db.union("OBJEKTART = 'Wald'")
    cells_seen = db.union("OBJEKTART = 'See'")

    # Set-Operationen (LAZY - baut nur Query-Plan)
    result = db.intersection(cells_wald, cells_seen)

    # Ausfuehrung (EAGER - fuehrt Query aus)
    db.area(result)                    # Gibt float zurueck
    db.intersects(cells_wald, cells_seen)  # Gibt bool zurueck
    result.run()                       # Gibt DuckDBPyRelation zurueck
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union, TYPE_CHECKING

import duckdb

if TYPE_CHECKING:
    from h3_spatial_engine import H3Engine as _H3Engine

# Typ-Alias fuer Predicate-Argumente: SQL-String oder DuckDB Relation
FeatureSet = Union[str, duckdb.DuckDBPyRelation]


@dataclass
class CellSet:
    """Lazy Query-Plan fuer H3 Cells.

    Enthaelt einen SQL-String der erst bei run()/area()/etc. ausgefuehrt wird.
    Ermoeglicht Query-Komposition ohne Zwischenmaterialisierung.

    Attributes:
        sql: SQL-Query String (SELECT cell, resolution FROM ...)
        resolution: Die (einheitliche) Resolution dieses CellSets
        _engine: Referenz zur H3Engine fuer Query-Ausfuehrung
    """

    sql: str
    resolution: int
    _engine: "H3Engine"

    def run(self) -> duckdb.DuckDBPyRelation:
        """Fuehrt die Query aus und gibt eine DuckDBPyRelation zurueck."""
        return self._engine.conn.sql(self.sql)

    def df(self):
        """Fuehrt die Query aus und gibt einen pandas DataFrame zurueck."""
        return self.run().df()

    def count(self) -> int:
        """Zaehlt die Anzahl Cells (fuehrt Query aus)."""
        result = self._engine.conn.execute(f"""
            SELECT COUNT(*) FROM ({self.sql})
        """).fetchone()
        return result[0]

    def __len__(self) -> int:
        """Zaehlt die Anzahl Cells (fuehrt Query aus)."""
        return self.count()

    def __repr__(self) -> str:
        return f"CellSet(resolution={self.resolution}, sql='{self.sql[:50]}...')"


class H3Engine:
    """DuckDB-basierte H3 Spatial Query Engine.

    Kombiniert DuckDB's Relational API (fuer allgemeine Queries) mit
    spezialisierten H3 Spatial Predicates (intersects, within, contains,
    intersection).

    Attributes:
        conn: DuckDB Connection (direkt nutzbar fuer Raw SQL)
    """

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialisiert die Engine mit einer DuckDB Datenbank.

        Args:
            db_path: Pfad zur DuckDB Datei (von import_to_duckdb.py erstellt)
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Datenbank nicht gefunden: {db_path}")

        self.conn = duckdb.connect(str(self.db_path), read_only=True)
        self.conn.execute("INSTALL spatial; LOAD spatial;")
        self.conn.execute("INSTALL h3 FROM community; LOAD h3;")

    def close(self):
        """Schliesst die Datenbankverbindung."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -------------------------------------------------------------------------
    # DuckDB Relational API
    # -------------------------------------------------------------------------

    @property
    def features(self) -> duckdb.DuckDBPyRelation:
        """DuckDB Relation auf die features Tabelle.

        Gibt ein DuckDBPyRelation zurueck -- DuckDB's volle Relational API
        ist direkt verfuegbar:

            db.features.filter("OBJEKTART = 'Wald'").aggregate("count(*)").df()
            db.features.project("NAME, h3_resolution").order("NAME").limit(10).df()

        Kann auch als Input fuer Spatial Predicates genutzt werden:

            wald = db.features.filter("OBJEKTART = 'Wald'")
            db.intersects(wald, seen)
        """
        return self.conn.table("features")

    # -------------------------------------------------------------------------
    # Hilfsmethoden
    # -------------------------------------------------------------------------

    def _to_table_expr(self, condition: FeatureSet) -> str:
        """Konvertiert str oder DuckDBPyRelation zu einem SQL-Tabellenausdruck.

        Args:
            condition: SQL WHERE-String oder DuckDBPyRelation

        Returns:
            SQL-Fragment das als FROM-Quelle nutzbar ist
        """
        if isinstance(condition, str):
            return f"(SELECT * FROM features WHERE {condition})"

        # DuckDBPyRelation: als temporaere View registrieren
        view_name = f"_h3_tmp_{id(condition)}"
        self.conn.register(view_name, condition)
        return view_name

    def _cleanup_views(self, *conditions: FeatureSet) -> None:
        """Raeumt temporaere Views auf die fuer Relations erstellt wurden."""
        for cond in conditions:
            if isinstance(cond, duckdb.DuckDBPyRelation):
                view_name = f"_h3_tmp_{id(cond)}"
                try:
                    self.conn.unregister(view_name)
                except Exception:
                    pass

    def _get_resolution_range(self, table_expr: str) -> tuple[int, int]:
        """Ermittelt min/max Resolution fuer einen Tabellenausdruck."""
        result = self.conn.execute(f"""
            SELECT MIN(h3_resolution), MAX(h3_resolution)
            FROM {table_expr}
        """).fetchone()
        return result[0], result[1]

    # -------------------------------------------------------------------------
    # Boolean Predicates (EAGER - fuehren Query aus)
    # -------------------------------------------------------------------------

    def intersects(self, a: CellSet, b: CellSet) -> bool:
        """
        Prueft ob zwei CellSets sich ueberschneiden.

        EAGER: Fuehrt die Query sofort aus.

        Args:
            a: CellSet A (von union())
            b: CellSet B (von union())

        Returns:
            True wenn mindestens eine Cell von A mit einer Cell von B matched

        Example:
            cells_a = db.union("OBJEKTART = 'Wald'")
            cells_b = db.union("OBJEKTART = 'See'")
            db.intersects(cells_a, cells_b)
        """
        # Resolution aus CellSets (keine Query noetig)
        target_res = min(a.resolution, b.resolution)

        # Query mit CTEs zusammenbauen und ausfuehren
        result = self.conn.execute(f"""
            WITH a_cells AS ({a.sql}),
                 b_cells AS ({b.sql}),
                 a_parents AS (
                     SELECT DISTINCT h3_cell_to_parent(cell, {target_res}) as parent
                     FROM a_cells
                 ),
                 b_parents AS (
                     SELECT DISTINCT h3_cell_to_parent(cell, {target_res}) as parent
                     FROM b_cells
                 )
            SELECT 1
            FROM a_parents
            JOIN b_parents ON a_parents.parent = b_parents.parent
            LIMIT 1
        """).fetchone()

        return result is not None

    def within(self, a: CellSet, b: CellSet) -> bool:
        """
        Prueft ob alle Cells von CellSet A innerhalb von CellSet B liegen.

        EAGER: Fuehrt die Query sofort aus.

        Args:
            a: CellSet A, das "innere" (von union())
            b: CellSet B, das "aeussere" (von union())

        Returns:
            True wenn alle Cells von A in B enthalten sind

        Example:
            cells_a = db.union("feature_id = 123")
            cells_b = db.union("OBJEKTART = 'Kanton'")
            db.within(cells_a, cells_b)
        """
        # Resolution aus CellSets (keine Query noetig)
        target_res = min(a.resolution, b.resolution)

        # Query mit CTEs zusammenbauen und ausfuehren
        result = self.conn.execute(f"""
            WITH a_cells AS ({a.sql}),
                 b_cells AS ({b.sql}),
                 a_parents AS (
                     SELECT DISTINCT h3_cell_to_parent(cell, {target_res}) as cell
                     FROM a_cells
                 ),
                 b_parents AS (
                     SELECT DISTINCT h3_cell_to_parent(cell, {target_res}) as cell
                     FROM b_cells
                 )
            SELECT COUNT(*)
            FROM a_parents
            WHERE cell NOT IN (SELECT cell FROM b_parents)
        """).fetchone()

        return result[0] == 0

    def contains(self, a: CellSet, b: CellSet) -> bool:
        """
        Prueft ob CellSet A alle Cells von CellSet B enthaelt.

        Dies ist das Inverse von within(): contains(A, B) == within(B, A)

        EAGER: Fuehrt die Query sofort aus.

        Args:
            a: CellSet A, das "aeussere" (von union())
            b: CellSet B, das "innere" (von union())

        Returns:
            True wenn alle Cells von B in A enthalten sind

        Example:
            cells_a = db.union("OBJEKTART = 'Kanton'")
            cells_b = db.union("feature_id = 123")
            db.contains(cells_a, cells_b)
        """
        return self.within(b, a)

    # -------------------------------------------------------------------------
    # Set-Operationen (geben DuckDBPyRelation mit 'cell' Spalte zurueck)
    # -------------------------------------------------------------------------

    def intersection(self, a: CellSet, b: CellSet) -> CellSet:
        """
        Berechnet die Intersection von zwei CellSets.

        LAZY: Baut nur den Query-Plan, fuehrt nicht aus.
        Verwendet SQL-String-Komposition mit CTEs.

        Args:
            a: CellSet A (von union())
            b: CellSet B (von union())

        Returns:
            CellSet (Lazy Query-Plan)

        Example:
            cells_a = db.union("OBJEKTART = 'Wald'")
            cells_b = db.union("OBJEKTART = 'See'")
            result = db.intersection(cells_a, cells_b)  # Lazy
            db.area(result)  # Fuehrt Query aus
        """
        # Resolution aus den CellSets (bereits bekannt, keine Query noetig)
        res_a = a.resolution
        res_b = b.resolution

        # Join auf der groeberen Resolution
        target_res = min(res_a, res_b)
        # Ergebnis auf der feineren Resolution
        finest_res = max(res_a, res_b)

        # SQL-String zusammenbauen mit CTEs
        sql = f"""
            WITH a_cells AS ({a.sql}),
                 b_cells AS ({b.sql}),
                 coarse_parents AS (
                     SELECT DISTINCT h3_cell_to_parent(cell, {target_res}) as parent
                     FROM {"a_cells" if res_a <= res_b else "b_cells"}
                 ),
                 fine_matched AS (
                     SELECT cell
                     FROM {"b_cells" if res_a <= res_b else "a_cells"}
                     WHERE h3_cell_to_parent(cell, {target_res}) IN (SELECT parent FROM coarse_parents)
                 )
            SELECT DISTINCT cell, {finest_res}::TINYINT as resolution
            FROM fine_matched
        """

        return CellSet(sql=sql, resolution=finest_res, _engine=self)

    # -------------------------------------------------------------------------
    # Area / Messung
    # -------------------------------------------------------------------------

    def union(self, feature_set: FeatureSet) -> CellSet:
        """Normalisiert alle Cells eines Feature-Sets auf die feinste Resolution.

        LAZY: Baut nur den Query-Plan, fuehrt nicht aus.
        Expandiert groebere Cells via h3_cell_to_children() und
        dedupliziert, um eine korrekte Union ohne Doppelzaehlung zu erhalten.

        Args:
            feature_set: SQL WHERE-String oder DuckDBPyRelation

        Returns:
            CellSet (Lazy Query-Plan)

        Example:
            cells = db.union("OBJEKTART = 'Wald'")  # Lazy
            cells.run()  # Fuehrt Query aus
            db.area(cells)  # Fuehrt Query aus und berechnet Flaeche
        """
        expr = self._to_table_expr(feature_set)
        min_res, max_res = self._get_resolution_range(expr)

        if max_res is None:
            sql = "SELECT NULL::UBIGINT as cell, NULL::TINYINT as resolution WHERE false"
            return CellSet(sql=sql, resolution=0, _engine=self)

        feature_ids_subquery = f"SELECT feature_id FROM {expr}"

        # Alle Resolutions gleich: keine Normalisierung noetig
        if min_res == max_res:
            sql = f"""
                SELECT DISTINCT l.cell, {max_res}::TINYINT as resolution
                FROM h3_lookup l
                WHERE l.feature_id IN ({feature_ids_subquery})
            """
        else:
            # Normalisierung: Coarse Cells zu Children auf feinster Resolution expandieren
            sql = f"""
                SELECT DISTINCT cell, {max_res}::TINYINT as resolution FROM (
                    SELECT l.cell
                    FROM h3_lookup l
                    WHERE l.feature_id IN ({feature_ids_subquery})
                    AND l.cell_res = {max_res}

                    UNION ALL

                    SELECT UNNEST(h3_cell_to_children(l.cell, {max_res}))
                    FROM h3_lookup l
                    WHERE l.feature_id IN ({feature_ids_subquery})
                    AND l.cell_res < {max_res}
                )
            """

        return CellSet(sql=sql, resolution=max_res, _engine=self)

    def area(self, cell_set: CellSet, unit: str = "km^2") -> float:
        """Berechnet die Flaeche eines CellSets.

        EAGER: Fuehrt die Query sofort aus.

        Args:
            cell_set: CellSet (von union() oder intersection())
            unit: Flaecheneinheit ('km^2' oder 'm^2')

        Returns:
            Flaeche in der angegebenen Einheit

        Example:
            db.area(db.union("OBJEKTART = 'Wald'"))
            cells_a = db.union("OBJEKTART = 'Wald'")
            cells_b = db.union("OBJEKTART = 'See'")
            db.area(db.intersection(cells_a, cells_b))
        """
        # Query mit CTE zusammenbauen und ausfuehren
        result = self.conn.execute(f"""
            WITH cells AS ({cell_set.sql})
            SELECT COALESCE(SUM(h3_cell_area(cell, '{unit}')), 0)
            FROM cells
        """).fetchone()
        return result[0]

    def total_area(self, resolution: int = 8, unit: str = "km^2") -> float:
        """Berechnet die Gesamtflaeche des Datensatzes (vereinfacht).

        Normalisiert alle Cells auf die angegebene Resolution via
        h3_cell_to_parent(), nimmt DISTINCT, und summiert die Flaechen.

        Args:
            resolution: Ziel-Resolution fuer Normalisierung (default 8)
            unit: Flaecheneinheit ('km^2' oder 'm^2')

        Returns:
            Gesamtflaeche in der angegebenen Einheit
        """
        result = self.conn.execute(f"""
            WITH
            exact AS (
                SELECT cell FROM h3_lookup WHERE cell_res = {resolution}
            ),
            finer AS (
                SELECT h3_cell_to_parent(cell, {resolution}) as cell
                FROM h3_lookup WHERE cell_res > {resolution}
            ),
            coarser AS (
                SELECT UNNEST(h3_cell_to_children(cell, {resolution})) as cell
                FROM h3_lookup WHERE cell_res < {resolution}
            ),
            all_cells AS (
                SELECT cell FROM exact
                UNION ALL SELECT cell FROM finer
                UNION ALL SELECT cell FROM coarser
            )
            SELECT COALESCE(SUM(h3_cell_area(cell, '{unit}')), 0)
            FROM (SELECT DISTINCT cell FROM all_cells)
        """).fetchone()
        return result[0]

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def count_cells(self, where: str) -> int:
        """Zaehlt die Anzahl H3 Cells fuer eine WHERE-Bedingung."""
        result = self.conn.execute(f"""
            SELECT SUM(h3_cell_count)
            FROM features
            WHERE {where}
        """).fetchone()
        return result[0] or 0

    def count_features(self, where: str) -> int:
        """Zaehlt die Anzahl Features fuer eine WHERE-Bedingung."""
        result = self.conn.execute(f"""
            SELECT COUNT(*)
            FROM features
            WHERE {where}
        """).fetchone()
        return result[0]

    def get_resolutions(self, where: str) -> list[int]:
        """Gibt alle verwendeten Resolutions fuer eine WHERE-Bedingung zurueck."""
        result = self.conn.execute(f"""
            SELECT DISTINCT h3_resolution
            FROM features
            WHERE {where}
            ORDER BY h3_resolution
        """).fetchall()
        return [row[0] for row in result]

    # -------------------------------------------------------------------------
    # Predicate Builders
    # -------------------------------------------------------------------------

    def intersects_predicate(self, cell_set: CellSet) -> str:
        """Gibt ein SQL-Praedikat zurueck fuer Features die mit dem CellSet intersecten.

        LAZY: Gibt nur einen SQL-String zurueck, fuehrt keine Query aus.
        Das Praedikat kann mit der DuckDB Relational API kombiniert werden.

        Args:
            cell_set: CellSet (von union())

        Returns:
            SQL-String der als WHERE-Bedingung nutzbar ist

        Example:
            # Alle Strassen die mit dem Matterhorn intersecten
            matterhorn_cells = db.union("feature_id = 123")
            predicate = db.intersects_predicate(matterhorn_cells)
            strassen = db.features.filter(f"OBJEKTART = 'Strasse' AND {predicate}")
        """
        return f"""
            feature_id IN (
                WITH target_cells AS ({cell_set.sql})
                SELECT DISTINCT l.feature_id
                FROM h3_lookup l,
                     target_cells tc
                WHERE h3_cell_to_parent(l.cell, LEAST(l.cell_res, {cell_set.resolution}))
                    = h3_cell_to_parent(tc.cell, LEAST(l.cell_res, {cell_set.resolution}))
            )
        """

    # -------------------------------------------------------------------------
    # Lookup-basierte Intersection (schnell, nutzt h3_lookup Index)
    # -------------------------------------------------------------------------

    def _has_lookup_table(self) -> bool:
        """Prueft ob die h3_lookup Tabelle existiert."""
        result = self.conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = 'h3_lookup'
        """).fetchone()
        return result[0] > 0

    def find_intersecting_features(
        self,
        feature_id: int,
        objektart_list: list[str] | None = None,
        dataset: str | None = None,
        exclude_id: int | None = None,
        exclude_ids: list[int] | None = None,
        order_by_size: bool = False,
        max_results: int | None = None,
    ) -> duckdb.DuckDBPyRelation:
        """Findet alle Features die mit einem gegebenen Feature raeumlich intersecten.

        Nutzt die vorberechnete h3_lookup Tabelle fuer schnelle Lookups.
        Cross-Resolution wird ueber Source-Side h3_cell_to_parent() gehandelt.

        Args:
            feature_id: ID des Quell-Features
            objektart_list: Optionale Liste von OBJEKTART-Werten zum Filtern
            dataset: Optionaler Source-Filter (z.B. 'swissNAMES3D_PKT')
            exclude_id: Feature-ID die ausgeschlossen werden soll
            exclude_ids: Liste von Feature-IDs die ausgeschlossen werden sollen
            order_by_size: Wenn True, nach Feature-Groesse sortieren (absteigend)
            max_results: Maximale Anzahl Ergebnisse

        Returns:
            DuckDBPyRelation mit Spalten: feature_id, NAME, OBJEKTART, source, UUID
        """
        if not self._has_lookup_table():
            raise RuntimeError(
                "h3_lookup Tabelle nicht gefunden. "
                "Bitte zuerst erstellen via Build-Pipeline."
            )

        # Build WHERE clauses for the final filter
        where_parts = ["f.NAME IS NOT NULL"]
        if objektart_list:
            quoted = ", ".join(f"'{o}'" for o in objektart_list)
            where_parts.append(f"f.OBJEKTART IN ({quoted})")
        if dataset is not None:
            where_parts.append(f"f.source = '{dataset}'")
        if exclude_id is not None:
            where_parts.append(f"f.feature_id != {int(exclude_id)}")
        if exclude_ids:
            ids_str = ", ".join(str(int(i)) for i in exclude_ids)
            where_parts.append(f"f.feature_id NOT IN ({ids_str})")
        where_clause = " AND ".join(where_parts)

        order_clause = ""
        if order_by_size:
            order_clause = """
                ORDER BY f.h3_cell_count * h3_cell_area(
                    h3_cell_to_parent(0, f.h3_resolution), 'km^2'
                ) ASC
            """

        limit_clause = f"LIMIT {int(max_results)}" if max_results else ""

        sql = f"""
            WITH source AS (
                SELECT cell, cell_res
                FROM h3_lookup
                WHERE feature_id = {int(feature_id)}
            ),
            coarser_matches AS (
                SELECT DISTINCT l.feature_id
                FROM h3_lookup l
                JOIN (
                    SELECT DISTINCT
                        h3_cell_to_parent(s.cell, tr.cell_res) as parent_cell,
                        tr.cell_res as target_res
                    FROM source s,
                         (SELECT DISTINCT cell_res FROM h3_lookup) tr
                    WHERE tr.cell_res <= s.cell_res
                ) sp ON l.cell = sp.parent_cell AND l.cell_res = sp.target_res
            ),
            finer_matches AS (
                SELECT DISTINCT l.feature_id
                FROM h3_lookup l
                JOIN source s ON h3_cell_to_parent(l.cell, s.cell_res) = s.cell
                WHERE l.cell_res > s.cell_res
            ),
            all_matches AS (
                SELECT feature_id FROM coarser_matches
                UNION
                SELECT feature_id FROM finer_matches
            )
            SELECT DISTINCT f.feature_id, f.NAME, f.OBJEKTART, f.source, f.UUID
            FROM all_matches m
            JOIN features f ON m.feature_id = f.feature_id
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """

        return self.conn.sql(sql)

    def find_overlapping_features(
        self,
        feature_id: int,
        objektart: str | None = None,
        dataset: str | None = None,
        max_results: int = 5,
    ) -> duckdb.DuckDBPyRelation:
        """Findet Features die raeumlich ueberlappen, gerankt nach Overlap-Groesse.

        Filtert nach OBJEKTART und/oder Dataset-Name.

        Args:
            feature_id: ID des Quell-Features
            objektart: OBJEKTART-Wert zum Filtern (z.B. 'Gemeindegebiet')
            dataset: Dataset-Name zum Filtern (z.B. 'gemeinden') — deprecated
            max_results: Maximale Anzahl Ergebnisse

        Returns:
            DuckDBPyRelation mit Spalten: feature_id, NAME, OBJEKTART, overlap_cells
        """
        if not self._has_lookup_table():
            raise RuntimeError(
                "h3_lookup Tabelle nicht gefunden. "
                "Bitte zuerst erstellen via Build-Pipeline."
            )

        # Build filter for lookup table and features table
        lookup_filters = []
        feature_filters = ["f.NAME IS NOT NULL"]

        if objektart is not None:
            feature_filters.append(f"f.OBJEKTART = '{objektart}'")
        if dataset is not None:
            feature_filters.append(f"f.source = '{dataset}'")

        # For the target resolution CTE, we need a way to identify target features
        # Use feature_filters on a subquery to get target resolutions
        if objektart is not None:
            res_filter = f"""
                SELECT DISTINCT l2.cell_res FROM h3_lookup l2
                JOIN features f2 ON l2.feature_id = f2.feature_id
                WHERE f2.OBJEKTART = '{objektart}'
            """
        elif dataset is not None:
            res_filter = f"""
                SELECT DISTINCT l2.cell_res FROM h3_lookup l2
                JOIN features f2 ON l2.feature_id = f2.feature_id
                WHERE f2.source = '{dataset}'
            """
        else:
            res_filter = "SELECT DISTINCT cell_res FROM h3_lookup"

        # Filter lookup joins to only consider target features
        target_subquery = None
        if objektart is not None:
            target_subquery = f"SELECT feature_id FROM features WHERE OBJEKTART = '{objektart}'"
        elif dataset is not None:
            target_subquery = f"SELECT feature_id FROM features WHERE source = '{dataset}'"

        lookup_where_coarse = f"l.feature_id IN ({target_subquery})" if target_subquery else ""
        lookup_where_fine = lookup_where_coarse

        coarse_extra = f"\n                AND {lookup_where_coarse}" if lookup_where_coarse else ""
        fine_extra = f"\n                AND {lookup_where_fine}" if lookup_where_fine else ""

        sql = f"""
            WITH source AS (
                SELECT cell, cell_res
                FROM h3_lookup
                WHERE feature_id = {int(feature_id)}
            ),
            coarser_matches AS (
                SELECT l.feature_id, COUNT(*) as overlap_cells
                FROM h3_lookup l
                JOIN (
                    SELECT DISTINCT
                        h3_cell_to_parent(s.cell, tr.cell_res) as parent_cell,
                        tr.cell_res as target_res
                    FROM source s,
                         ({res_filter}) tr
                    WHERE tr.cell_res <= s.cell_res
                ) sp ON l.cell = sp.parent_cell AND l.cell_res = sp.target_res{coarse_extra}
                GROUP BY l.feature_id
            ),
            finer_matches AS (
                SELECT l.feature_id, COUNT(*) as overlap_cells
                FROM h3_lookup l
                JOIN source s ON h3_cell_to_parent(l.cell, s.cell_res) = s.cell
                WHERE l.cell_res > s.cell_res{fine_extra}
                GROUP BY l.feature_id
            ),
            all_matches AS (
                SELECT feature_id, SUM(overlap_cells) as overlap_cells
                FROM (
                    SELECT * FROM coarser_matches
                    UNION ALL
                    SELECT * FROM finer_matches
                )
                GROUP BY feature_id
            )
            SELECT f.feature_id, f.NAME, f.OBJEKTART, m.overlap_cells
            FROM all_matches m
            JOIN features f ON m.feature_id = f.feature_id
            WHERE {' AND '.join(feature_filters)}
            ORDER BY m.overlap_cells DESC
            LIMIT {int(max_results)}
        """

        return self.conn.sql(sql)
