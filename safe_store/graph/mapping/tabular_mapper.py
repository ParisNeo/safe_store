import csv
import io
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import pipmaster as pm

from ascii_colors import ASCIIColors
from ...core.exceptions import ConfigurationError, SafeStoreError
from ..ontology.tbox import TBoxManager


class TabularMapper:
    """
    Declarative Mapping Engine for transforming structured data (CSV, XLSX, SQLite)
    into grounded ABox RDF graphs within SafeStore.
    """

    def __init__(self, store: Any, tbox: Optional[TBoxManager] = None):
        self.store = store
        self.tbox = tbox

    def _resolve_template(self, template: str, row_dict: Dict[str, Any]) -> str:
        """Fills template variables like http://example.org/person/{id} with row values."""
        formatted = template
        for key, val in row_dict.items():
            token = f"{{{key}}}"
            if token in formatted:
                formatted = formatted.replace(token, str(val).strip())
        return formatted

    def map_csv(self, file_path: Union[str, Path], mapping_rules: Dict[str, Any]) -> Dict[str, int]:
        """
        Maps a CSV file into graph nodes and relationships according to declarative mapping rules.
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"CSV file not found: {path_obj}")

        with open(path_obj, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        return self._map_records(rows, mapping_rules, source_identifier=str(path_obj.resolve()))

    def map_excel(
        self,
        file_path: Union[str, Path],
        mapping_rules: Dict[str, Any],
        sheet_name: Optional[Union[str, int]] = 0
    ) -> Dict[str, int]:
        """
        Maps an Excel (.xlsx, .xls) spreadsheet into graph nodes and relationships.
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Excel file not found: {path_obj}")

        try:
            pm.ensure_packages(["pandas", "openpyxl"])
            import pandas as pd
        except ImportError as e:
            raise ConfigurationError("Mapping Excel files requires 'pandas' and 'openpyxl'.") from e

        df = pd.read_excel(path_obj, sheet_name=sheet_name)
        rows = df.to_dict(orient="records")

        # Clean NaN values
        cleaned_rows = []
        for r in rows:
            cleaned_row = {k: ("" if pd.isna(v) else v) for k, v in r.items()}
            cleaned_rows.append(cleaned_row)

        return self._map_records(cleaned_rows, mapping_rules, source_identifier=f"{path_obj.resolve()}:{sheet_name}")

    def map_sqlite_table(
        self,
        db_path: Union[str, Path],
        table_name: str,
        mapping_rules: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Maps rows from an external SQLite table into graph nodes and relationships.
        """
        src_conn = sqlite3.connect(str(db_path))
        src_conn.row_factory = sqlite3.Row
        cursor = src_conn.cursor()
        
        try:
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = [dict(row) for row in cursor.fetchall()]
        finally:
            src_conn.close()

        return self._map_records(rows, mapping_rules, source_identifier=f"{db_path}:{table_name}")

    def _map_records(
        self,
        rows: List[Dict[str, Any]],
        mapping_rules: Dict[str, Any],
        source_identifier: str
    ) -> Dict[str, int]:
        """
        Executes declarative mapping rules over row records.
        """
        entity_mappings = mapping_rules.get("entity_mappings", [])
        rel_mappings = mapping_rules.get("relationship_mappings", [])

        nodes_created = 0
        triples_generated = 0

        uri_to_node_id: Dict[str, int] = {}

        self.store._ensure_connection()
        conn = self.store.conn
        assert conn is not None

        try:
            conn.execute("BEGIN")

            # 1. Process Entity Mappings (Nodes & Datatype Properties)
            for row in rows:
                for ent_map in entity_mappings:
                    class_uri = ent_map.get("class", "http://example.org/ontology/Thing")
                    class_label = class_uri.split("/")[-1].split("#")[-1]
                    subj_template = ent_map.get("subject_template", "")
                    
                    subj_uri = self._resolve_template(subj_template, row)
                    if not subj_uri or "{" in subj_uri:
                        continue

                    # Extract mapped properties
                    props = {"identifying_value": subj_uri, "uri": subj_uri}
                    for col_name, prop_uri in ent_map.get("properties", {}).items():
                        if col_name in row and row[col_name] is not None and str(row[col_name]).strip() != "":
                            props[prop_uri] = row[col_name]
                            triples_generated += 1

                    triples_generated += 1

                    # Persist node into database
                    sig = f"{class_label}:{subj_uri}"
                    props_json = json.dumps(props)

                    cursor = conn.cursor()
                    cursor.execute("SELECT node_id FROM graph_nodes WHERE unique_signature = ?", (sig,))
                    existing = cursor.fetchone()

                    if existing:
                        node_id = existing[0]
                        conn.execute("UPDATE graph_nodes SET node_properties = ? WHERE node_id = ?", (props_json, node_id))
                    else:
                        cursor.execute(
                            "INSERT INTO graph_nodes (node_label, node_properties, unique_signature) VALUES (?, ?, ?)",
                            (class_label, props_json, sig)
                        )
                        node_id = cursor.lastrowid
                        nodes_created += 1

                    uri_to_node_id[subj_uri] = node_id

            # 2. Process Relationship Mappings (Object Properties)
            for row in rows:
                for rel_map in rel_mappings:
                    pred_uri = rel_map.get("predicate", "http://example.org/ontology/relatedTo")
                    pred_label = pred_uri.split("/")[-1].split("#")[-1]
                    
                    src_template = rel_map.get("source_template", "")
                    tgt_template = rel_map.get("target_template", "")

                    src_uri = self._resolve_template(src_template, row)
                    tgt_uri = self._resolve_template(tgt_template, row)

                    src_id = uri_to_node_id.get(src_uri)
                    tgt_id = uri_to_node_id.get(tgt_uri)

                    if src_id and tgt_id:
                        conn.execute(
                            "INSERT INTO graph_relationships (source_node_id, target_node_id, relationship_type, relationship_properties) "
                            "VALUES (?, ?, ?, ?)",
                            (src_id, tgt_id, pred_label, json.dumps({"uri": pred_uri}))
                        )
                        triples_generated += 1

            conn.commit()
            ASCIIColors.success(f"Mapped {len(rows)} records from '{source_identifier}': {nodes_created} nodes, {triples_generated} triples.")

            return {
                "records_processed": len(rows),
                "entities_created": len(uri_to_node_id),
                "triples_generated": triples_generated
            }

        except Exception as e:
            if conn.in_transaction:
                conn.rollback()
            raise SafeStoreError(f"Error during tabular mapping of '{source_identifier}': {e}") from e