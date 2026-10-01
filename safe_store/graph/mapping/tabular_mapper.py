import csv
import io
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import json
import pipmaster as pm

from ascii_colors import ASCIIColors
from ...core.exceptions import ConfigurationError, SafeStoreError
from ..ontology.tbox import TBoxManager
from ...utils.json_parsing import robust_json_parser


class TabularMapper:
    """
    Declarative Mapping Engine for transforming structured data (CSV, XLSX, SQLite)
    into grounded ABox RDF graphs within SafeStore.

    Supports:
    1. Direct mapping via rules dictionary (JSON).
    2. Mapping via mapping CSV table specification.
    3. AI-assisted mapping rule synthesis via LLM with user verification.
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

    @staticmethod
    def parse_mapping_csv(mapping_csv_content_or_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parses a 4-column mapping CSV specification into declarative mapping rules.

        Format:
        type,source_column,target_class_or_predicate,template_or_property
        entity,id,Person,http://example.org/person/{id}
        property,name,Person,hasName
        entity,company_id,Company,http://example.org/company/{company_id}
        relationship,,worksFor,http://example.org/person/{id} -> http://example.org/company/{company_id}
        """
        if isinstance(mapping_csv_content_or_path, Path) or (isinstance(mapping_csv_content_or_path, str) and Path(mapping_csv_content_or_path).is_file()):
            text = Path(mapping_csv_content_or_path).read_text(encoding='utf-8')
        else:
            text = str(mapping_csv_content_or_path)

        reader = csv.DictReader(io.StringIO(text.strip()))
        entity_map: Dict[str, Dict[str, Any]] = {}
        rel_mappings = []

        for row in reader:
            m_type = str(row.get("type", "")).strip().lower()
            col = str(row.get("source_column", "")).strip()
            target = str(row.get("target_class_or_predicate", "")).strip()
            template_or_prop = str(row.get("template_or_property", "")).strip()

            if m_type == "entity":
                cls_uri = target if target.startswith("http") else f"http://example.org/ontology/{target}"
                tmpl = template_or_prop or f"http://example.org/{target.lower()}/{{{col}}}"
                entity_map[target] = {
                    "class": cls_uri,
                    "subject_template": tmpl,
                    "properties": {}
                }

            elif m_type == "property":
                if target in entity_map:
                    prop_uri = template_or_prop if template_or_prop.startswith("http") else f"http://example.org/ontology/{template_or_prop}"
                    entity_map[target]["properties"][col] = prop_uri

            elif m_type == "relationship":
                pred_uri = target if target.startswith("http") else f"http://example.org/ontology/{target}"
                if "->" in template_or_prop:
                    src_tmpl, tgt_tmpl = [s.strip() for s in template_or_prop.split("->", 1)]
                else:
                    parts = [p.strip() for p in template_or_prop.split(",")]
                    src_tmpl = parts[0] if len(parts) > 0 else ""
                    tgt_tmpl = parts[1] if len(parts) > 1 else ""

                rel_mappings.append({
                    "predicate": pred_uri,
                    "source_template": src_tmpl,
                    "target_template": tgt_tmpl
                })

        return {
            "entity_mappings": list(entity_map.values()),
            "relationship_mappings": rel_mappings
        }

    def propose_mapping_with_llm(
        self,
        csv_data_or_path: Union[str, Path],
        user_guidance: Optional[str] = None,
        sample_rows_count: int = 4
    ) -> Dict[str, Any]:
        """
        Inspects sample CSV rows, analyzes the active TBox ontology schema, and prompts
        the LLM generator to propose declarative mapping rules.
        """
        if isinstance(csv_data_or_path, Path) or (isinstance(csv_data_or_path, str) and Path(csv_data_or_path).is_file()):
            text = Path(csv_data_or_path).read_text(encoding='utf-8')
        else:
            text = str(csv_data_or_path)

        reader = csv.reader(io.StringIO(text.strip()))
        all_rows = list(reader)
        if not all_rows:
            return {"entity_mappings": [], "relationship_mappings": []}

        headers = all_rows[0]
        sample_rows = all_rows[1:sample_rows_count + 1]

        # Format sample for prompt
        sample_preview_lines = [", ".join(headers)]
        for r in sample_rows:
            sample_preview_lines.append(", ".join(r))
        csv_sample_str = "\n".join(sample_preview_lines)

        tbox_schema = self.tbox.to_prompt_schema() if self.tbox else "Standard entity categories: Person, Company, Organization, Product, Tool, Location."

        system_prompt = (
            "You are an expert knowledge graph architect and ontology engineer. "
            "Analyze the CSV columns and sample data against the provided domain ontology. "
            "Formulate declarative mapping rules in strictly well-formed JSON conforming to the TabularMapper schema."
        )

        guidance_text = f"Additional Guidance: {user_guidance}\n" if user_guidance else ""

        user_prompt = f"""CSV Headers and Sample Records:
```csv
{csv_sample_str}
```

Domain Ontology Schema:
{tbox_schema}
{guidance_text}
Task:
1. Identify entity classes to instantiate from the CSV columns. Assign each an identifying subject_template URI (e.g. "http://example.org/person/{{id}}").
2. Map attribute columns to datatype properties on those entities.
3. Map foreign key or associative column pairs to directed relationship_mappings (source_template, target_template, predicate).

Respond STRICTLY with a single JSON object in this format:
```json
{{
    "entity_mappings": [
        {{
            "class": "http://example.org/ontology/ClassName",
            "subject_template": "http://example.org/entity/{{column_name}}",
            "properties": {{
                "column_name": "http://example.org/ontology/propertyName"
            }}
        }}
    ],
    "relationship_mappings": [
        {{
            "predicate": "http://example.org/ontology/predicateName",
            "source_template": "http://example.org/source/{{src_id}}",
            "target_template": "http://example.org/target/{{tgt_id}}"
        }}
    ]
}}
```"""

        llm = getattr(self.store, "llm_generator", None)
        if not llm:
            # Deterministic heuristic mapping proposal when no LLM is configured
            return self._heuristic_mapping_proposal(headers, sample_rows)

        try:
            raw = llm(prompt=user_prompt, system_prompt=system_prompt, json_mode=True)
            parsed = robust_json_parser(raw)
            if isinstance(parsed, dict) and ("entity_mappings" in parsed or "relationship_mappings" in parsed):
                return {
                    "entity_mappings": parsed.get("entity_mappings", []),
                    "relationship_mappings": parsed.get("relationship_mappings", [])
                }
        except Exception as e:
            ASCIIColors.warning(f"LLM mapping proposal failed: {e}. Falling back to heuristic proposal.")

        return self._heuristic_mapping_proposal(headers, sample_rows)

    def _heuristic_mapping_proposal(self, headers: List[str], sample_rows: List[List[str]]) -> Dict[str, Any]:
        """Generates baseline mapping rules by inspecting common naming patterns (id, name, type, ref)."""
        id_col = next((h for h in headers if h.lower() in ("id", "sku", "uuid", "key", "code")), headers[0] if headers else "id")
        class_name = "Item"
        for h in headers:
            if "person" in h.lower() or "user" in h.lower() or "author" in h.lower():
                class_name = "Person"
                break
            if "company" in h.lower() or "org" in h.lower() or "supplier" in h.lower():
                class_name = "Company"
                break

        props = {}
        for h in headers:
            if h != id_col:
                props[h] = f"http://example.org/ontology/{h}"

        return {
            "entity_mappings": [
                {
                    "class": f"http://example.org/ontology/{class_name}",
                    "subject_template": f"http://example.org/{class_name.lower()}/{{{id_col}}}",
                    "properties": props
                }
            ],
            "relationship_mappings": []
        }

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

    def map_csv_with_mapping_file(
        self,
        data_csv_path: Union[str, Path],
        mapping_file_path: Union[str, Path]
    ) -> Dict[str, int]:
        """
        Maps a CSV data file using a mapping file (.csv or .json specification).
        """
        m_path = Path(mapping_file_path)
        if not m_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {m_path}")

        if m_path.suffix.lower() == ".json":
            mapping_rules = json.loads(m_path.read_text(encoding='utf-8'))
        else:
            mapping_rules = self.parse_mapping_csv(m_path)

        return self.map_csv(data_csv_path, mapping_rules=mapping_rules)

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
        """Executes declarative mapping rules over row records."""
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
                            prop_key = prop_uri.split('/')[-1].split('#')[-1]
                            props[prop_key] = row[col_name]
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