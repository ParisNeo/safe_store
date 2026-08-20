import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path
from ascii_colors import ASCIIColors

from ...core import db
from ...core.exceptions import GraphProcessingError, ConfigurationError
from ...utils.json_parsing import robust_json_parser
from ..ontology.tbox import TBoxManager


class OntologyTextExtractor:
    """
    Extracts ABox knowledge graph instances from unstructured text chunks,
    strictly constrained by a TBox ontology schema, and registers chunk provenance.
    """

    def __init__(
        self,
        tbox: TBoxManager,
        llm_executor: Optional[Callable[[str], str]] = None
    ):
        self.tbox = tbox
        self.llm_executor = llm_executor or (lambda p: '{"nodes": [], "relationships": []}')

    def generate_extraction_prompt(self, chunk_text: str, user_guidance: Optional[str] = None) -> str:
        """Constructs an ontology-grounded prompt for the LLM."""
        tbox_schema = self.tbox.to_prompt_schema()
        guidance = user_guidance or "Extract all entities, attributes, and relationships conforming strictly to the ontology."

        prompt = f"""**CRITICAL INSTRUCTION: You are a semantic knowledge graph extraction expert.
Your task is to extract entities (nodes), properties, and relationships from the provided text, strictly adhering to the TBox ontology schema below.**

{tbox_schema}

**Extraction Rules:**
1. ONLY extract nodes whose `label` matches a class defined in the ontology schema.
2. For each node, include an `identifying_value` (e.g. unique name, title, or code) and only extract properties matching defined DatatypeProperties.
3. ONLY extract relationships matching defined ObjectProperties, respecting the Source and Target class constraints.
4. Format the output STRICTLY as a single JSON object inside a markdown code block.

**User Guidance:**
{guidance}

---
**Text to process:**
{chunk_text}
---

**Expected JSON Output Structure:**
```json
{{
    "nodes": [
        {{
            "label": "ClassName",
            "properties": {{
                "identifying_value": "Entity Identifier (Mandatory)",
                "propertyName": "PropertyValue"
            }}
        }}
    ],
    "relationships": [
        {{
            "source_node_label": "SourceClassName",
            "source_node_identifying_value": "SourceEntityIdentifier",
            "target_node_label": "TargetClassName",
            "target_node_identifying_value": "TargetEntityIdentifier",
            "type": "RelationshipType",
            "properties": {{}}
        }}
    ]
}}
```
"""
        return prompt

    def extract_from_chunk(
        self,
        conn: Any,
        chunk_id: int,
        chunk_text: str,
        user_guidance: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes ontology-grounded extraction on a chunk, persists nodes and relationships,
        and links them to the chunk provenance ID.
        """
        prompt = self.generate_extraction_prompt(chunk_text, user_guidance)
        raw_response = self.llm_executor(prompt)
        
        try:
            parsed = robust_json_parser(raw_response)
        except Exception as e:
            ASCIIColors.warning(f"Failed to parse LLM extraction for chunk {chunk_id}: {e}")
            return {"nodes_created": 0, "relationships_created": 0}

        nodes_data = parsed.get("nodes", [])
        rels_data = parsed.get("relationships", [])

        node_map: Dict[Tuple[str, str], int] = {}
        nodes_created = 0
        rels_created = 0

        # 1. Insert/Fuse Nodes
        for n in nodes_data:
            if not isinstance(n, dict) or "label" not in n or "properties" not in n:
                continue
            label = str(n["label"])
            props = n["properties"]
            id_val = str(props.get("identifying_value") or props.get("name") or "")
            if not id_val:
                continue

            sig = f"{label}:{id_val.strip().lower()}"
            node_id = db.add_or_update_graph_node(conn, label, props, sig)
            db.link_node_to_chunk(conn, node_id, chunk_id)
            node_map[(label, id_val)] = node_id
            nodes_created += 1

        # 2. Insert Relationships
        for r in rels_data:
            if not isinstance(r, dict):
                continue
            src_label = str(r.get("source_node_label", ""))
            src_val = str(r.get("source_node_identifying_value", ""))
            tgt_label = str(r.get("target_node_label", ""))
            tgt_val = str(r.get("target_node_identifying_value", ""))
            rel_type = str(r.get("type", ""))

            src_id = node_map.get((src_label, src_val)) or db.get_graph_node_by_signature(conn, f"{src_label}:{src_val.strip().lower()}")
            tgt_id = node_map.get((tgt_label, tgt_val)) or db.get_graph_node_by_signature(conn, f"{tgt_label}:{tgt_val.strip().lower()}")

            if src_id and tgt_id and rel_type:
                props_json = json.dumps(r.get("properties", {}))
                db.add_graph_relationship(conn, src_id, tgt_id, rel_type, props_json)
                rels_created += 1

        return {
            "nodes_created": nodes_created,
            "relationships_created": rels_created
        }