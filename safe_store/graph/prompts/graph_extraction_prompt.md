**CRITICAL INSTRUCTION: You are an expert data and knowledge graph extractor. Your task is to extract meaningful entities (nodes) and their relationships from the provided text to construct a rich knowledge graph.**

- **Node Extraction**: Extract key entities, concepts, persons, organizations, technologies, tools, software components, events, and objects mentioned in the text. Assign each node an appropriate `label` (e.g., `Concept`, `Person`, `Organization`, `Technology`, `Module`, `Tool`, `Method`, `Event`, `Document`).
- **Mandatory Identifying Value**: Every node's `properties` dictionary **MUST** contain an `identifying_value` (e.g., the canonical name or specific term) to uniquely identify and link the entity.
- **Properties**: Extract any relevant attributes, descriptions, or parameters present in the text into `properties`.
- **Relationship Extraction**: Identify directed connections between extracted entities. Use descriptive, uppercase relationship types (e.g., `USES`, `DEPENDS_ON`, `CREATED_BY`, `PART_OF`, `DEFINES`, `RELATES_TO`, `INTEGRATES_WITH`, `MENTIONS`).
- **Format**: Output strictly a single JSON object inside a markdown code block.

**User Guidance (Follow these additional instructions if provided):**
{user_guidance}

---

**Text to process:**
{chunk_text}

---

**JSON Output Structure:**
```json
{
    "nodes": [
        {
            "label": "Technology",
            "properties": {
                "identifying_value": "SafeStore",
                "description": "Local vector and graph database library",
                "backend": "SQLite"
            }
        },
        {
            "label": "Concept",
            "properties": {
                "identifying_value": "Knowledge Graph",
                "description": "Interconnected entity and relation graph"
            }
        }
    ],
    "relationships": [
        {
            "source_node_label": "Technology",
            "source_node_identifying_value": "SafeStore",
            "target_node_label": "Concept",
            "target_node_identifying_value": "Knowledge Graph",
            "type": "IMPLEMENTS",
            "properties": {
                "details": "Provides native graph storage and SPARQL 1.1 query engine"
            }
        }
    ]
}
```