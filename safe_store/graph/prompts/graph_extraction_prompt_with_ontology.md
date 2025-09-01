**CRITICAL INSTRUCTION: You are a data extraction expert. Your task is to extract entities (nodes) and relationships from the provided text, strictly adhering to the ontology schema below.**

- **ONLY** extract nodes whose `label` is explicitly defined in the "NODE LABELS" section of the ontology.
- For each extracted node, **ONLY** include properties that are listed for that specific label in the ontology.
- **ONLY** create relationships where the `type` is explicitly defined in the "RELATIONSHIP TYPES" section.
- If the ontology specifies `Source` and `Target` constraints for a relationship, you **MUST** respect them.
- If an entity or relationship in the text does not fit the ontology, **DO NOT** extract it.
- Format the output as a single JSON object inside a markdown code block.

---
**Ontology Schema (You MUST adhere to this strictly):**
{ontology_schema}
---

**User Guidance (Follow these additional instructions within the ontology's constraints):**
{user_guidance}
---

**Text to process:**
{chunk_text}
---

**JSON Output Structure (Populate this structure according to the rules):**
```json
{{
    "nodes": [
        {{
            "label": "LabelFromOntology",
            "properties": {{
                "identifying_value": "A unique value for this entity",
                "property_from_ontology": "Value from text",
                "...": "..."
            }}
        }}
    ],
    "relationships": [
        {{
            "source_node_label": "SourceLabelFromOntology",
            "source_node_identifying_value": "Identifier of the source node",
            "target_node_label": "TargetLabelFromOntology",
            "target_node_identifying_value": "Identifier of the target node",
            "type": "RelationshipTypeFromOntology",
            "properties": {{
                "role": "A role or description if applicable"
            }}
        }}
    ]
}}