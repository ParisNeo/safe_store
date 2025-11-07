**CRITICAL INSTRUCTION: You are a data extraction expert. Your task is to extract entities (nodes) and relationships from the provided text, strictly adhering to the ontology schema below.**

- **ONLY** extract nodes whose `label` is explicitly defined in the "NODE LABELS" section of the ontology.
- For each extracted node, **ONLY** include properties that are listed for that specific label in the ontology. Be exhaustive and extract every property defined in the ontology that is present in the text.
- **ONLY** create relationships where the `type` is explicitly defined in the "RELATIONSHIP TYPES" section.
- You **MUST** respect the `Source` and `Target` constraints for relationships if they are specified.
- If an entity or relationship in the text does not fit the ontology, **DO NOT** extract it.
- Every node's `properties` object **MUST** contain an `identifying_value`. This is a unique name or identifier for the entity (e.g., "John Doe", "Acme Corporation") and is used to link relationships.
- Format the output as a single JSON object inside a markdown code block.

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
                "identifying_value": "A unique value for this entity (MANDATORY)",
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
```