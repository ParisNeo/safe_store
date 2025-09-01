Extract entities (nodes) and their relationships from the following text based on your understanding of the context.
Format the output strictly as a JSON object.
**The entire JSON output MUST be enclosed in a single markdown code block starting with ```json and ending with ```.**

---
**Extraction Rules:**
1.  **Nodes:** For each entity, create a node object.
2.  **Mandatory Properties:** Every node's `properties` object **MUST** contain `identifying_value` and `type`.
    - `identifying_value`: A unique name or identifier for the entity (e.g., "John Doe", "Acme Corporation"). This is critical for creating relationships. Entities that are the same **MUST** have the exact same `identifying_value`.
    - `type`: A category for the entity (e.g., "Person", "Company", "Location").
3.  **Inferred Properties:** Add any other relevant properties you can infer from the text.
4.  **Consistency:** Nodes that share the same `type` should have a similar set of properties where applicable.
5.  **Relationships:** Define connections between nodes using their `identifying_value`.

---
**User Guidance (Additional instructions):**
{user_guidance}
---

**Text to process:**
{chunk_text}
---

**JSON Structure Example (Use this as a template):**
```json
{{
    "nodes": [
        {{
            "label": "LabelA",
            "properties": {{
                "identifying_value": "Unique Identifier for A",
                "type": "TypeA",
                "inferred_property_1": "Value 1",
                "another_property": "Some other value"
            }}
        }},
        {{
            "label": "LabelB",
            "properties": {{
                "identifying_value": "Unique Identifier for B",
                "type": "TypeB",
                "some_other_prop": "Value 2"
            }}
        }}
    ],
    "relationships": [
        {{
            "source_node_label": "LabelA",
            "source_node_identifying_value": "Unique Identifier for A",
            "target_node_label": "LabelB",
            "target_node_identifying_value": "Unique Identifier for B",
            "type": "RELATIONSHIP_TYPE",
            "properties": {{
                "description": "Describes the relationship"
            }}
        }}
    ]
}}

# Warning!!
For nodes: label, properties/identifying_value and properties/type are mandatory. never generate a node that doesn't have these entries
For relationships: source_node_label and target_node_label properties/descrition are also mandatory