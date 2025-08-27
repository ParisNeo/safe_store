# [UPDATE] prompts/graph_extraction_prompt.md
Extract entities (nodes) and their relationships from the following text.
Format the output strictly as a JSON object.
**The entire JSON output MUST be enclosed in a single markdown code block starting with ```json and ending with ```.**

---
**Ontology Schema (If not empty, You MUST respect this schema for all labels and relationship types):**
{ontology_schema}
---

**User Guidance (Additional instructions):**
{user_guidance}
---

**Text to process:**
{chunk_text}
---

JSON Structure Example (Use this structure, but with data from the text):
```json
{{
    "nodes": [
        {{"label": "LabelA", "properties": {{"identifying_property": "Value 1", "another_property": "Some other value"}}}},
        {{"label": "LabelB", "properties": {{"identifying_property": "Value 2", "industry": "An industry value"}}}}
    ],
    "relationships": [
        {{
            "source_node_label": "LabelA", 
            "source_node_identifying_value": "Value 1",
            "target_node_label": "LabelB", 
            "target_node_identifying_value": "Value 2",
            "type": "RELATIONSHIP_TYPE", 
            "properties": {{"role": "A role or description"}}
        }}
    ]
}}
```