Extract entities (nodes) and their relationships from the following text.
Format the output strictly as a JSON object.
**The entire JSON output MUST be enclosed in a single markdown code block starting with ```json and ending with ```.**

---
**Extraction Guidance:**
{user_guidance}
---

JSON Structure Example:
```json
{{
    "nodes": [
        {{"label": "Person", "properties": {{"name": "John Doe", "title": "Engineer"}}}},
        {{"label": "Company", "properties": {{"name": "Acme Corp", "industry": "Tech"}}}}
    ],
    "relationships": [
        {{"source_node_label": "Person", "source_node_identifying_value": "John Doe",
            "target_node_label": "Company", "target_node_identifying_value": "Acme Corp",
            "type": "WORKS_AT", "properties": {{"role": "Engineer"}}}}
    ]
}}
```

For each node:
- "label": A general type (e.g., "Person", "Company", "Product", "Location", "Organization", "ResearchPaper", "University", "Journal").
- "properties": Dictionary of relevant attributes. Pay close attention to the **Extraction Guidance**. Ensure properties like "name", "title", or other unique identifiers are included if available.

For each relationship:
- "source_node_label": Label of the source node.
- "source_node_identifying_value": The value of a primary identifying property from the source node (e.g., if source node is `{{ "label": "Person", "properties": {{"name": "John Doe"}}}}`, this value would be "John Doe". Use the most prominent identifier like name or title).
- "target_node_label": Label of the target node.
- "target_node_identifying_value": Similar to "source_node_identifying_value" for the target node.
- "type": Relationship type in UPPER_SNAKE_CASE (e.g., "WORKS_AT", "CEO_OF", "PUBLISHED_IN").
- "properties": Optional dictionary for relationship attributes. Make sure the entries are in form "property":"detail"

Text to process:
---
{chunk_text}
---

Extracted JSON (wrapped in ```json ... ```):