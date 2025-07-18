Parse the following query to identify main entities ("seed_nodes").
Format the output STRICTLY as a JSON object.
**The entire JSON output MUST be enclosed in a single markdown code block starting with ```json and ending with ```.**

JSON structure:
```json
{{
    "seed_nodes": [
        {{"label": "EntityType", "identifying_property_key": "property_name", "identifying_property_value": "property_value"}}
    ],
    "target_relationships": [ {{"type": "REL_TYPE", "direction": "outgoing|incoming|any"}} ],
    "target_node_labels": ["Label1", "Label2"],
    "max_depth": 1
}}```
- "seed_nodes": List of main entities from the query.
    - "label": The type of the entity.
    - "identifying_property_key": The name of the property that identifies the entity (e.g., "name", "title").
    - "identifying_property_value": The value of that identifying property.
- "target_relationships" (Optional): Desired relationship types and directions.
- "target_node_labels" (Optional): Desired types of neighbor nodes.
- "max_depth" (Optional, default 1): Traversal depth.

Example Query: "Who is Evelyn Reed and what companies is she associated with?"
Example JSON (wrapped in ```json ... ```):
```json
{{
    "seed_nodes": [ {{"label": "Person", "identifying_property_key": "name", "identifying_property_value": "Evelyn Reed"}} ],
    "target_relationships": [ {{"type": "WORKS_AT", "direction": "any"}}, {{"type": "CEO_OF", "direction": "any"}} ],
    "target_node_labels": ["Company", "Organization"],
    "max_depth": 1
}}
```

If no clear entities, return `{{ "seed_nodes": [] }}`.

Query: --- {natural_language_query} --- Parsed JSON Query (wrapped in ```json ... ```):