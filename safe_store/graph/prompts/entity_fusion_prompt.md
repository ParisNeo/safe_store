Given a "New Entity" extracted from a document and a list of "Candidate Existing Entities" from a knowledge graph, determine if the New Entity should be merged with one of the existing entities.

**New Entity:**
- Label: {new_entity_label}
- Properties: {new_entity_properties}

**Candidate Existing Entities:**
{candidate_entities_str}

**Task:**
Analyze the entities and decide if the "New Entity" represents the same real-world concept as one of the candidates.

**Output Format:**
Respond with a JSON object in a markdown code block.
- If a merge is appropriate, identify the `node_id` of the best candidate to merge with.
- If no candidate is a suitable match, decide to create a new entity.

**JSON Response Structure:**
```json
{{
  "decision": "MERGE" | "CREATE_NEW",
  "reason": "Your detailed reasoning for the decision.",
  "merge_target_id": <node_id_of_candidate_to_merge_with_if_decision_is_MERGE> | null
}}
```

**Example 1 (Merge):**
New Entity: {{ "label": "Person", "properties": {{ "name": "Dr. Smith", "affiliation": "MIT" }} }}
Candidates: [ {{ "node_id": 101, "properties": {{ "name": "Dr. J. Smith", "title": "Professor" }} }} ]
Response:
```json
{{
  "decision": "MERGE",
  "reason": "The new entity 'Dr. Smith' from MIT very likely refers to the existing entity 'Dr. J. Smith', who is a professor. The names are a close match.",
  "merge_target_id": 101
}}
```

**Example 2 (Create New):**
New Entity: {{ "label": "Company", "properties": {{ "name": "Innovate Inc." }} }}
Candidates: [ {{ "node_id": 205, "properties": {{ "name": "Innovate Corp", "location": "New York" }} }} ]
Response:
```json
{{
  "decision": "CREATE_NEW",
  "reason": "'Innovate Inc.' and 'Innovate Corp' could be different companies despite the similar names. Without more context, it's safer to create a new entity.",
  "merge_target_id": null
}}
```
Your decision: