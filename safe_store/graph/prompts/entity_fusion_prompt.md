# [NEW & COMPLETE] prompts/entity_fusion_prompt.md
Your task is to determine if two entities of the same type are, in fact, the same entity based on their properties.

**Entity Type:** {entity_label}

---

**Entity A Properties:**
```json
{node_a_properties}
```

---

**Entity B Properties:**
```json
{node_b_properties}
```

---

**Analysis:**
Carefully compare the properties of Entity A and Entity B. Do they refer to the same real-world entity? Consider variations in naming, partial information, or different levels of detail.

**Output Format:**
You MUST respond with only a single, well-formed JSON object in a markdown code block. The JSON object must have two keys:
1.  `"is_same"`: A boolean (`true` or `false`).
2.  `"reasoning"`: A brief, one-sentence explanation for your decision.

**Example Response:**
```json
{{
    "is_same": true,
    "reasoning": "Both entities share the same unique identifier and have highly similar descriptive properties."
}}
```