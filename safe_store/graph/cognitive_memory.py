import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

from ascii_colors import ASCIIColors
from ..core.exceptions import SafeStoreError, GraphError, NodeNotFoundError
from ..core import db


class CognitiveMemoryStore:
    """
    High-level cognitive memory and thought reorganization system for LLMs.
    Provides episodic event logging, associative memory traversal, concept bridging,
    chunk grounding, and standardized LLM tool function calling interfaces.
    """

    def __init__(self, graph_store: Any):
        self.graph_store = graph_store

    @property
    def conn(self):
        return self.graph_store.conn

    # -------------------------------------------------------------------------
    # 1. Episodic Memory Operations
    # -------------------------------------------------------------------------
    def record_episode(
        self,
        title: str,
        description: str,
        participants: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
        source_chunk_ids: Optional[List[int]] = None,
        outcome: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Records a discrete episodic memory event, linking participants, timestamp,
        and grounding source text chunks.
        """
        event_time = timestamp or datetime.datetime.utcnow().isoformat()
        episode_id_str = f"episode_{uuid.uuid4().hex[:8]}"

        properties = {
            "identifying_value": title,
            "title": title,
            "description": description,
            "timestamp": event_time,
            "episode_id": episode_id_str,
            "outcome": outcome or "completed"
        }
        if metadata:
            properties.update(metadata)

        # 1. Create Episode Node
        episode_node_id = self.graph_store.add_node("EpisodicMemory", properties)

        # 2. Link Participants
        if participants:
            for person_name in participants:
                person_nodes = db.find_node_by_label_and_property_value(self.conn, "Person", person_name, limit=1)
                if person_nodes:
                    p_id = person_nodes[0]["node_id"]
                else:
                    p_id = self.graph_store.add_node("Person", {"identifying_value": person_name, "name": person_name})
                
                self.graph_store.add_relationship(p_id, episode_node_id, "participatedIn", {"role": "participant"})

        # 3. Ground to Source Chunks
        if source_chunk_ids:
            for cid in source_chunk_ids:
                db.link_node_to_chunk(self.conn, episode_node_id, cid)

        ASCIIColors.success(f"Recorded episodic memory: '{title}' (ID: {episode_node_id})")
        return episode_node_id

    # -------------------------------------------------------------------------
    # 2. Associative Memory & Concept Bridging
    # -------------------------------------------------------------------------
    def recall_associative(
        self,
        concept_or_entity: str,
        max_hops: int = 2,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Explores associative memory pathways originating from an entity or concept,
        returning connected individuals, episodes, and grounded chunk evidence.
        """
        seed_nodes = self._find_seed_entities(concept_or_entity)
        if not seed_nodes:
            query_vec = self.graph_store.embedder.vectorize([concept_or_entity])[0]
            candidate_ids = db.search_graph_nodes_by_vector(self.conn, query_vec, top_k=3)
            seed_nodes = [db.get_node_details_db(self.conn, cid) for cid in candidate_ids if cid]

        if not seed_nodes:
            return {"query": concept_or_entity, "associations": [], "grounded_chunks": []}

        visited_nodes: Dict[int, Dict[str, Any]] = {}
        visited_rels: List[Dict[str, Any]] = []
        queue: List[Tuple[int, int]] = []

        for node in seed_nodes:
            nid = node["node_id"]
            visited_nodes[nid] = node
            queue.append((nid, 0))

        head = 0
        while head < len(queue):
            curr_id, curr_depth = queue[head]
            head += 1

            if curr_depth >= max_hops:
                continue

            relationships = db.get_relationships_for_node_db(self.conn, curr_id, None, "any", limit=50)
            for rel in relationships:
                visited_rels.append(rel)
                neighbor = rel["target_node"] if rel["source_node_id"] == curr_id else rel["source_node"]
                nbr_id = neighbor["node_id"]

                if nbr_id not in visited_nodes:
                    visited_nodes[nbr_id] = neighbor
                    queue.append((nbr_id, curr_depth + 1))

        node_ids = list(visited_nodes.keys())
        chunks_map = db.get_chunk_ids_for_nodes_db(self.conn, node_ids)
        all_chunk_ids = list(set(cid for c_list in chunks_map.values() for cid in c_list))
        
        grounded_chunks = db.get_chunk_details_db(self.conn, all_chunk_ids, self.graph_store.encryptor) if all_chunk_ids else []

        return {
            "query": concept_or_entity,
            "seed_nodes": [n.get("label", "") + ": " + str(n.get("properties", {}).get("name", n.get("properties", {}).get("title", ""))) for n in seed_nodes],
            "associated_entities": list(visited_nodes.values())[:top_k],
            "relationships": visited_rels[:top_k * 2],
            "grounded_chunks": grounded_chunks[:5]
        }

    def _find_seed_entities(self, name_or_title: str) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        pattern = f"%{name_or_title}%"
        sql = """
            SELECT node_id, node_label, node_properties, unique_signature
            FROM graph_nodes
            WHERE node_label LIKE ? 
               OR json_extract(node_properties, '$.name') LIKE ?
               OR json_extract(node_properties, '$.title') LIKE ?
               OR json_extract(node_properties, '$.identifying_value') LIKE ?
            LIMIT 5
        """
        rows = cursor.execute(sql, (pattern, pattern, pattern, pattern)).fetchall()
        return [{"node_id": r[0], "label": r[1], "properties": json.loads(r[2]) if r[2] else {}, "unique_signature": r[3]} for r in rows]

    # -------------------------------------------------------------------------
    # 3. Grounding Individuals to Text Chunks
    # -------------------------------------------------------------------------
    def link_individual_to_chunks(self, entity_id_or_name: Union[int, str], chunk_ids: List[int]) -> None:
        """Explicitly registers chunk provenance for a graph entity."""
        if isinstance(entity_id_or_name, int):
            node_id = entity_id_or_name
        else:
            seeds = self._find_seed_entities(entity_id_or_name)
            if not seeds:
                node_id = self.graph_store.add_node("Person", {"identifying_value": entity_id_or_name, "name": entity_id_or_name})
            else:
                node_id = seeds[0]["node_id"]

        for cid in chunk_ids:
            db.link_node_to_chunk(self.conn, node_id, cid)
        ASCIIColors.success(f"Linked entity ID {node_id} to {len(chunk_ids)} chunk(s).")

    def get_grounded_evidence(self, entity_name: str) -> List[Dict[str, Any]]:
        """Fetches all source text chunks associated with an individual or concept."""
        seeds = self._find_seed_entities(entity_name)
        if not seeds:
            return []

        node_ids = [s["node_id"] for s in seeds]
        chunk_map = db.get_chunk_ids_for_nodes_db(self.conn, node_ids)
        all_chunk_ids = [cid for clist in chunk_map.values() for cid in clist]
        
        return db.get_chunk_details_db(self.conn, all_chunk_ids, self.graph_store.encryptor) if all_chunk_ids else []

    # -------------------------------------------------------------------------
    # 4. LLM Function Calling Tools
    # -------------------------------------------------------------------------
    def get_llm_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns standard JSON tool schemas for LLM function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_sparql_query",
                    "description": "Execute a SPARQL 1.1 SELECT, ASK, CONSTRUCT, or DESCRIBE query across the knowledge graph.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The SPARQL query string."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_sparql_update",
                    "description": "Execute a SPARQL 1.1 UPDATE command (INSERT DATA, DELETE DATA, DELETE WHERE, MODIFY) to reorganize or update the knowledge graph.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "update_command": {"type": "string", "description": "The SPARQL UPDATE command string."}
                        },
                        "required": ["update_command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "record_episodic_memory",
                    "description": "Record an episodic memory event (temporal episode with title, description, participants, and outcome).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Short title of the episode/event."},
                            "description": {"type": "string", "description": "Detailed explanation of what occurred."},
                            "participants": {"type": "array", "items": {"type": "string"}, "description": "List of participant entity names."},
                            "outcome": {"type": "string", "description": "Outcome or resolution of the event."},
                            "source_chunk_ids": {"type": "array", "items": {"type": "integer"}, "description": "Optional chunk IDs for evidence grounding."}
                        },
                        "required": ["title", "description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "recall_associative_memory",
                    "description": "Recall associative pathways and concepts connected to a specific topic or individual.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "concept": {"type": "string", "description": "The concept, person, or organization to explore."},
                            "max_hops": {"type": "integer", "default": 2, "description": "Graph traversal depth."}
                        },
                        "required": ["concept"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_grounded_evidence",
                    "description": "Retrieve actual source text chunk excerpts that ground a knowledge graph individual or concept.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_name": {"type": "string", "description": "Name of the entity to retrieve source text for."}
                        },
                        "required": ["entity_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "link_individual_to_chunks",
                    "description": "Bind an individual entity or concept to one or more document text chunk IDs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_name": {"type": "string", "description": "Name of the entity."},
                            "chunk_ids": {"type": "array", "items": {"type": "integer"}, "description": "List of chunk IDs to link."}
                        },
                        "required": ["entity_name", "chunk_ids"]
                    }
                }
            }
        ]

    def dispatch_llm_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Executes an LLM tool call and returns the JSON-serializable result."""
        if tool_name == "execute_sparql_query":
            return self.graph_store.query_sparql(arguments["query"])
        elif tool_name == "execute_sparql_update":
            return self.graph_store.execute_sparql_update(arguments["update_command"])
        elif tool_name == "record_episodic_memory":
            return self.record_episode(
                title=arguments["title"],
                description=arguments["description"],
                participants=arguments.get("participants"),
                outcome=arguments.get("outcome"),
                source_chunk_ids=arguments.get("source_chunk_ids")
            )
        elif tool_name == "recall_associative_memory":
            return self.recall_associative(
                concept_or_entity=arguments["concept"],
                max_hops=arguments.get("max_hops", 2)
            )
        elif tool_name == "get_grounded_evidence":
            return self.get_grounded_evidence(arguments["entity_name"])
        elif tool_name == "link_individual_to_chunks":
            self.link_individual_to_chunks(arguments["entity_name"], arguments["chunk_ids"])
            return {"status": "linked", "entity": arguments["entity_name"], "chunk_ids": arguments["chunk_ids"]}
        else:
            raise ValueError(f"Unknown tool name: '{tool_name}'")