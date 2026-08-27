import sqlite3
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import pipmaster as pm
from ascii_colors import ASCIIColors
from ...core.exceptions import QueryError
from ...core import db

try:
    import rdflib
    from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL
    from rdflib.plugins.sparql.processor import SPARQLResult
except ImportError:
    pm.ensure_packages(["rdflib"])
    import rdflib
    from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL
    from rdflib.plugins.sparql.processor import SPARQLResult


class SparqlEngine:
    """
    W3C SPARQL 1.1 Compliant Query & Update Engine backed by SafeStore's SQLite Graph/Triple Store.
    Supports SELECT, ASK, CONSTRUCT, DESCRIBE, and UPDATE commands (INSERT DATA, DELETE DATA, DELETE WHERE).
    """

    def __init__(self, conn: sqlite3.Connection, default_ns: str = "http://example.org/"):
        self.conn = conn
        self.default_ns = default_ns

    def _hydrate_rdf_graph(self) -> Graph:
        """
        Hydrates an in-memory RDFLib Graph from SafeStore's SQLite graph tables.
        """
        g = Graph()
        g.bind("ex", Namespace("http://example.org/"))
        g.bind("ont", Namespace("http://example.org/ontology/"))
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        g.bind("owl", OWL)

        cursor = self.conn.cursor()
        
        # 1. Fetch nodes and properties
        cursor.execute("SELECT node_id, node_label, node_properties FROM graph_nodes")
        nodes_data = cursor.fetchall()
        node_uri_map: Dict[int, URIRef] = {}

        for node_id, label, props_json in nodes_data:
            props = json.loads(props_json) if props_json else {}
            identifying_val = props.get("identifying_value") or props.get("name") or str(node_id)

            if props.get("uri"):
                uri_str = props["uri"]
            elif identifying_val and not str(identifying_val).isdigit():
                clean_id = str(identifying_val).strip()
                uri_str = f"http://example.org/{clean_id}"
            else:
                uri_str = f"http://example.org/node/{node_id}"

            node_uri = URIRef(uri_str)
            node_uri_map[node_id] = node_uri

            # Type and Label triples
            class_uri = URIRef(f"http://example.org/ontology/{label}")
            g.add((node_uri, RDF.type, class_uri))
            g.add((node_uri, URIRef("http://example.org/ontology/label"), Literal(label)))

            # Literal and URI property triples
            seen_properties = set()
            for k, v in props.items():
                if k in ("uri", "identifying_value", "other_identifiers") or v is None or str(v).strip() == "":
                    continue

                if k.startswith("http://") or k.startswith("https://"):
                    prop_uri = URIRef(k)
                    prop_key = k
                else:
                    prop_uri = URIRef(f"http://example.org/ontology/{k}")
                    prop_key = f"http://example.org/ontology/{k}"

                if prop_key in seen_properties:
                    continue
                seen_properties.add(prop_key)

                if isinstance(v, (str, int, float, bool)):
                    g.add((node_uri, prop_uri, Literal(v)))
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, (str, int, float, bool)):
                            g.add((node_uri, prop_uri, Literal(item)))

        # 2. Fetch relationships
        cursor.execute("SELECT source_node_id, target_node_id, relationship_type, relationship_properties FROM graph_relationships")
        rel_data = cursor.fetchall()

        for src_id, tgt_id, rel_type, rel_props_json in rel_data:
            if src_id in node_uri_map and tgt_id in node_uri_map:
                src_uri = node_uri_map[src_id]
                tgt_uri = node_uri_map[tgt_id]
                
                props = json.loads(rel_props_json) if rel_props_json else {}
                pred_uri_str = props.get("uri")
                pred_uri = URIRef(pred_uri_str) if pred_uri_str else URIRef(f"http://example.org/{rel_type}")
                
                g.add((src_uri, pred_uri, tgt_uri))
                g.add((src_uri, URIRef(f"http://example.org/ontology/{rel_type}"), tgt_uri))

        return g

    def execute_query(self, sparql_query: str) -> Dict[str, Any]:
        """
        Executes any SPARQL 1.1 query (SELECT, ASK, CONSTRUCT, DESCRIBE) against the graph.
        """
        clean_query = sparql_query.strip()
        if not clean_query:
            raise QueryError("SPARQL query cannot be empty.")

        try:
            g = self._hydrate_rdf_graph()
            qres = g.query(clean_query)

            if qres.type == "SELECT":
                variables = [str(v) for v in qres.vars] if qres.vars else []
                bindings = []
                for row in qres:
                    row_dict = {}
                    for var in variables:
                        val = row[var]
                        if val is not None:
                            val_type = "uri" if isinstance(val, URIRef) else "literal"
                            row_dict[var] = {
                                "type": val_type,
                                "value": str(val)
                            }
                    bindings.append(row_dict)
                return {
                    "head": {"vars": variables},
                    "results": {"bindings": bindings}
                }

            elif qres.type == "ASK":
                return {"boolean": bool(qres.askAnswer)}

            elif qres.type in ("CONSTRUCT", "DESCRIBE"):
                triples = []
                for s, p, o in qres.graph:
                    triples.append({
                        "subject": {"type": "uri" if isinstance(s, URIRef) else "literal", "value": str(s)},
                        "predicate": {"type": "uri" if isinstance(p, URIRef) else "literal", "value": str(p)},
                        "object": {"type": "uri" if isinstance(o, URIRef) else "literal", "value": str(o)}
                    })
                return {"graph": triples, "triples": triples}

            else:
                raise QueryError(f"Unsupported query result type: {qres.type}")

        except QueryError:
            raise
        except Exception as e:
            ASCIIColors.error(f"SPARQL execution error: {e}")
            raise QueryError(f"SPARQL query failed: {e}") from e

    def execute_update(self, sparql_update: str) -> Dict[str, Any]:
        """
        Executes a W3C SPARQL 1.1 Update command (INSERT DATA, DELETE DATA, DELETE WHERE,
        MODIFY) and synchronizes all changes back to SQLite graph tables.
        """
        clean_update = sparql_update.strip()
        if not clean_update:
            raise QueryError("SPARQL update command cannot be empty.")

        try:
            g = self._hydrate_rdf_graph()
            initial_count = len(g)
            
            g.update(clean_update)
            final_count = len(g)

            synced_nodes, synced_rels = self._sync_graph_to_sqlite(g)

            return {
                "status": "success",
                "triples_before": initial_count,
                "triples_after": final_count,
                "nodes_synchronized": synced_nodes,
                "relationships_synchronized": synced_rels
            }

        except QueryError:
            raise
        except Exception as e:
            ASCIIColors.error(f"SPARQL update error: {e}")
            raise QueryError(f"SPARQL update failed: {e}") from e

    def _sync_graph_to_sqlite(self, g: Graph) -> Tuple[int, int]:
        """
        Synchronizes an updated RDF graph back into SQLite graph tables atomically.
        """
        cursor = self.conn.cursor()
        
        entities: Dict[URIRef, Dict[str, Any]] = {}
        relationships: List[Tuple[URIRef, URIRef, str, Dict[str, Any]]] = []

        for s, p, o in g:
            s_uri = str(s)
            
            if p == RDF.type:
                if s not in entities:
                    entities[s] = {"label": "Thing", "properties": {"uri": s_uri, "identifying_value": s_uri.split('/')[-1].split('#')[-1]}}
                entities[s]["label"] = str(o).split('/')[-1].split('#')[-1]

            elif isinstance(o, Literal):
                if s not in entities:
                    entities[s] = {"label": "Thing", "properties": {"uri": s_uri, "identifying_value": s_uri.split('/')[-1].split('#')[-1]}}
                prop_key = str(p).split('/')[-1].split('#')[-1]
                entities[s]["properties"][prop_key] = o.toPython()

            elif isinstance(o, URIRef):
                if s not in entities:
                    entities[s] = {"label": "Thing", "properties": {"uri": s_uri, "identifying_value": s_uri.split('/')[-1].split('#')[-1]}}
                if o not in entities:
                    entities[o] = {"label": "Thing", "properties": {"uri": str(o), "identifying_value": str(o).split('/')[-1].split('#')[-1]}}
                
                rel_label = str(p).split('/')[-1].split('#')[-1]
                relationships.append((s, o, rel_label, {"uri": str(p)}))

        uri_to_node_id: Dict[str, int] = {}
        for uri_ref, data in entities.items():
            label = data["label"]
            props = data["properties"]
            uri_str = str(uri_ref)
            sig = f"{label}:{uri_str}"
            
            cursor.execute("SELECT node_id FROM graph_nodes WHERE unique_signature = ?", (sig,))
            row = cursor.fetchone()
            if row:
                node_id = row[0]
                cursor.execute("UPDATE graph_nodes SET node_label = ?, node_properties = ? WHERE node_id = ?", 
                               (label, json.dumps(props), node_id))
            else:
                cursor.execute("INSERT INTO graph_nodes (node_label, node_properties, unique_signature) VALUES (?, ?, ?)",
                               (label, json.dumps(props), sig))
                node_id = cursor.lastrowid
            
            uri_to_node_id[uri_str] = node_id

        cursor.execute("DELETE FROM graph_relationships")
        for src_uri, tgt_uri, rel_type, rel_props in relationships:
            src_id = uri_to_node_id.get(str(src_uri))
            tgt_id = uri_to_node_id.get(str(tgt_uri))
            if src_id and tgt_id:
                cursor.execute(
                    "INSERT INTO graph_relationships (source_node_id, target_node_id, relationship_type, relationship_properties) "
                    "VALUES (?, ?, ?, ?)",
                    (src_id, tgt_id, rel_type, json.dumps(rel_props))
                )

        self.conn.commit()
        return len(entities), len(relationships)