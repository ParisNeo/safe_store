# examples/graph_usage.py
import safe_store
from safe_store import GraphStore, LogLevel
import pipmaster as pm
pm.ensure_packages(["lollms_client"])
from lollms_client import LollmsClient
from lollms_client.lollms_types import MSG_TYPE 
from ascii_colors import ASCIIColors, trace_exception

from pathlib import Path
import json
import shutil
from typing import Dict, List, Any, Optional, Callable 

# --- Configuration ---
DB_FILE = "graph_example_store.db"
DOC_DIR = Path("temp_docs_graph_example")

# --- LOLLMS Client Configuration ---
BINDING_NAME = "ollama" 
HOST_ADDRESS = "http://localhost:11434"
MODEL_NAME = "mistral:latest" 

LC_CLIENT: Optional[LollmsClient] = None

def initialize_lollms_client() -> bool:
    global LC_CLIENT
    if LC_CLIENT is None:
        ASCIIColors.info(f"Initializing LollmsClient: Binding='{BINDING_NAME}', Host='{HOST_ADDRESS}', Model='{MODEL_NAME}'")
        lc_params: Dict[str, Any] = {
            "binding_name": BINDING_NAME, "host_address": HOST_ADDRESS, "model_name": MODEL_NAME,
        }
        if lc_params.get("host_address") is None and BINDING_NAME in ["openai"]: del lc_params["host_address"]
        try:
            LC_CLIENT = LollmsClient(**lc_params)
            if not hasattr(LC_CLIENT, 'binding') or LC_CLIENT.binding is None :
                 ASCIIColors.error(f"LollmsClient binding '{BINDING_NAME}' could not be loaded."); LC_CLIENT = None; return False
            ASCIIColors.success("LollmsClient initialized (ping skipped for speed).")
            return True
        except Exception as e:
            ASCIIColors.error(f"Failed to initialize LollmsClient: {e}"); trace_exception(e); LC_CLIENT = None; return False
    return True

# This single callback now just executes the prompt given by GraphStore
def llm_executor_callback(full_prompt: str) -> str:
    """
    Executes a given prompt using the global LollmsClient and returns the raw string response.
    This callback is used by GraphStore for both graph extraction and query parsing.
    """
    global LC_CLIENT
    if LC_CLIENT is None: 
        raise ConnectionError("LollmsClient not initialized. Cannot execute prompt.")

    ASCIIColors.debug(f"LLM Executor: Sending prompt (len {len(full_prompt)}) to LLM...")
    # ASCIIColors.debug(f"LLM Executor: Full prompt:\n{full_prompt[:1000]}...") # Log snippet of prompt

    response_text: Optional[str] = None
    try:
        # generate_code is used because GraphStore's internal prompts ask for JSON in markdown
        response_text = LC_CLIENT.generate_code(
            full_prompt, 
            language="json", # Expected language within the markdown block
            max_size=4096,   # Generous max size
            temperature=0.05, # Low temperature for structured output
            top_k=5
        )
        ASCIIColors.cyan(f"GraphStore: Raw LLM response for query parsing \n{response_text}")
        if not response_text: # generate_code might return empty if no block found
            ASCIIColors.warning("LLM Executor: generate_code returned empty response.")
            return "" # Return empty string, GraphStore will handle it

        ASCIIColors.debug(f"LLM Executor: Raw response from generate_code (len {len(response_text)}):\n{response_text[:500]}...")
        return response_text # Return the (hopefully clean JSON) string

    except Exception as e:
        ASCIIColors.error(f"Error during LLM execution in callback: {e}")
        trace_exception(e)
        # Re-raise as a generic exception, GraphStore's LLMCallbackError will wrap it
        raise RuntimeError(f"LLM execution failed: {e}") from e


def print_header(title: str): print("\n" + "="*20 + f" {title} " + "="*20)

def cleanup():
    print_header("Cleaning Up")
    paths_to_remove = [Path(DB_FILE), Path(f"{DB_FILE}.lock"), Path(f"{DB_FILE}-wal"), Path(f"{DB_FILE}-shm"), DOC_DIR]
    for p in paths_to_remove:
        if p.is_file():
            try: p.unlink(missing_ok=True); print(f"- Removed file: {p}")
            except OSError as e: print(f"- Warning: Could not remove file {p}: {e}")
        elif p.is_dir(): shutil.rmtree(p, ignore_errors=True); print(f"- Removed directory: {p}")

if __name__ == "__main__":
    cleanup()
    if not initialize_lollms_client():
        ASCIIColors.error("Exiting: LollmsClient initialization failure."); exit(1)

    print_header("Preparing Documents & SafeStore")
    ASCIIColors.set_log_level(LogLevel.INFO) 
    store: Optional[safe_store.SafeStore] = None
    doc_ids: List[int] = []
    doc1_path = DOC_DIR / "company_info.txt"
    doc2_path = DOC_DIR / "research_paper_snippet.txt"

    try:
        DOC_DIR.mkdir(exist_ok=True, parents=True)
        doc1_content = "Acme Innovations, led by CEO Dr. Evelyn Reed, is a tech company based in Silicon Valley. Their flagship product, 'NovaCore', was launched in 2023. John Doe works as a Senior Engineer at Acme Innovations and reports to Dr. Reed. Acme Innovations is a competitor of Beta Solutions. Beta Solutions is located in New York. Dr. Reed previously worked at GenTech Inc. before founding Acme Innovations."
        doc1_path.write_text(doc1_content.strip(), encoding='utf-8')
        doc2_content = "The research paper 'Quantum Entanglement in Nanostructures' by Dr. Alice Smith and Prof. Bob Johnson was published in the 'Journal of Advanced Physics'. Dr. Smith is affiliated with MIT. Prof. Bob Johnson is from Stanford University. This paper cites work by Dr. Evelyn Reed on early quantum theories. Dr. Reed is also known for her work at Acme Innovations."
        doc2_path.write_text(doc2_content.strip(), encoding='utf-8')
        
        store = safe_store.SafeStore(DB_FILE)
        with store:
            store.add_document(doc1_path, chunk_size=250, chunk_overlap=40)
            store.add_document(doc2_path, chunk_size=250, chunk_overlap=40)
            docs_in_db = store.list_documents()
            for doc_info in docs_in_db: doc_ids.append(doc_info['doc_id'])
    except Exception as e: ASCIIColors.error(f"Error with SafeStore: {e}"); trace_exception(e); exit(1)
    if not doc_ids: ASCIIColors.error("No documents in SafeStore. Exiting."); exit(1)

    print_header("Initializing GraphStore & Building Graph")
    graph_store_instance: Optional[GraphStore] = None
    try:
        graph_store_instance = GraphStore(
            db_path=DB_FILE,
            llm_executor_callback=llm_executor_callback, # Pass the single executor callback
            # Optionally, provide custom prompt templates here if needed:
            # graph_extraction_prompt_template="Your custom extraction prompt with {chunk_text}",
            # query_parsing_prompt_template="Your custom query parsing prompt with {natural_language_query}"
        )
        with graph_store_instance: 
            for doc_id, doc_path_obj in zip(doc_ids, [doc1_path, doc2_path]):
                 ASCIIColors.info(f"Building graph for document ID: {doc_id} ('{doc_path_obj.name}')...")
                 graph_store_instance.build_graph_for_document(doc_id)
        ASCIIColors.success("GraphStore initialized and graph building complete.")
    except Exception as e: ASCIIColors.error(f"Error with GraphStore: {e}"); trace_exception(e); exit(1)

    if graph_store_instance:
        with graph_store_instance:
            print_header("Demonstrating Graph Read Methods")
            # ... (Read method demonstrations remain the same as previous example) ...
            person_nodes = graph_store_instance.get_nodes_by_label("Person", limit=10)
            ASCIIColors.blue("\n1. Nodes with label 'Person':")
            if person_nodes: [print(f"  - ID: {n['node_id']}, Sig: {n.get('unique_signature')}, Props: {n.get('properties')}") for n in person_nodes]
            else: print("  No 'Person' nodes found.")
            evelyn_reed_node_id: Optional[int] = next((n['node_id'] for n in person_nodes if isinstance(n.get('properties'), dict) and n['properties'].get('name', '').lower() == 'dr. evelyn reed'), None)
            if evelyn_reed_node_id: ASCIIColors.green(f"(Found Dr. Evelyn Reed, Node ID: {evelyn_reed_node_id})")
            elif person_nodes: evelyn_reed_node_id = person_nodes[0]['node_id']; ASCIIColors.yellow(f"(Using first Person node ID: {evelyn_reed_node_id} as Dr. Reed not found by name)")
            if evelyn_reed_node_id:
                ASCIIColors.blue(f"\n2. Details for Node ID {evelyn_reed_node_id}:")
                details = graph_store_instance.get_node_details(evelyn_reed_node_id); print(f"  - {details}" if details else "Not found.")
                ASCIIColors.blue(f"\n3. Relationships for Node ID {evelyn_reed_node_id} (any):")
                rels = graph_store_instance.get_relationships(evelyn_reed_node_id, direction="any", limit=5)
                if rels: [print(f"  - ID: {r['relationship_id']}, Type: {r['type']}, Src: {r['source_node_id']}, Tgt: {r['target_node_id']}, Props: {r.get('properties')}") for r in rels]
                else: print("  No relationships found.")
                ASCIIColors.blue(f"\n4. Outgoing Neighbors of Node ID {evelyn_reed_node_id}:")
                neighbors = graph_store_instance.find_neighbors(evelyn_reed_node_id, direction="outgoing", limit=5)
                if neighbors: [print(f"  - ID: {n['node_id']}, Label: {n['label']}, Props: {n.get('properties')}") for n in neighbors]
                else: print("  No outgoing neighbors found.")
                ASCIIColors.blue(f"\n5. Chunks linked to Node ID {evelyn_reed_node_id}:")
                chunks = graph_store_instance.get_chunks_for_node(evelyn_reed_node_id, limit=2)
                if chunks: [print(f"  - Chunk ID: {c['chunk_id']}, File: {Path(c['file_path']).name}, Text: '{c['chunk_text'][:70]}...'") for c in chunks]
                else: print("  No chunks found linked.")
            else: ASCIIColors.warning("Could not find a suitable 'Person' node ID for detailed read method demos.")


            print_header("Demonstrating query_graph")
            queries_to_test = [
                "Who is Dr. Evelyn Reed and what companies is she associated with?",
                "What products does Acme Innovations have?",
                "Tell me about John Doe's work."
            ]
            output_modes_to_test: List[str] = ["graph_only", "chunks_summary", "full"]

            for nl_query in queries_to_test:
                print_header(f"Query: \"{nl_query}\"")
                for mode in output_modes_to_test:
                    ASCIIColors.blue(f"\n--- Output Mode: {mode} ---")
                    try:
                        result = graph_store_instance.query_graph(nl_query, output_mode=mode)
                        if mode == "graph_only":
                            nodes = result.get('nodes',[])
                            rels = result.get('relationships',[])
                            print(f"  Nodes found: {len(nodes)}, Relationships: {len(rels)}")
                            if nodes: print(f"  Sample Node 0: {nodes[0] if nodes else 'N/A'}")
                            if rels: print(f"  Sample Relationship 0: {rels[0] if rels else 'N/A'}")
                        elif mode == "chunks_summary":
                            print(f"  Chunks found: {len(result)}")
                            for i, chunk_res in enumerate(result[:2]):
                                print(f"    Chunk {i+1}: ID {chunk_res.get('chunk_id')}, File: {Path(chunk_res.get('file_path','N/A')).name}, Text: '{str(chunk_res.get('chunk_text','N/A'))[:70]}...'")
                                print(f"      Linked to graph nodes: {chunk_res.get('linked_graph_nodes')}")
                        elif mode == "full":
                            graph_part = result.get('graph',{})
                            chunks_part = result.get('chunks',[])
                            print(f"  Graph: Nodes {len(graph_part.get('nodes',[]))}, Rels {len(graph_part.get('relationships',[]))}")
                            print(f"  Chunks found: {len(chunks_part)}")
                            if chunks_part:
                                print(f"    Sample Chunk 0: ID {chunks_part[0].get('chunk_id')}, Text: '{str(chunks_part[0].get('chunk_text','N/A'))[:70]}...'")
                                print(f"      Linked to: {chunks_part[0].get('linked_graph_nodes')}")
                            if graph_part.get('nodes'): print(f"    Sample Node 0: {graph_part['nodes'][0] if graph_part['nodes'] else 'N/A'}")
                    except safe_store.LLMCallbackError as e: ASCIIColors.error(f"  LLM Callback/Parsing Error for query: {e}")
                    except safe_store.ConfigurationError as e: ASCIIColors.warning(f"  Query demo skipped: {e}")
                    except Exception as e: ASCIIColors.error(f"  Error during query_graph: {e}"); trace_exception(e)
    
    print_header("Example Finished")
    ASCIIColors.info(f"Database file is at: {Path(DB_FILE).resolve()}")