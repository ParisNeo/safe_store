# [FINAL & ROBUST] examples/graph_usage.py
import safe_store
from safe_store import GraphStore, LogLevel, SafeStore
import pipmaster as pm

pm.ensure_packages(["lollms_client"])
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors, trace_exception
import sqlite3
from pathlib import Path
import json
import shutil
from typing import Dict, List, Any, Optional

# --- Configuration ---
DB_FILE = "graph_example_store.db"
DOC_DIR = Path("temp_docs_graph_example")

# --- LOLLMS Client Configuration ---
BINDING_NAME = "ollama"
HOST_ADDRESS = "http://localhost:11434"
MODEL_NAME = "mistral:latest"

# --- Ontology Definitions ---
DETAILED_ONTOLOGY = {
    "nodes": {
        "Person": {"description": "A human individual.", "properties": {"name": "string", "title": "string"}},
        "Company": {"description": "A commercial business.", "properties": {"name": "string", "location": "string"}},
        "Product": {"description": "A product created by a company.", "properties": {"name": "string"}},
        "ResearchPaper": {"description": "An academic publication.", "properties": {"title": "string"}},
        "University": {"description": "An institution of higher education.", "properties": {"name": "string"}}
    },
    "relationships": {
        "WORKS_AT": {"description": "Person is employed by Company.", "source": "Person", "target": "Company"},
        "CEO_OF": {"description": "Person is the CEO of Company.", "source": "Person", "target": "Company"},
        "FOUNDED": {"description": "Person founded a Company.", "source": "Person", "target": "Company"},
        "COMPETITOR_OF": {"description": "Company is a competitor of another Company.", "source": "Company", "target": "Company"},
        "PRODUCES": {"description": "Company creates a Product.", "source": "Company", "target": "Product"},
        "AUTHOR_OF": {"description": "Person wrote a ResearchPaper.", "source": "Person", "target": "ResearchPaper"},
        "AFFILIATED_WITH": {"description": "Person is associated with a University.", "source": "Person", "target": "University"}
    }
}
SIMPLE_ONTOLOGY = {
    "nodes": {"Entity": {"description": "A person, company, or organization.", "properties": {"name": "string"}}},
    "relationships": {"IS_RELATED_TO": {"description": "Indicates a general connection between two entities.", "source": "Entity", "target": "Entity"}}
}

LC_CLIENT: Optional[LollmsClient] = None

def initialize_lollms_client() -> bool:
    global LC_CLIENT
    if LC_CLIENT is None:
        ASCIIColors.info(f"Initializing LollmsClient: Binding='{BINDING_NAME}', Host='{HOST_ADDRESS}', Model='{MODEL_NAME}'")
        try:
            LC_CLIENT = LollmsClient(llm_binding_name=BINDING_NAME, llm_binding_config={"host_address": HOST_ADDRESS, "model_name": MODEL_NAME})
            if not LC_CLIENT.llm:
                 ASCIIColors.error(f"LollmsClient binding '{BINDING_NAME}' is not ready."); LC_CLIENT = None; return False
            ASCIIColors.success("LollmsClient initialized and ready.")
            return True
        except Exception as e:
            ASCIIColors.error(f"Failed to initialize LollmsClient: {e}"); trace_exception(e); LC_CLIENT = None; return False
    return True

def llm_executor_callback(full_prompt: str) -> str:
    global LC_CLIENT
    if LC_CLIENT is None: raise ConnectionError("LollmsClient not initialized.")
    try:
        return LC_CLIENT.generate_code(full_prompt, language="json", temperature=0.05, top_k=10)
    except Exception as e:
        raise RuntimeError(f"LLM execution for JSON failed: {e}") from e

def generate_answer_from_context(question: str, graph_data: Dict, chunks_data: Optional[List[Dict]] = None) -> str:
    global LC_CLIENT
    if LC_CLIENT is None: return "LLM not available."
    context_lines = ["--- CONTEXT ---"]
    if graph_data and graph_data.get("nodes"):
        context_lines.append("\n[Graph Information]:")
        node_map = {n['node_id']: n for n in graph_data['nodes']}
        for node in graph_data['nodes']:
            context_lines.append(f"- Node {node['node_id']} ({node['label']}): {json.dumps(node.get('properties', {}))}")
        for rel in graph_data.get('relationships', []):
            src_name = node_map.get(rel['source_node_id'], {}).get('properties', {}).get('name', f"ID:{rel['source_node_id']}")
            tgt_name = node_map.get(rel['target_node_id'], {}).get('properties', {}).get('name', f"ID:{rel['target_node_id']}")
            context_lines.append(f"- Relationship: '{src_name}' --[{rel['type']}]--> '{tgt_name}'")
    if chunks_data:
        context_lines.append("\n[Relevant Text Snippets]:")
        for i, chunk in enumerate(chunks_data):
            context_lines.append(f"- Snippet {i+1}: \"{chunk['chunk_text']}\"")
    context_lines.append("\n--- END OF CONTEXT ---")
    context_str = "\n".join(context_lines)

    prompt = (f"Answer the user's question based ONLY on the provided context. Do not use prior knowledge.\n\n"
              f"{context_str}\n\nQuestion: {question}")
    
    ASCIIColors.magenta("--- Sending Synthesis Prompt to LLM ---")
    try:
        return LC_CLIENT.generate_text(prompt, n_predict=512)
    except Exception as e:
        ASCIIColors.error(f"Error during answer synthesis: {e}")
        return "Error generating the answer."

def print_header(title: str):
    print("\n" + "="*25 + f" {title} " + "="*25)

def cleanup():
    print_header("Cleaning Up Previous Run")
    paths = [Path(DB_FILE), Path(f"{DB_FILE}.lock"), Path(f"{DB_FILE}-wal"), Path(f"{DB_FILE}-shm"), DOC_DIR]
    for p in paths:
        try:
            if p.is_file(): p.unlink(missing_ok=True); print(f"- Removed file: {p}")
            elif p.is_dir(): shutil.rmtree(p, ignore_errors=True); print(f"- Removed directory: {p}")
        except OSError as e: print(f"- Warning: Could not remove {p}: {e}")

def clear_graph_data(conn: sqlite3.Connection):
    ASCIIColors.warning("\nClearing all existing graph data from the database...")
    try:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM node_chunk_links;")
        conn.execute("DELETE FROM graph_relationships;")
        conn.execute("DELETE FROM graph_nodes;")
        conn.execute("UPDATE chunks SET graph_processed_at = NULL;")
        conn.commit()
        ASCIIColors.success("Graph data cleared.")
    except sqlite3.Error as e:
        conn.rollback()
        ASCIIColors.error(f"Failed to clear graph data: {e}")

if __name__ == "__main__":
    cleanup()
    if not initialize_lollms_client():
        ASCIIColors.error("Exiting: LollmsClient initialization failure."); exit(1)

    ASCIIColors.set_log_level(LogLevel.INFO)
    
    try:
        print_header("Preparing Documents (One-time setup)")
        DOC_DIR.mkdir(exist_ok=True, parents=True)
        doc1_content = "Acme Innovations, led by CEO Dr. Evelyn Reed, is a tech company based in Silicon Valley. Their flagship product, 'NovaCore', was launched in 2023. John Doe works as a Senior Engineer at Acme Innovations and reports to Dr. Reed. Acme Innovations is a competitor of Beta Solutions."
        (DOC_DIR / "company_info.txt").write_text(doc1_content.strip(), encoding='utf-8')
        doc2_content = "The research paper 'Quantum Entanglement in Nanostructures' by Dr. Alice Smith cites work by Dr. Evelyn Reed on early quantum theories. Dr. Reed is also known for her work at Acme Innovations."
        (DOC_DIR / "research_paper_snippet.txt").write_text(doc2_content.strip(), encoding='utf-8')

        with SafeStore(db_path=DB_FILE) as store:
            store.add_document(DOC_DIR / "company_info.txt")
            store.add_document(DOC_DIR / "research_paper_snippet.txt")
            
            print_header("PASS 1: Building Graph with DETAILED Ontology")
            graph_store_detailed = GraphStore(store=store, llm_executor_callback=llm_executor_callback, ontology=DETAILED_ONTOLOGY)
            graph_store_detailed.build_graph_for_all_documents()
            ASCIIColors.success("Graph building with detailed ontology complete.")

            print_header("DEMO 1.1: RAG Query (Who is Dr. Evelyn Reed?)")
            query = "Who is Dr. Evelyn Reed and what companies is she associated with?"
            result = graph_store_detailed.query_graph(query, output_mode="full")
            full_answer = generate_answer_from_context(query, result.get('graph'), result.get('chunks'))
            ASCIIColors.green("Final Answer (from Graph + Chunks):")
            print(full_answer)

            print_header("DEMO 1.2: Manually Editing the Graph")
            ASCIIColors.info("We will manually add a new product 'ChronoLeap' and link it to an 'Acme' company.")
            
            # **CORRECTED:** Use a flexible search to find the Acme node
            company_nodes = graph_store_detailed.get_nodes_by_label("Company")
            acme_node = next((n for n in company_nodes if 'acme' in n.get('properties', {}).get('name', '').lower()), None)

            if acme_node:
                acme_id = acme_node['node_id']
                acme_name = acme_node['properties']['name']
                ASCIIColors.info(f"Found '{acme_name}' with Node ID: {acme_id}")
                
                product_id = graph_store_detailed.add_node(label="Product", properties={"name": "ChronoLeap"})
                ASCIIColors.info(f"Created new 'ChronoLeap' product with Node ID: {product_id}")
                
                rel_id = graph_store_detailed.add_relationship(acme_id, product_id, "PRODUCES")
                ASCIIColors.info(f"Linked them with 'PRODUCES' relationship (ID: {rel_id})")

                print_header("DEMO 1.3: Querying the Manually Added Data")
                manual_query = "What new products does Acme produce?"
                manual_result = graph_store_detailed.query_graph(manual_query, output_mode="full")
                manual_answer = generate_answer_from_context(manual_query, manual_result.get('graph'))
                ASCIIColors.green("Final Answer (from Graph-Only):")
                print(manual_answer)
            else:
                ASCIIColors.warning("Could not find any 'Acme' company node to perform manual edit demo.")

            print_header("PASS 2: Rebuilding Graph with SIMPLE Ontology")
            clear_graph_data(store.conn)

            graph_store_simple = GraphStore(store=store, llm_executor_callback=llm_executor_callback, ontology=SIMPLE_ONTOLOGY)
            graph_store_simple.build_graph_for_all_documents()
            ASCIIColors.success("Graph building with simple ontology complete.")
            
            print_header("DEMO 2.1: Observing the new simple graph structure")
            simple_nodes = graph_store_simple.get_nodes_by_label("Entity", limit=10)
            ASCIIColors.blue("\nNodes extracted with the simple 'Entity' label:")
            if simple_nodes:
                for n in simple_nodes: print(f"  - ID: {n['node_id']}, Props: {n.get('properties')}")
            else:
                print("  No 'Entity' nodes found.")

    except Exception as e:
        ASCIIColors.error(f"An unexpected error occurred in the main process: {e}")
        trace_exception(e)
    finally:
        print_header("Example Finished")
        ASCIIColors.info(f"Database file is at: {Path(DB_FILE).resolve()}")