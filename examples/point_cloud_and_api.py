# examples/point_cloud_and_api.py
import safe_store
from pathlib import Path
import shutil
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import pipmaster as pm

# Ensure necessary packages for PCA and the example are installed
pm.ensure_packages(["scikit-learn", "pandas"])

# --- Helper Functions ---
def print_header(title):
    print("\n" + "="*10 + f" {title} " + "="*10)

def setup_environment():
    """Cleans up old files and creates new ones for the example."""
    print_header("Setting Up Example Environment")
    db_file = Path("point_cloud_example.db")
    doc_dir = Path("temp_docs_point_cloud")
    
    # Clean up DB and its artifacts
    for p in [db_file, Path(f"{db_file}.lock"), Path(f"{db_file}-wal"), Path(f"{db_file}-shm")]:
        p.unlink(missing_ok=True)
    
    # Clean up and create doc directory
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
    doc_dir.mkdir(exist_ok=True)

    # Create sample documents with metadata
    (doc_dir / "animals.txt").write_text(
        "The quick brown fox jumps over the lazy dog. A fast red fox is athletic. The sleepy dog rests."
    )
    (doc_dir / "tech.txt").write_text(
        "Python is a versatile programming language. Many developers use Python for AI. RAG pipelines are a common use case."
    )
    (doc_dir / "space.txt").write_text(
        "The sun is a star at the center of our solar system. The Earth revolves around the sun. Space exploration is fascinating."
    )
    
    print("- Created sample documents and cleaned up old database.")
    return db_file, doc_dir

# --- Main Logic ---
DB_FILE, DOC_DIR = setup_environment()

print_header("Initializing SafeStore and Indexing Documents")
# Initialize SafeStore
store = safe_store.SafeStore(
    db_path=DB_FILE,
    vectorizer_name="st",
    vectorizer_config={"model": "all-MiniLM-L6-v2"},
    chunk_size=10, # small chunks for more points
    chunk_overlap=2
)

# Add documents to the store with metadata
with store:
    store.add_document(DOC_DIR / "animals.txt", metadata={"topic": "animals", "source": "fiction"})
    store.add_document(DOC_DIR / "tech.txt", metadata={"topic": "technology", "source": "documentation"})
    store.add_document(DOC_DIR / "space.txt", metadata={"topic": "space", "source": "science"})

print("- Documents indexed successfully.")

# --- Data Export for Visualization ---
print_header("Exporting Point Cloud Data")
with store:
    point_cloud_data = store.export_point_cloud(output_format='dict')

# Save data to a JSON file for the web page to fetch
web_dir = Path("point_cloud_web_app")
web_dir.mkdir(exist_ok=True)
data_file = web_dir / "data.json"
with open(data_file, "w") as f:
    json.dump(point_cloud_data, f)

print(f"- Point cloud data exported to {data_file}")

# --- Web Server and HTML Page ---

# NOTE on CDNs: For this self-contained example, we use CDNs for Tailwind CSS and Plotly.
# This is for simplicity and ease of use. In a production environment, you should follow
# best practices by installing these libraries as part of a build process.
# - Tailwind: https://tailwindcss.com/docs/installation
# - Plotly: https://plotly.com/javascript/getting-started/
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SafeStore | 2D Chunk Visualization</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        /* Custom scrollbar for a more subtle look */
        .custom-scrollbar::-webkit-scrollbar { width: 8px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background-color: rgba(100, 116, 139, 0.5); border-radius: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background-color: rgba(100, 116, 139, 0.8); }
    </style>
</head>
<body class="bg-slate-50 dark:bg-slate-900 text-slate-800 dark:text-slate-200 font-sans antialiased">

    <main class="container mx-auto p-4 md:p-8">
        <!-- Header -->
        <header class="text-center mb-8 md:mb-12">
            <h1 class="text-3xl md:text-4xl font-bold text-slate-900 dark:text-white">
                2D Document Chunk Visualization
            </h1>
            <p class="mt-2 text-lg text-slate-600 dark:text-slate-400 max-w-3xl mx-auto">
                An interactive PCA plot of vectorized document chunks. Each point represents a piece of text, clustered by semantic similarity. Powered by <code class="font-mono text-sm bg-slate-200 dark:bg-slate-700 p-1 rounded">SafeStore</code>.
            </p>
        </header>

        <!-- Main Content Area -->
        <div class="grid grid-cols-1 lg:grid-cols-5 gap-8">

            <!-- Plot Card -->
            <div class="lg:col-span-3 bg-white dark:bg-slate-800 rounded-xl shadow-lg p-4 sm:p-6 h-[60vh] md:h-[70vh]">
                <div id="plot" class="w-full h-full"></div>
            </div>

            <!-- Chunk Info Card -->
            <div class="lg:col-span-2 bg-white dark:bg-slate-800 rounded-xl shadow-lg p-4 sm:p-6">
                <h2 class="text-2xl font-semibold text-slate-900 dark:text-white mb-4">Chunk Inspector</h2>
                <div id="chunk-info-container" class="relative min-h-[300px] lg:h-[calc(70vh-80px)]">
                    <!-- This content will be replaced by JS -->
                </div>
            </div>
        </div>
    </main>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const chunkCache = new Map();
            const chunkInfoContainer = document.getElementById('chunk-info-container');
            const plotDiv = document.getElementById('plot');

            const states = {
                initial: `
                    <div class="flex flex-col items-center justify-center h-full text-center text-slate-500 dark:text-slate-400">
                        <svg class="w-12 h-12 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M15.042 21.672L13.684 16.6m0 0l-2.51 2.225.569-9.47 5.227 7.917-3.286-.672zm-7.518-.267A8.25 8.25 0 1120.25 10.5M8.288 14.212A5.25 5.25 0 1117.25 10.5" /></svg>
                        <h3 class="font-semibold text-lg">Hover over a point</h3>
                        <p class="text-sm">The text content of the selected chunk will appear here.</p>
                    </div>`,
                loading: `
                    <div class="flex items-center justify-center h-full text-center text-slate-500 dark:text-slate-400">
                        <svg class="animate-spin -ml-1 mr-3 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                        <span>Loading chunk text...</span>
                    </div>`,
                error: (message) => `
                    <div class="flex flex-col items-center justify-center h-full text-center text-red-500">
                        <h3 class="font-semibold text-lg">Error</h3>
                        <p class="text-sm">${message}</p>
                    </div>`,
                content: (data) => `
                    <div class="custom-scrollbar h-full overflow-y-auto bg-slate-50 dark:bg-slate-700/50 p-4 rounded-lg">
                        <div class="font-mono text-xs text-slate-500 dark:text-slate-400 mb-4">
                            <p><strong>Chunk ID:</strong> ${data.chunk_id}</p>
                            <p><strong>Document:</strong> ${data.file_path.split(/[\\\\/]/).pop()}</p>
                            <p><strong>Metadata:</strong> ${JSON.stringify(data.document_metadata)}</p>
                        </div>
                        <p class="whitespace-pre-wrap font-mono text-sm text-slate-700 dark:text-slate-300">${data.chunk_text}</p>
                    </div>`
            };

            function setInfoState(state, data = null) {
                chunkInfoContainer.innerHTML = typeof states[state] === 'function' ? states[state](data) : states[state];
            }

            async function fetchChunkData(chunkId) {
                if (chunkCache.has(chunkId)) {
                    return chunkCache.get(chunkId);
                }
                setInfoState('loading');
                try {
                    const response = await fetch('/chunk/' + chunkId);
                    if (!response.ok) { throw new Error(`Server returned status ${response.status}`); }
                    const data = await response.json();
                    chunkCache.set(chunkId, data);
                    return data;
                } catch (error) {
                    console.error("Fetch error:", error);
                    return { error: `Could not fetch data for chunk ${chunkId}.` };
                }
            }

            // Set initial state immediately for better UX
            setInfoState('initial');

            fetch('data.json')
                .then(response => response.json())
                .then(data => {
                    if (!data || data.length === 0) {
                        plotDiv.innerHTML = '<div class="flex items-center justify-center h-full text-slate-500">No point cloud data found.</div>';
                        return;
                    }

                    const uniqueDocs = [...new Set(data.map(d => d.document_path))];
                    const traces = uniqueDocs.map(docPath => {
                        const points = data.filter(d => d.document_path === docPath);
                        // CORRECTED: The JS template literal `${...}` needs to be escaped for Python's f-string by doubling the braces `{{...}}`
                        // But since this is a multi-line string literal, we don't need f-string formatting here. The original issue was a typo in the variable.
                        const textLabels = points.map(p => `<b>${p.document_title}</b><br>Topic: ${p.metadata.topic || 'N/A'}`);
                        return {
                            x: points.map(p => p.x),
                            y: points.map(p => p.y), // This was the original typo location
                            mode: 'markers',
                            type: 'scatter',
                            name: points[0].document_title,
                            text: textLabels, // Using the correctly generated labels
                            customdata: points.map(p => p.chunk_id),
                            hoverinfo: 'text',
                            marker: { size: 12, opacity: 0.8 }
                        };
                    });

                    const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                    const layout = {
                        title: 'PCA of Document Chunks',
                        hovermode: 'closest',
                        showlegend: true,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: isDarkMode ? '#e2e8f0' : '#334155' },
                        legend: { title: { text: 'Documents' } },
                        xaxis: { title: 'PCA Component 1', gridcolor: isDarkMode ? '#334155' : '#e2e8f0' },
                        yaxis: { title: 'PCA Component 2', gridcolor: isDarkMode ? '#334155' : '#e2e8f0' }
                    };

                    Plotly.newPlot('plot', traces, layout, {responsive: true});

                    plotDiv.on('plotly_hover', async function(eventData) {
                        const chunkId = eventData.points[0].customdata;
                        const data = await fetchChunkData(chunkId);
                        if (data.error) {
                            setInfoState('error', data.error);
                        } else {
                            setInfoState('content', data);
                        }
                    });

                    plotDiv.on('plotly_unhover', () => setInfoState('initial'));
                })
                .catch(err => {
                    console.error("Error loading data.json:", err);
                    setInfoState('error', 'Could not load data.json. Is the server running correctly?');
                });
        });
    </script>
</body>
</html>
"""

# Write the HTML file
index_file = web_dir / "index.html"
index_file.write_text(html_content)

# Define a custom request handler to serve files and provide an API
class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(web_dir), **kwargs)

    def do_GET(self):
        if self.path.startswith('/chunk/'):
            try:
                chunk_id = int(self.path.split('/')[-1])
                with store:
                    chunk_data = store.get_chunk_by_id(chunk_id)
                
                if chunk_data:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(chunk_data).encode('utf-8'))
                else:
                    self.send_error(404, "Chunk not found")
            except Exception as e:
                self.send_error(500, str(e))
            return
        
        super().do_GET()

print(f"- Wrote web application files to '{web_dir.resolve()}'")

# --- Run Server ---
PORT = 8008
server_address = ('', PORT)
httpd = HTTPServer(server_address, CustomHandler)
url = f"http://localhost:{PORT}"

print_header("Starting Web Server")
print(f"Serving visualization at: {url}")
print("Please open the URL in your web browser.")
print("Press Ctrl+C to stop the server.")

# Open browser in a separate thread to not block the server start
threading.Timer(1.5, lambda: webbrowser.open(url)).start()

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\n- Server stopped.")
finally:
    httpd.server_close()
    # Final cleanup
    print_header("Final Cleanup")
    if DOC_DIR.exists(): shutil.rmtree(DOC_DIR)
    if web_dir.exists(): shutil.rmtree(web_dir)
    DB_FILE.unlink(missing_ok=True)
    print("- Removed all temporary files and database.")