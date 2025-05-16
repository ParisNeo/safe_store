// webui/static/script.js

// --- Configuration ---
const API_ENDPOINT = "/upload-file/";
const GRAPH_DATA_ENDPOINT = "/graph-data/";
const MAX_FILE_SIZE_MB = 5;

// --- DOM Elements ---
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const graphContainer = document.getElementById("graph-container");
const nodeInfoPanel = document.getElementById("node-info-panel");
const edgeInfoPanel = document.getElementById("edge-info-panel");
const nodeLabelDisplay = document.getElementById("node-label-display");
const nodePropertiesDisplay = document.getElementById("node-properties-display");
const edgeFromDisplay = document.getElementById("edge-from-display");
const edgeToDisplay = document.getElementById("edge-to-display");
const edgeTypeDisplay = document.getElementById("edge-type-display");
const edgePropertiesDisplay = document.getElementById("edge-properties-display");
const noSelectionMessage = document.getElementById("no-selection");

// --- Vis.js Network Instance (Global) ---
let network; // REMOVED ': any'

// --- Helper Functions ---
function showStatus(message, type = "success") { // REMOVED type annotations
    uploadStatus.innerHTML = `<i class="fas fa-${type === 'success' ? 'check-circle' : 'times-circle'}"></i> ${message}`;
    uploadStatus.className = `mt-2 text-${type}`;
}

function clearStatus() {
    uploadStatus.innerHTML = "";
    uploadStatus.className = "mt-2";
}

function bytesToMB(bytes) { // REMOVED type annotation
    return (bytes / (1024 * 1024)).toFixed(2);
}

// --- File Upload Handling ---
uploadForm?.addEventListener("submit", async (event) => { // REMOVED ': Event'
    event.preventDefault();

    const file = fileInput?.files?.[0];
    if (!file) {
        showStatus("Please select a file.", "error");
        return;
    }

    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
        showStatus(`File is too large. Max size is ${MAX_FILE_SIZE_MB}MB.`, "error");
        return;
    }

    clearStatus();
    showStatus("Uploading...", "info"); // Assuming you want an info style

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(API_ENDPOINT, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            // Use errorData.detail if available, otherwise provide a generic message
            const detail = errorData && errorData.detail ? errorData.detail : "Unknown error";
            showStatus(`Upload failed: ${detail}`, "error");
            console.error("Upload error:", errorData);
            return;
        }

        const data = await response.json();
        showStatus(`File uploaded and processed successfully.`, "success");
        console.log("Upload success:", data);
        fetchGraphData();

    } catch (error) {
        showStatus(`Upload failed: ${error}`, "error");
        console.error("Upload error:", error);
    }
});

// --- Graph Data Fetching ---
async function fetchGraphData() { // REMOVED type annotations
    try {
        const response = await fetch(GRAPH_DATA_ENDPOINT);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log("Graph data fetched:", data);
        renderGraph(data.nodes, data.edges);
    } catch (error) {
        console.error("Error fetching graph data:", error);
        if (graphContainer) { // Check if graphContainer exists
            graphContainer.innerHTML = `<div class="alert alert-danger">Error loading graph data: ${error}</div>`;
        }
    }
}

// --- Graph Rendering with Vis.js Network ---
function renderGraph(nodes, edges) { // REMOVED type annotations
    if (!graphContainer) { // Check if graphContainer exists
        console.error("Graph container not found!");
        return;
    }
    graphContainer.innerHTML = "";

    const visNodes = new vis.DataSet(nodes);
    const visEdges = new vis.DataSet(edges);

    const data = {
        nodes: visNodes,
        edges: visEdges,
    };

    const options = {
        nodes: {
            shape: "dot",
            size: 16,
            font: {
                size: 14,
                color: '#1d2129',
            },
            borderWidth: 2,
            borderWidthSelected: 3,
            color: {
                border: '#2B7CE9',
                background: '#97C2FC',
                highlight: {
                    border: '#2B7CE9',
                    background: '#D2E5FF'
                }
            },
            shadow: {
                enabled: true,
                color: 'rgba(0,0,0,0.2)',
                size: 10,
                x: 5,
                y: 5
            },
        },
        edges: {
            color: '#2B7CE9',
            width: 2,
            smooth: {
                enabled: true,
                type: 'dynamic',
                roundness: 0.5
            },
            shadow: {
                enabled: true,
                color: 'rgba(0,0,0,0.1)',
                size: 5,
                x: 2,
                y: 2
            },
            font: {
                size: 12,
                color: '#1d2129',
                strokeWidth: 3,
                strokeColor: '#ffffff'
            }
        },
        interaction: {
            tooltipDelay: 200,
        },
        layout: {
            hierarchical: {
                enabled: false,
                direction: "UD",
                sortMethod: "directed",
            },
            improvedLayout: true,
        },
        physics: {
            enabled: true,
            barnesHut: {
                gravitationalConstant: -80000,
                centralGravity: 0.01,
                springLength: 100,
                springConstant: 0.01,
                damping: 0.09,
                avoidOverlap: 0.1
            },
            stabilization: {
                enabled: true,
                iterations: 1000,
                updateInterval: 25
            }
        }
    };

    network = new vis.Network(graphContainer, data, options);

    network.on("click", (params) => { // REMOVED ': any'
        // Ensure all panel elements exist before trying to modify them
        if (nodeInfoPanel) nodeInfoPanel.style.display = "none";
        if (edgeInfoPanel) edgeInfoPanel.style.display = "none";
        if (noSelectionMessage) noSelectionMessage.style.display = "block";

        const { nodes: clickedNodes, edges: clickedEdges } = params; // Destructure to avoid conflict

        if (clickedNodes.length > 0) {
            const nodeId = clickedNodes[0];
            const node = visNodes.get(nodeId);
            if (node && nodeLabelDisplay && nodePropertiesDisplay && nodeInfoPanel && noSelectionMessage) {
                nodeLabelDisplay.textContent = node.original_label || node.label;
                nodePropertiesDisplay.textContent = JSON.stringify(node.properties, null, 2);
                nodeInfoPanel.style.display = "block";
                noSelectionMessage.style.display = "none";
            }
        } else if (clickedEdges.length > 0) {
            const edgeId = clickedEdges[0];
            const edge = visEdges.get(edgeId);
            if (edge && edgeFromDisplay && edgeToDisplay && edgeTypeDisplay && edgePropertiesDisplay && edgeInfoPanel && noSelectionMessage) {
                edgeFromDisplay.textContent = String(edge.from); // Ensure string conversion for display
                edgeToDisplay.textContent = String(edge.to);     // Ensure string conversion for display
                edgeTypeDisplay.textContent = edge.label || "No Type";
                edgePropertiesDisplay.textContent = JSON.stringify(edge.properties, null, 2);
                edgeInfoPanel.style.display = "block";
                noSelectionMessage.style.display = "none";
            }
        }
    });
}

// --- Initial Graph Data Fetch on Load ---
if (document.readyState === 'loading') { //DOMContentLoaded alternative
    document.addEventListener('DOMContentLoaded', fetchGraphData);
} else {
    fetchGraphData(); // Or call it directly if already loaded
}