// webui/static_assets/script.js

// --- Configuration ---
const API_BASE_URL = ""; // Assuming API calls are relative to the current host
const API_ENDPOINT_UPLOAD = `${API_BASE_URL}/upload-file/`;
const API_ENDPOINT_GRAPH_DATA = `${API_BASE_URL}/graph-data/`;
const API_ENDPOINT_GRAPH_SEARCH = `${API_BASE_URL}/graph/search/`;
const API_ENDPOINT_GRAPH_FUSE = `${API_BASE_URL}/graph/fuse/`;
const API_ENDPOINT_NODE = `${API_BASE_URL}/graph/node/`;
const API_ENDPOINT_EDGE = `${API_BASE_URL}/graph/edge/`;
const API_ENDPOINT_NEIGHBORS = (nodeId) => `${API_BASE_URL}/graph/node/${nodeId}/neighbors`;
const API_ENDPOINT_PATH = `${API_BASE_URL}/graph/path`;
const API_ENDPOINT_CHAT = `${API_BASE_URL}/api/chat/rag`;
const API_ENDPOINT_DATABASES = `${API_BASE_URL}/api/databases`; // GET, POST
const API_ENDPOINT_DATABASE_ACTION = (dbName, action) => `${API_BASE_URL}/api/databases/${encodeURIComponent(dbName)}/${action}`; // activate, delete

// --- DOM Elements ---
// (DOM elements are unchanged from previous step)
const themeToggleBtn = document.getElementById("theme-toggle-btn"), themeIconSun = document.getElementById("theme-icon-sun"), themeIconMoon = document.getElementById("theme-icon-moon"), settingsBtn = document.getElementById("settings-btn"), uploadDocumentBtn = document.getElementById("upload-document-btn"), databasesBtn = document.getElementById("databases-btn"), graphLoadingOverlay = document.getElementById("graph-loading-overlay"), graphContainer = document.getElementById("graph-container"), graphActionStatus = document.getElementById("graph-action-status"), taskProgressContainer = document.getElementById("task-progress-container"), nodeInfoPanel = document.getElementById("node-info-panel"), edgeInfoPanel = document.getElementById("edge-info-panel"), nodeIDDisplay = document.getElementById("node-id-display"), nodeLabelDisplay = document.getElementById("node-label-display"), nodePropertiesDisplay = document.getElementById("node-properties-display"), editSelectedNodeBtn = document.getElementById("edit-selected-node-btn"), expandNeighborsBtn = document.getElementById("expand-neighbors-btn"), edgeIDDisplay = document.getElementById("edge-id-display"), edgeFromDisplay = document.getElementById("edge-from-display"), edgeToDisplay = document.getElementById("edge-to-display"), edgeTypeDisplay = document.getElementById("edge-type-display"), edgePropertiesDisplay = document.getElementById("edge-properties-display"), editSelectedEdgeBtn = document.getElementById("edit-selected-edge-btn"), noSelectionMessage = document.getElementById("no-selection"), graphSearchForm = document.getElementById("graph-search-form"), graphSearchInput = document.getElementById("graph-search-input"), clearSearchBtn = document.getElementById("clear-search-btn"), searchResultsContainer = document.getElementById("search-results-container"), searchResultsList = document.getElementById("search-results-list"), editModeToggle = document.getElementById("edit-mode-toggle"), editModeHint = document.getElementById("edit-mode-hint"), togglePhysicsBtn = document.getElementById("toggle-physics-btn"), physicsBtnText = document.getElementById("physics-btn-text"), fuseEntitiesBtn = document.getElementById("fuse-entities-btn"), fitGraphBtn = document.getElementById("fit-graph-btn"), layoutSelect = document.getElementById("layout-select"), applyLayoutBtn = document.getElementById("apply-layout-btn"), findPathForm = document.getElementById("find-path-form"), startNodeInput = document.getElementById("start-node-input"), endNodeInput = document.getElementById("end-node-input"), uploadModal = document.getElementById("upload-modal"), closeUploadModalBtn = document.getElementById("close-upload-modal-btn"), uploadForm = document.getElementById("upload-form"), fileInput = document.getElementById("file-input"), extractionGuidanceInput = document.getElementById("extraction-guidance-input"), fileListPreview = document.getElementById("file-list-preview"), uploadSubmitBtn = document.getElementById("upload-submit-btn"), uploadProgressArea = document.getElementById("upload-progress-area"), uploadOverallStatus = document.getElementById("upload-overall-status"), chatPanel = document.getElementById("chat-panel"), chatMessages = document.getElementById("chat-messages"), chatForm = document.getElementById("chat-form"), chatInput = document.getElementById("chat-input"), chatSubmitBtn = document.getElementById("chat-submit-btn");

// ADD/EDIT MODAL DOM ELEMENTS
const nodeAddModal = document.getElementById("node-add-modal"), cancelNodeAddBtn = document.getElementById("cancel-node-add-btn"), nodeAddForm = document.getElementById("node-add-form"), modalAddNodeLabel = document.getElementById("modal-add-node-label");
const edgeAddModal = document.getElementById("edge-add-modal"), cancelEdgeAddBtn = document.getElementById("cancel-edge-add-btn"), edgeAddForm = document.getElementById("edge-add-form"), modalAddEdgeLabel = document.getElementById("modal-add-edge-label");
const nodeEditModal = document.getElementById("node-edit-modal"), closeNodeEditModalBtn = document.getElementById("close-node-edit-modal-btn"), cancelNodeEditBtn = document.getElementById("cancel-node-edit-btn"), modalNodeId = document.getElementById("modal-node-id"), modalNodeLabel = document.getElementById("modal-node-label"), modalNodeProperties = document.getElementById("modal-node-properties"), saveNodeChangesBtn = document.getElementById("save-node-changes-btn");
const edgeEditModal = document.getElementById("edge-edit-modal"), closeEdgeEditModalBtn = document.getElementById("close-edge-edit-modal-btn"), cancelEdgeEditBtn = document.getElementById("cancel-edge-edit-btn"), modalEdgeId = document.getElementById("modal-edge-id"), modalEdgeLabel = document.getElementById("modal-edge-label"), modalEdgeProperties = document.getElementById("modal-edge-properties"), saveEdgeChangesBtn = document.getElementById("save-edge-changes-btn");

// Settings & DB Modals
const settingsModal = document.getElementById("settings-modal"), closeSettingsModalBtn = document.getElementById("close-settings-modal-btn"), modalThemeToggle = document.getElementById("modal-theme-toggle"), settingPhysicsOnLoad = document.getElementById("setting-physics-on-load"), saveSettingsBtn = document.getElementById("save-settings-btn");
const databasesModal = document.getElementById("databases-modal"), closeDatabasesModalBtn = document.getElementById("close-databases-modal-btn"), databaseListContainer = document.getElementById("database-list-container"), createNewDbBtn = document.getElementById("create-new-db-btn");
const createDbModal = document.getElementById("create-db-modal"), closeCreateDbModalBtn = document.getElementById("close-create-db-modal-btn"), cancelCreateDbBtn = document.getElementById("cancel-create-db-btn"), createDbForm = document.getElementById("create-db-form"), createDbNameInput = document.getElementById("create-db-name-input"), createDbError = document.getElementById("create-db-error");

// --- Global State & Vis.js Instances ---
let network;
let nodesDataSet = new vis.DataSet();
let edgesDataSet = new vis.DataSet();
let appSettings = { theme: 'light', physicsOnLoad: true, editModeEnabled: false };
let isEditingText = false;
let currentLayoutMethod = "default";
let socket = null;
let sessionId = null;
const markdownConverter = new showdown.Converter();

// --- Helper Functions ---
function showUserStatus(element, message, type = "success", duration = 4000) { if (!element) return; let iconClass = "fa-check-circle text-green-500"; if (type === "error") iconClass = "fa-times-circle text-red-500"; else if (type === "info") iconClass = "fa-info-circle text-blue-500"; else if (type === "warning") iconClass = "fa-exclamation-triangle text-yellow-500"; element.innerHTML = `<i class="fas ${iconClass} mr-2"></i><span>${message}</span>`; if (duration > 0) { setTimeout(() => { if(element.innerHTML.includes(message)) element.innerHTML = ""; }, duration); } }
async function apiRequest(endpoint, method = 'GET', body = null, isFormData = false) { const options = { method }; if (!isFormData && body) { options.headers = { 'Content-Type': 'application/json' }; options.body = JSON.stringify(body); } else if (isFormData && body) { options.body = body; } try { const response = await fetch(endpoint, options); if (!response.ok) { const errorData = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` })); throw new Error(errorData.detail || `Request failed`); } return response.status === 204 || response.status === 202 ? response : await response.json(); } catch (error) { console.error(`API call to ${endpoint} failed:`, error); throw error; } }

// --- Socket.IO Management ---
function setupSocketIO() { socket = io({ path: '/sio' }); socket.on('connect', () => { sessionId = socket.id; console.log("Socket.IO connected:", sessionId); }); socket.on('progress_update', handleProgressUpdate); socket.on('disconnect', () => { console.log("Socket.IO disconnected."); sessionId = null; }); socket.on('connect_error', (error) => { console.error("Socket.IO connection error:", error); }); }
function handleProgressUpdate(data) { const { task_id, progress, message } = data; let el = document.getElementById(`progress-${task_id}`); if (!el) { el = document.createElement('div'); el.id = `progress-${task_id}`; el.className = 'p-2 my-1 bg-gray-100 dark:bg-gray-700 rounded-md text-xs'; el.innerHTML = `<div class="font-semibold mb-1 truncate" id="progress-message-${task_id}"></div><div class="progress-bar-container"><div id="progress-bar-${task_id}" class="progress-bar" style="width: 0%;"></div></div>`; taskProgressContainer.appendChild(el); } document.getElementById(`progress-message-${task_id}`).textContent = message; document.getElementById(`progress-bar-${task_id}`).style.width = `${progress * 100}%`; if (progress >= 1.0) { const bar = document.getElementById(`progress-bar-${task_id}`); bar.classList.remove('bg-blue-600'); bar.classList.add(message.toLowerCase().includes("error") ? 'bg-red-600' : 'bg-green-600'); setTimeout(() => el.remove(), 8000); } }

// --- Modal Management ---
function openModal(modal) { modal?.classList.remove("hidden"); }
function closeModal(modal) { modal?.classList.add("hidden"); }

// --- Theme & Settings Management ---
function applyTheme(theme) { appSettings.theme = theme; if (theme === 'dark') { document.documentElement.classList.add('dark'); themeIconSun?.classList.add('hidden'); themeIconMoon?.classList.remove('hidden'); } else { document.documentElement.classList.remove('dark'); themeIconSun?.classList.remove('hidden'); themeIconMoon?.classList.add('hidden'); } if(modalThemeToggle) modalThemeToggle.checked = theme === 'dark'; updateVisThemeOptions(); }
function loadAppSettings() { const s = localStorage.getItem('graphExplorerSettings'); if (s) { try { appSettings = { ...appSettings, ...JSON.parse(s) }; } catch (e) { console.error("Bad settings:", e); } } applyTheme(appSettings.theme); if(settingPhysicsOnLoad) settingPhysicsOnLoad.checked = appSettings.physicsOnLoad; if(editModeToggle) { editModeToggle.checked = appSettings.editModeEnabled; editModeToggle.dispatchEvent(new Event('change')); } updatePhysicsButtonText(); }
function saveAppSettings() { appSettings.physicsOnLoad = settingPhysicsOnLoad.checked; localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings)); if (network) network.setOptions({ physics: { enabled: appSettings.physicsOnLoad } }); updatePhysicsButtonText(); }
function updatePhysicsButtonText() { if(physicsBtnText) physicsBtnText.textContent = appSettings.physicsOnLoad ? "Stop Physics" : "Start Physics"; }
themeToggleBtn?.addEventListener("click", () => applyTheme(appSettings.theme === 'dark' ? 'light' : 'dark'));
modalThemeToggle?.addEventListener("change", (e) => applyTheme(e.target.checked ? 'dark' : 'dark'));
settingsBtn?.addEventListener("click", () => openModal(settingsModal));
closeSettingsModalBtn?.addEventListener("click", () => closeModal(settingsModal));
saveSettingsBtn?.addEventListener("click", () => { saveAppSettings(); closeModal(settingsModal); });

// --- Database & Upload Modals Logic (largely unchanged) ---
databasesBtn?.addEventListener("click", () => { loadAndDisplayDatabases(); openModal(databasesModal); });
closeDatabasesModalBtn?.addEventListener("click", () => closeModal(databasesModal));
async function loadAndDisplayDatabases() { try { const dbs = await apiRequest(API_ENDPOINT_DATABASES); databaseListContainer.innerHTML = ''; dbs.forEach(db => { const div = document.createElement('div'); div.className = `p-3 rounded-md flex items-center ${db.is_active ? 'bg-blue-100 dark:bg-blue-900' : 'bg-gray-100 dark:bg-gray-700'}`; div.innerHTML = `<div><p class="font-semibold dark:text-gray-200">${db.name}${db.is_active ? '<span class="text-green-500 text-sm ml-2">(Active)</span>' : ''}</p></div><div class="ml-auto space-x-2">${!db.is_active ? `<button data-db-name="${db.name}" class="activate-db-btn text-sm bg-green-500 text-white px-3 py-1 rounded">Activate</button>` : ''}<button data-db-name="${db.name}" class="delete-db-btn text-sm bg-red-600 text-white px-3 py-1 rounded" ${dbs.length<=1?'disabled':''}><i class="fas fa-trash"></i></button></div>`; databaseListContainer.appendChild(div); }); document.querySelectorAll('.activate-db-btn').forEach(b => b.addEventListener('click', handleActivateDatabase)); document.querySelectorAll('.delete-db-btn').forEach(b => b.addEventListener('click', handleDeleteDatabase)); } catch (e) { databaseListContainer.innerHTML = `<p class="text-red-500">Error: ${e.message}</p>`; } }
async function handleActivateDatabase(e) { if (confirm(`Activate "${e.target.dataset.dbName}" & reload?`)) { try { await apiRequest(API_ENDPOINT_DATABASE_ACTION(e.target.dataset.dbName, 'activate'), 'PUT'); window.location.reload(); } catch (err) { alert(`Failed: ${err.message}`); } } }
async function handleDeleteDatabase(e) { if (confirm(`Delete config for "${e.target.closest('button').dataset.dbName}"?`)) { try { await apiRequest(API_ENDPOINT_DATABASE_ACTION(e.target.closest('button').dataset.dbName, 'delete'), 'DELETE'); loadAndDisplayDatabases(); } catch (err) { alert(`Failed: ${err.message}`); } } }
createNewDbBtn?.addEventListener('click', () => { createDbForm.reset(); createDbError.textContent = ''; openModal(createDbModal); });
closeCreateDbModalBtn?.addEventListener('click', () => closeModal(createDbModal)); cancelCreateDbBtn?.addEventListener('click', () => closeModal(createDbModal));
createDbForm?.addEventListener('submit', async (e) => { e.preventDefault(); const name = createDbNameInput.value.trim(); if (!name) return; try { await apiRequest(API_ENDPOINT_DATABASES, 'POST', { name }); closeModal(createDbModal); loadAndDisplayDatabases(); } catch (err) { createDbError.textContent = `Error: ${err.message}`; } });
uploadDocumentBtn?.addEventListener("click", () => { fileInput.value = ''; fileListPreview.innerHTML = ''; uploadSubmitBtn.disabled = true; openModal(uploadModal); });
closeUploadModalBtn?.addEventListener("click", () => closeModal(uploadModal));
fileInput?.addEventListener("change", () => { fileListPreview.innerHTML = ''; if (fileInput.files.length) { Array.from(fileInput.files).forEach(f => { fileListPreview.innerHTML += `<div class="text-xs p-1 truncate">${f.name}</div>`; }); uploadSubmitBtn.disabled = false; } else { uploadSubmitBtn.disabled = true; } });
uploadForm?.addEventListener("submit", async (e) => { e.preventDefault(); if (!fileInput.files.length || !sessionId) { showUserStatus(uploadOverallStatus, "Not connected", "error"); return; } uploadSubmitBtn.disabled = true; const g = extractionGuidanceInput.value.trim(); for (const f of fileInput.files) { const fd = new FormData(); fd.append("file", f); fd.append("guidance", g); fd.append("sid", sessionId); try { const r = await apiRequest(API_ENDPOINT_UPLOAD, "POST", fd, true); const d = await r.json(); showUserStatus(uploadOverallStatus, `Submitted '${f.name}'.`, "info", 6000); } catch (err) { showUserStatus(uploadOverallStatus, `Failed '${f.name}': ${err.message}`, "error", 0); } } setTimeout(fetchGraphData, 3000); uploadSubmitBtn.disabled = false; });

// --- Graph Data Fetching & Rendering ---
async function fetchGraphData() { graphLoadingOverlay.classList.remove('hidden'); showUserStatus(graphActionStatus, "Loading graph...", "info", 0); try { const data = await apiRequest(API_ENDPOINT_GRAPH_DATA); nodesDataSet.clear(); edgesDataSet.clear(); nodesDataSet.add(data.nodes.map(n => ({ ...n, original_label: n.label }))); edgesDataSet.add(data.edges); if (!network) renderGraph(); else network.setData({ nodes: nodesDataSet, edges: edgesDataSet }); applyCurrentLayout(); showUserStatus(graphActionStatus, "Graph loaded.", "success"); } catch (error) { showUserStatus(graphActionStatus, `Error loading graph: ${error.message}`, "error"); } finally { graphLoadingOverlay.classList.add('hidden'); } }
function getVisThemeColors() { const isDark = appSettings.theme === 'dark'; return { nodeFont: isDark ? '#E5E7EB' : '#1F2937', edgeFont: isDark ? '#D1D5DB' : '#374151', edgeStroke: isDark ? '#1F2937' : '#FFFFFF', nodeBorder: isDark ? '#60A5FA' : '#2563EB', nodeBg: isDark ? '#3B82F6' : '#93C5FD', nodeHighlightBorder: isDark ? '#93C5FD' : '#1D4ED8', nodeHighlightBg: isDark ? '#60A5FA' : '#BFDBFE', edgeColor: isDark ? '#3B82F6' : '#60A5FA' }; }

function renderGraph() {
    if (!graphContainer) return;
    graphContainer.innerHTML = ""; 
    const c = getVisThemeColors();
    const options = {
        nodes: { shape: "dot", size: 15, font: { size: 13, color: c.nodeFont }, borderWidth: 2, color: { border: c.nodeBorder, background: c.nodeBg, highlight: { border: c.nodeHighlightBorder, background: c.nodeHighlightBg }}},
        edges: { color: {color: c.edgeColor, highlight: c.nodeHighlightBorder }, width: 1.5, smooth: { type: 'continuous' }, arrows: { to: { enabled: true, scaleFactor: 0.6 }}, font: { size: 10, color: c.edgeFont, strokeWidth: 2, strokeColor: c.edgeStroke, align: 'middle' }},
        interaction: { tooltipDelay: 150, navigationButtons: false, keyboard: true, hover: true },
        physics: { enabled: appSettings.physicsOnLoad, solver: 'barnesHut', barnesHut: { gravitationalConstant: -25000, centralGravity: 0.05, springLength: 110 }, stabilization: { iterations: 300, fit: true }},
        layout: { hierarchical: { enabled: false } },
        manipulation: {
            enabled: false,
            // Use custom modals instead of prompts for better UX and to avoid blocking
            addNode: (nodeData, callback) => {
                // Pre-fill position from where the user clicked
                nodeData.x = nodeData.x;
                nodeData.y = nodeData.y;
                openNodeAddModal(nodeData, callback);
            },
            editNode: (nodeData, callback) => {
                openNodeEditModal(nodeData, callback);
            },
            addEdge: (edgeData, callback) => {
                openEdgeAddModal(edgeData, callback);
            },
            editEdge: {
                editWithoutDrag: (edgeData, callback) => {
                    openEdgeEditModal(edgeData, callback);
                }
            },
            deleteNode: async (data, callback) => {
                if (confirm(`Delete ${data.nodes.length} node(s)?`)) {
                    try {
                        for (const id of data.nodes) await apiRequest(`${API_ENDPOINT_NODE}${id}`, 'DELETE');
                        callback(data);
                    } catch (e) { alert(`Failed to delete: ${e}`); callback(null); }
                } else callback(null);
            },
            deleteEdge: async (data, callback) => {
                if (confirm(`Delete ${data.edges.length} edge(s)?`)) {
                    try {
                        for (const id of data.edges) await apiRequest(`${API_ENDPOINT_EDGE}${id}`, 'DELETE');
                        callback(data);
                    } catch (e) { alert(`Failed to delete: ${e}`); callback(null); }
                } else callback(null);
            },
        }
    };
    network = new vis.Network(graphContainer, { nodes: nodesDataSet, edges: edgesDataSet }, options);
    network.on("click", updateSelectionInfoPanel);
    network.on("doubleClick", (p) => { if (editModeToggle.checked) { if (p.nodes.length) openNodeEditModal(nodesDataSet.get(p.nodes[0]), (d) => nodesDataSet.update(d)); else if (p.edges.length) openEdgeEditModal(edgesDataSet.get(p.edges[0]), (d) => edgesDataSet.update(d)); } });
}
function updateVisThemeOptions() { if (!network) return; const c = getVisThemeColors(); network.setOptions({ nodes: { font: { color: c.nodeFont }, color: { border: c.nodeBorder, background: c.nodeBg, highlight: { border: c.nodeHighlightBorder, background: c.nodeHighlightBg }}}, edges: { color: {color: c.edgeColor, highlight: c.nodeHighlightBorder }, font: { color: c.edgeFont, strokeColor: c.edgeStroke }} }); }
function updateSelectionInfoPanel(params) {
    nodeInfoPanel.style.display = "none"; edgeInfoPanel.style.display = "none";
    noSelectionMessage.style.display = "block"; editSelectedNodeBtn.disabled = true; editSelectedEdgeBtn.disabled = true; expandNeighborsBtn.disabled = true;
    const { nodes, edges } = params;
    if (nodes.length > 0) {
        const node = nodesDataSet.get(nodes[0]);
        if (node) {
            nodeIDDisplay.textContent = node.id; nodeLabelDisplay.textContent = node.original_label || node.label || "N/A";
            nodePropertiesDisplay.textContent = JSON.stringify(node.properties || {}, null, 2);
            nodeInfoPanel.style.display = "block"; noSelectionMessage.style.display = "none";
            editSelectedNodeBtn.disabled = !editModeToggle.checked; editSelectedNodeBtn.dataset.nodeId = node.id;
            expandNeighborsBtn.disabled = false; expandNeighborsBtn.dataset.nodeId = node.id;
        }
    } else if (edges.length > 0) {
        const edge = edgesDataSet.get(edges[0]);
        if (edge) {
            edgeIDDisplay.textContent = edge.id; edgeFromDisplay.textContent = String(edge.from); edgeToDisplay.textContent = String(edge.to);
            edgeTypeDisplay.textContent = edge.label || "No Type"; edgePropertiesDisplay.textContent = JSON.stringify(edge.properties || {}, null, 2);
            edgeInfoPanel.style.display = "block"; noSelectionMessage.style.display = "none";
            editSelectedEdgeBtn.disabled = !editModeToggle.checked; editSelectedEdgeBtn.dataset.edgeId = edge.id;
        }
    }
}

// --- Add/Edit Modal Logic ---
let visJsAddNodeCallback, visJsAddEdgeCallback;
let tempNodeData, tempEdgeData;

function openNodeAddModal(nodeData, callback) {
    visJsAddNodeCallback = callback;
    tempNodeData = nodeData;
    modalAddNodeLabel.value = 'NewLabel';
    openModal(nodeAddModal);
    modalAddNodeLabel.focus();
}
function openEdgeAddModal(edgeData, callback) {
    visJsAddEdgeCallback = callback;
    tempEdgeData = edgeData;
    modalAddEdgeLabel.value = '';
    openModal(edgeAddModal);
    modalAddEdgeLabel.focus();
}

nodeAddForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const label = modalAddNodeLabel.value.trim();
    if (!label) return;
    try {
        const newNode = await apiRequest(API_ENDPOINT_NODE, 'POST', { label, properties: {} });
        // Combine server data with original position data
        tempNodeData.id = newNode.id;
        tempNodeData.label = newNode.label;
        tempNodeData.properties = newNode.properties;
        tempNodeData.original_label = newNode.label;
        if(visJsAddNodeCallback) visJsAddNodeCallback(tempNodeData);
    } catch (error) {
        alert(`Failed to add node: ${error}`);
        if(visJsAddNodeCallback) visJsAddNodeCallback(null);
    } finally {
        closeModal(nodeAddModal);
    }
});
edgeAddForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const label = modalAddEdgeLabel.value.trim();
    const payload = { from_node_id: tempEdgeData.from, to_node_id: tempEdgeData.to, label: label || null, properties: {} };
    try {
        const newEdge = await apiRequest(API_ENDPOINT_EDGE, 'POST', payload);
        // Combine server data with original from/to data
        tempEdgeData.id = newEdge.id;
        tempEdgeData.label = newEdge.label;
        tempEdgeData.properties = newEdge.properties;
        if(visJsAddEdgeCallback) visJsAddEdgeCallback(tempEdgeData);
    } catch (error) {
        alert(`Failed to add edge: ${error}`);
        if(visJsAddEdgeCallback) visJsAddEdgeCallback(null);
    } finally {
        closeModal(edgeAddModal);
    }
});

cancelNodeAddBtn?.addEventListener('click', () => { if(visJsAddNodeCallback) visJsAddNodeCallback(null); closeModal(nodeAddModal); });
cancelEdgeAddBtn?.addEventListener('click', () => { if(visJsAddEdgeCallback) visJsAddEdgeCallback(null); closeModal(edgeAddModal); });

// --- Edit Modal Logic (updated for full functionality) ---
function setupEditModal(modal, closeBtn, cancelBtn, saveBtn, saveFn) { closeBtn?.addEventListener("click", () => closeModal(modal)); cancelBtn?.addEventListener("click", () => closeModal(modal)); saveBtn?.addEventListener("click", saveFn); }
let saveNodeChanges = async () => {}; let saveEdgeChanges = async () => {};
setupEditModal(nodeEditModal, closeNodeEditModalBtn, cancelNodeEditBtn, saveNodeChangesBtn, () => saveNodeChanges());
setupEditModal(edgeEditModal, closeEdgeEditModalBtn, cancelEdgeEditBtn, saveEdgeChangesBtn, () => saveEdgeChanges());
function openNodeEditModal(nodeData, visJsCallback) { modalNodeId.value = nodeData.id; modalNodeLabel.value = nodeData.original_label || nodeData.label || ''; modalNodeProperties.value = JSON.stringify(nodeData.properties || {}, null, 2); saveNodeChanges = async () => { try { const payload = { label: modalNodeLabel.value.trim(), properties: JSON.parse(modalNodeProperties.value) }; const updatedNode = await apiRequest(`${API_ENDPOINT_NODE}${nodeData.id}`, 'PUT', payload); const updatedVisData = { ...updatedNode, original_label: updatedNode.label, id: updatedNode.id }; if (visJsCallback) visJsCallback(updatedVisData); else nodesDataSet.update(updatedVisData); closeModal(nodeEditModal); updateSelectionInfoPanel({ nodes: [nodeData.id], edges: [] }); } catch (error) { alert(`Failed to update node: ${error}`); if (visJsCallback) visJsCallback(null); } }; openModal(nodeEditModal); }
function openEdgeEditModal(edgeData, visJsCallback) { modalEdgeId.value = edgeData.id; modalEdgeLabel.value = edgeData.label || ''; modalEdgeProperties.value = JSON.stringify(edgeData.properties || {}, null, 2); saveEdgeChanges = async () => { try { const payload = { label: modalEdgeLabel.value.trim() || null, properties: JSON.parse(modalEdgeProperties.value) }; const updatedEdge = await apiRequest(`${API_ENDPOINT_EDGE}${edgeData.id}`, 'PUT', payload); const updatedVisData = { ...updatedEdge, from: updatedEdge.from_node_id, to: updatedEdge.to_node_id, id: updatedEdge.id }; if (visJsCallback) visJsCallback(updatedVisData); else edgesDataSet.update(updatedVisData); closeModal(edgeEditModal); updateSelectionInfoPanel({ nodes: [], edges: [edgeData.id] }); } catch (error) { alert(`Failed to update edge: ${error}`); if (visJsCallback) visJsCallback(null); } }; openModal(edgeEditModal); }
editSelectedNodeBtn?.addEventListener("click", () => { const id = editSelectedNodeBtn.dataset.nodeId; if (id && nodesDataSet.get(id)) openNodeEditModal(nodesDataSet.get(id), (d) => nodesDataSet.update(d)); });
editSelectedEdgeBtn?.addEventListener("click", () => { const id = editSelectedEdgeBtn.dataset.edgeId; if (id && edgesDataSet.get(id)) openEdgeEditModal(edgesDataSet.get(id), (d) => edgesDataSet.update(d)); });

// --- Graph Controls & Search ---
editModeToggle?.addEventListener("change", (e) => { if (!network) return; const c = e.target.checked; appSettings.editModeEnabled = c; localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings)); network.setOptions({ manipulation: { enabled: c } }); editModeHint.textContent = c ? "Toolbar active. Double-click to edit." : "Allows direct graph manipulation."; updateSelectionInfoPanel(network.getSelection()); });
togglePhysicsBtn?.addEventListener("click", () => { if (!network) return; appSettings.physicsOnLoad = !appSettings.physicsOnLoad; network.setOptions({ physics: { enabled: appSettings.physicsOnLoad } }); updatePhysicsButtonText(); if(settingPhysicsOnLoad) settingPhysicsOnLoad.checked = appSettings.physicsOnLoad; localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings)); });
fitGraphBtn?.addEventListener("click", () => network?.fit());
applyLayoutBtn?.addEventListener("click", () => { currentLayoutMethod = layoutSelect.value; applyCurrentLayout(); });
fuseEntitiesBtn?.addEventListener("click", async () => { if (!sessionId) { showUserStatus(graphActionStatus, "Not connected", "error"); return; } if (!confirm("Fuse entities?")) return; showUserStatus(graphActionStatus, "Starting fusion...", "info", 0); try { const r = await apiRequest(API_ENDPOINT_GRAPH_FUSE, 'POST', { sid_body: { sid: sessionId } }); const d = await r.json(); showUserStatus(graphActionStatus, `Fusion started (Task: ${d.task_id}).`, "success", 10000); } catch (e) { showUserStatus(graphActionStatus, `Failed: ${e.message}`, "error"); } });
findPathForm?.addEventListener("submit", async (e) => { e.preventDefault(); const start = parseInt(startNodeInput.value, 10), end = parseInt(endNodeInput.value, 10); if (isNaN(start) || isNaN(end)) { showUserStatus(graphActionStatus, "Valid IDs required", "warning"); return; } showUserStatus(graphActionStatus, `Finding path...`, "info", 0); try { const d = await apiRequest(API_ENDPOINT_PATH, 'POST', { start_node_id: start, end_node_id: end }); network.unselectAll(); network.selectNodes(d.nodes.map(n => n.node_id)); network.selectEdges(d.relationships.map(r => r.relationship_id)); network.fit({ nodes: d.nodes.map(n => n.node_id), animation: true }); showUserStatus(graphActionStatus, `Path found.`, "success"); } catch (err) { showUserStatus(graphActionStatus, `Path failed: ${err.message}`, "error"); } });
expandNeighborsBtn?.addEventListener("click", async (e) => { const id = parseInt(e.currentTarget.dataset.nodeId, 10); if (isNaN(id)) return; showUserStatus(graphActionStatus, `Expanding...`, "info", 0); try { const d = await apiRequest(API_ENDPOINT_NEIGHBORS(id)); const newNodes = d.nodes.map(n => ({...n, id: n.node_id, label: n.label, group: n.label, properties: n.properties, original_label: n.label })); const newEdges = d.relationships.map(r => ({...r, id: r.relationship_id, from: r.source_node_id, to: r.target_node_id, label: r.type, properties: r.properties})); nodesDataSet.update(newNodes); edgesDataSet.update(newEdges); showUserStatus(graphActionStatus, `Added ${newNodes.length} neighbors.`, "success"); } catch (err) { showUserStatus(graphActionStatus, `Failed: ${err.message}`, "error"); } });
function applyCurrentLayout() { if (!network) return; const text = layoutSelect.options[layoutSelect.selectedIndex].text; showUserStatus(graphActionStatus, `Applying ${text}...`, "info", 0); let opts = { hierarchical: { enabled: false } }, phys = true; switch (currentLayoutMethod) { case "hierarchicalRepulsion": opts.hierarchical = { enabled: true, direction: "UD", sortMethod: "directed" }; network.setOptions({ physics: { solver: 'hierarchicalRepulsion' }}); break; case "hierarchicalDirected": opts.hierarchical = { enabled: true, direction: "LR", sortMethod: "directed" }; phys = false; break; default: network.setOptions({ physics: { solver: 'barnesHut' }}); break; } network.setOptions({ layout: opts, physics: { enabled: phys } }); if (!phys) network.stabilize(); appSettings.physicsOnLoad = phys; updatePhysicsButtonText(); if(settingPhysicsOnLoad) settingPhysicsOnLoad.checked = phys; localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings)); setTimeout(() => { network.fit(); showUserStatus(graphActionStatus, `${text} applied.`, "success"); }, 100); }
graphSearchForm?.addEventListener("submit", async(e) => { e.preventDefault(); const q = graphSearchInput.value.trim(); if (!q) return; showUserStatus(graphActionStatus, "Searching...", "info", 0); clearSearch(false); try { const res = await apiRequest(`${API_ENDPOINT_GRAPH_SEARCH}?${new URLSearchParams({ q })}`); showUserStatus(graphActionStatus, `Found ${res.nodes.length} nodes.`, "success"); populateSearchResults(res); searchResultsContainer.classList.remove("hidden"); } catch (err) { showUserStatus(graphActionStatus, `Search failed: ${err.message}`, "error"); } });
function populateSearchResults(res) { searchResultsList.innerHTML = ""; res.nodes.forEach(n => { const b = document.createElement("button"); b.className = "w-full text-left p-1.5 text-xs rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"; b.dataset.type = "node"; b.dataset.id = n.id; b.innerHTML = `<i class="fas fa-circle mr-2 text-blue-400"></i> ${n.label}`; b.addEventListener("click", handleSearchResultClick); searchResultsList.appendChild(b); }); }
function handleSearchResultClick(e) { const { type, id } = e.currentTarget.dataset; if (type === 'node') { network.focus(id, { animation: true, scale: 1.2 }); network.selectNodes([id]); } }
function clearSearch(clearInput = true) { if (clearInput) graphSearchInput.value = ""; searchResultsList.innerHTML = ""; searchResultsContainer.classList.add("hidden"); if (network) network.unselectAll(); if(clearInput) showUserStatus(graphActionStatus, "", "info", 1); }
graphSearchInput?.addEventListener("input", () => { clearSearchBtn.classList.toggle("hidden", !graphSearchInput.value); if (!graphSearchInput.value) clearSearch(true); });
clearSearchBtn?.addEventListener("click", () => clearSearch(true));

// --- RAG Chat Logic ---
chatForm?.addEventListener("submit", handleChatSubmit);
function addChatMessage(message, sender, isThinking = false) { const d = document.createElement("div"); d.className = `chat-message ${sender === 'user' ? 'chat-user' : 'chat-ai'}`; if (isThinking) { d.innerHTML = `<i class="fas fa-spinner fa-spin"></i>`; d.id = "thinking-indicator"; } else if (sender === 'ai') { d.innerHTML = markdownConverter.makeHtml(message); } else { d.textContent = message; } chatMessages.appendChild(d); chatMessages.scrollTop = chatMessages.scrollHeight; }
async function handleChatSubmit(e) { e.preventDefault(); const q = chatInput.value.trim(); if (!q) return; addChatMessage(q, 'user'); chatInput.value = ''; addChatMessage('', 'ai', true); try { const r = await apiRequest(API_ENDPOINT_CHAT, 'POST', { query: q }); document.getElementById("thinking-indicator")?.remove(); addChatMessage(r.answer, 'ai'); } catch (err) { document.getElementById("thinking-indicator")?.remove(); addChatMessage(`Error: ${err.message}`, 'ai'); } }

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => { loadAppSettings(); setupSocketIO(); fetchGraphData(); });