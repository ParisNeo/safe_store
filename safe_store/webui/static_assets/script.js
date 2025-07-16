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
// Navbar & General UI
const themeToggleBtn = document.getElementById("theme-toggle-btn");
const themeIconSun = document.getElementById("theme-icon-sun");
const themeIconMoon = document.getElementById("theme-icon-moon");
const settingsBtn = document.getElementById("settings-btn");
const uploadDocumentBtn = document.getElementById("upload-document-btn");
const databasesBtn = document.getElementById("databases-btn");
const graphLoadingOverlay = document.getElementById("graph-loading-overlay");

// Graph & Sidebar
const graphContainer = document.getElementById("graph-container");
const graphActionStatus = document.getElementById("graph-action-status");
const taskProgressContainer = document.getElementById("task-progress-container");
const nodeInfoPanel = document.getElementById("node-info-panel");
const edgeInfoPanel = document.getElementById("edge-info-panel");
const nodeIDDisplay = document.getElementById("node-id-display");
const nodeLabelDisplay = document.getElementById("node-label-display");
const nodePropertiesDisplay = document.getElementById("node-properties-display");
const editSelectedNodeBtn = document.getElementById("edit-selected-node-btn");
const expandNeighborsBtn = document.getElementById("expand-neighbors-btn");
const edgeIDDisplay = document.getElementById("edge-id-display");
const edgeFromDisplay = document.getElementById("edge-from-display");
const edgeToDisplay = document.getElementById("edge-to-display");
const edgeTypeDisplay = document.getElementById("edge-type-display");
const edgePropertiesDisplay = document.getElementById("edge-properties-display");
const editSelectedEdgeBtn = document.getElementById("edit-selected-edge-btn");
const noSelectionMessage = document.getElementById("no-selection");

// Graph Controls (Sidebar)
const graphSearchForm = document.getElementById("graph-search-form");
const graphSearchInput = document.getElementById("graph-search-input");
const clearSearchBtn = document.getElementById("clear-search-btn");
const searchResultsContainer = document.getElementById("search-results-container");
const searchResultsList = document.getElementById("search-results-list");
const editModeToggle = document.getElementById("edit-mode-toggle");
const editModeHint = document.getElementById("edit-mode-hint");
const togglePhysicsBtn = document.getElementById("toggle-physics-btn");
const physicsBtnText = document.getElementById("physics-btn-text");
const fuseEntitiesBtn = document.getElementById("fuse-entities-btn");
const fitGraphBtn = document.getElementById("fit-graph-btn");
const layoutSelect = document.getElementById("layout-select");
const applyLayoutBtn = document.getElementById("apply-layout-btn");
const findPathForm = document.getElementById("find-path-form");
const startNodeInput = document.getElementById("start-node-input");
const endNodeInput = document.getElementById("end-node-input");

// Upload Modal
const uploadModal = document.getElementById("upload-modal");
const closeUploadModalBtn = document.getElementById("close-upload-modal-btn");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const extractionGuidanceInput = document.getElementById("extraction-guidance-input");
const fileListPreview = document.getElementById("file-list-preview");
const uploadSubmitBtn = document.getElementById("upload-submit-btn");
const uploadProgressArea = document.getElementById("upload-progress-area");
const uploadOverallStatus = document.getElementById("upload-overall-status");

// Chat Panel
const chatPanel = document.getElementById("chat-panel");
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSubmitBtn = document.getElementById("chat-submit-btn");

// Node/Edge Edit Modals
const nodeEditModal = document.getElementById("node-edit-modal");
const closeNodeEditModalBtn = document.getElementById("close-node-edit-modal-btn");
const cancelNodeEditBtn = document.getElementById("cancel-node-edit-btn");
const modalNodeId = document.getElementById("modal-node-id");
const modalNodeLabel = document.getElementById("modal-node-label");
const modalNodeProperties = document.getElementById("modal-node-properties");
const saveNodeChangesBtn = document.getElementById("save-node-changes-btn");
const edgeEditModal = document.getElementById("edge-edit-modal");
const closeEdgeEditModalBtn = document.getElementById("close-edge-edit-modal-btn");
const cancelEdgeEditBtn = document.getElementById("cancel-edge-edit-btn");
const modalEdgeId = document.getElementById("modal-edge-id");
const modalEdgeLabel = document.getElementById("modal-edge-label");
const modalEdgeProperties = document.getElementById("modal-edge-properties");
const saveEdgeChangesBtn = document.getElementById("save-edge-changes-btn");

// Appearance Settings Modal
const settingsModal = document.getElementById("settings-modal");
const closeSettingsModalBtn = document.getElementById("close-settings-modal-btn");
const modalThemeToggle = document.getElementById("modal-theme-toggle");
const settingPhysicsOnLoad = document.getElementById("setting-physics-on-load");
const saveSettingsBtn = document.getElementById("save-settings-btn");

// Database Management Modals
const databasesModal = document.getElementById("databases-modal");
const closeDatabasesModalBtn = document.getElementById("close-databases-modal-btn");
const databaseListContainer = document.getElementById("database-list-container");
const createNewDbBtn = document.getElementById("create-new-db-btn");
const createDbModal = document.getElementById("create-db-modal");
const closeCreateDbModalBtn = document.getElementById("close-create-db-modal-btn");
const cancelCreateDbBtn = document.getElementById("cancel-create-db-btn");
const createDbForm = document.getElementById("create-db-form");
const createDbNameInput = document.getElementById("create-db-name-input");
const createDbError = document.getElementById("create-db-error");

// --- Global State & Vis.js Instances ---
let network;
let nodesDataSet = new vis.DataSet();
let edgesDataSet = new vis.DataSet();
let appSettings = { theme: 'light', physicsOnLoad: true, editModeEnabled: false };
let isEditingText = false;
let currentLayoutMethod = "default";
let webSocket = null;
let clientId = `webui-${Math.random().toString(36).substr(2, 9)}`;
const markdownConverter = new showdown.Converter();


// --- Helper Functions ---
function showUserStatus(element, message, type = "success", duration = 4000) {
    if (!element) return;
    let iconClass = "fa-check-circle text-green-500";
    if (type === "error") iconClass = "fa-times-circle text-red-500";
    else if (type === "info") iconClass = "fa-info-circle text-blue-500";
    else if (type === "warning") iconClass = "fa-exclamation-triangle text-yellow-500";

    element.innerHTML = `<i class="fas ${iconClass} mr-2"></i><span class="align-middle">${message}</span>`;
    
    if (duration > 0) {
        setTimeout(() => { if(element.innerHTML.includes(message)) element.innerHTML = ""; }, duration);
    }
}

async function apiRequest(endpoint, method = 'GET', body = null, isFormData = false) {
    const options = { method };
    if (!isFormData && body) {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify(body);
    } else if (isFormData && body) {
        options.body = body;
    }

    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
            throw new Error(errorData.detail || `Request failed with status ${response.status}`);
        }
        return response.status === 204 || response.status === 202 ? response : await response.json();
    } catch (error) {
        console.error(`API call to ${endpoint} failed:`, error);
        throw error;
    }
}

// --- WebSocket Management ---
function setupWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/progress/${clientId}`;

    webSocket = new WebSocket(wsUrl);

    webSocket.onopen = () => console.log("WebSocket connection established.");
    webSocket.onmessage = (event) => handleProgressUpdate(JSON.parse(event.data));
    webSocket.onclose = () => {
        console.log("WebSocket connection closed. Attempting to reconnect...");
        setTimeout(setupWebSocket, 3000); // Reconnect after 3 seconds
    };
    webSocket.onerror = (error) => console.error("WebSocket error:", error);
}

function handleProgressUpdate(data) {
    const { task_id, progress, message } = data;
    let progressElement = document.getElementById(`progress-${task_id}`);

    if (!progressElement) {
        progressElement = document.createElement('div');
        progressElement.id = `progress-${task_id}`;
        progressElement.className = 'p-2 my-1 bg-gray-100 dark:bg-gray-700 rounded-md text-xs';
        progressElement.innerHTML = `
            <div class="font-semibold mb-1 truncate" id="progress-message-${task_id}"></div>
            <div class="progress-bar-container">
                <div id="progress-bar-${task_id}" class="progress-bar" style="width: 0%;"></div>
            </div>
        `;
        taskProgressContainer.appendChild(progressElement);
    }

    const messageElement = document.getElementById(`progress-message-${task_id}`);
    const barElement = document.getElementById(`progress-bar-${task_id}`);

    messageElement.textContent = message;
    barElement.style.width = `${progress * 100}%`;

    if (progress >= 1.0) {
        // Change bar color on completion
        if (message.toLowerCase().includes("error")) {
             barElement.classList.remove('bg-blue-600');
             barElement.classList.add('bg-red-600');
        } else {
             barElement.classList.remove('bg-blue-600');
             barElement.classList.add('bg-green-600');
        }
        // Remove after a delay
        setTimeout(() => progressElement.remove(), 8000);
    }
}

// --- Modal Management ---
function openModal(modalElement) { modalElement?.classList.remove("hidden"); }
function closeModal(modalElement) { modalElement?.classList.add("hidden"); }

// --- Theme Management ---
function applyTheme(theme) {
    appSettings.theme = theme;
    if (theme === 'dark') {
        document.documentElement.classList.add('dark');
        themeIconSun?.classList.add('hidden');
        themeIconMoon?.classList.remove('hidden');
        if (modalThemeToggle) modalThemeToggle.checked = true;
    } else {
        document.documentElement.classList.remove('dark');
        themeIconSun?.classList.remove('hidden');
        themeIconMoon?.classList.add('hidden');
        if (modalThemeToggle) modalThemeToggle.checked = false;
    }
    updateVisThemeOptions();
}
themeToggleBtn?.addEventListener("click", () => applyTheme(appSettings.theme === 'dark' ? 'light' : 'dark'));
modalThemeToggle?.addEventListener("change", (e) => applyTheme(e.target.checked ? 'dark' : 'light'));

// --- Settings Management ---
function loadAppSettings() {
    const storedSettings = localStorage.getItem('graphExplorerSettings');
    if (storedSettings) {
        try {
            appSettings = { ...appSettings, ...JSON.parse(storedSettings) };
        } catch (e) { console.error("Failed to parse stored settings:", e); }
    }
    applyTheme(appSettings.theme);
    if(settingPhysicsOnLoad) settingPhysicsOnLoad.checked = appSettings.physicsOnLoad;
    if(editModeToggle) {
        editModeToggle.checked = appSettings.editModeEnabled;
        editModeToggle.dispatchEvent(new Event('change')); 
    }    
    updatePhysicsButtonText();
}

function saveAppSettings() {
    appSettings.physicsOnLoad = settingPhysicsOnLoad.checked;
    localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings));
    if (network) network.setOptions({ physics: { enabled: appSettings.physicsOnLoad } });
    updatePhysicsButtonText();
}
settingsBtn?.addEventListener("click", () => openModal(settingsModal));
closeSettingsModalBtn?.addEventListener("click", () => closeModal(settingsModal));
saveSettingsBtn?.addEventListener("click", () => {
    saveAppSettings();
    closeModal(settingsModal);
});
function updatePhysicsButtonText() {
    if(physicsBtnText) physicsBtnText.textContent = appSettings.physicsOnLoad ? "Stop Physics" : "Start Physics";
}

// --- Database Management ---
databasesBtn?.addEventListener("click", () => {
    loadAndDisplayDatabases();
    openModal(databasesModal);
});
closeDatabasesModalBtn?.addEventListener("click", () => closeModal(databasesModal));

async function loadAndDisplayDatabases() {
    try {
        const databases = await apiRequest(API_ENDPOINT_DATABASES);
        databaseListContainer.innerHTML = '';
        if (databases.length === 0) {
            databaseListContainer.innerHTML = `<p class="text-xs italic dark:text-gray-400">No databases configured.</p>`;
            return;
        }
        databases.forEach(db => {
            const div = document.createElement('div');
            div.className = `p-3 rounded-md flex justify-between items-center ${db.is_active ? 'bg-blue-100 dark:bg-blue-900 border border-blue-400' : 'bg-gray-100 dark:bg-gray-700'}`;
            div.innerHTML = `
                <div class="flex-grow">
                    <p class="font-semibold dark:text-gray-200 text-base">${db.name} ${db.is_active ? '<span class="text-green-500 text-sm">(Active)</span>' : ''}</p>
                    <p class="text-xs text-gray-500 dark:text-gray-400 font-mono" title="${db.db_file}">DB: ${db.db_file}</p>
                    <p class="text-xs text-gray-500 dark:text-gray-400 font-mono" title="${db.doc_dir}">Docs: ${db.doc_dir}</p>
                </div>
                <div class="space-x-2 flex-shrink-0 ml-4">
                    ${!db.is_active ? `<button data-db-name="${db.name}" class="activate-db-btn px-3 py-1 bg-green-500 hover:bg-green-600 text-white rounded text-sm font-medium">Activate</button>` : ''}
                    <button data-db-name="${db.name}" class="delete-db-btn px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-sm font-medium" ${databases.length <= 1 ? 'disabled title="Cannot delete the last database"' : ''}><i class="fas fa-trash"></i></button>
                </div>
            `;
            databaseListContainer.appendChild(div);
        });

        document.querySelectorAll('.activate-db-btn').forEach(btn => btn.addEventListener('click', handleActivateDatabase));
        document.querySelectorAll('.delete-db-btn').forEach(btn => btn.addEventListener('click', handleDeleteDatabase));
    } catch (error) {
        databaseListContainer.innerHTML = `<p class="text-sm italic text-red-500">Error loading databases: ${error.message}</p>`;
    }
}

async function handleActivateDatabase(event) {
    const dbName = event.target.dataset.dbName;
    if (confirm(`Activate "${dbName}"? The application will reload.`)) {
        try {
            await apiRequest(API_ENDPOINT_DATABASE_ACTION(dbName, 'activate'), 'PUT');
            graphContainer.innerHTML = `<div class="p-4 text-lg flex items-center justify-center h-full"><i class="fas fa-sync fa-spin mr-3"></i> Reloading with new database...</div>`;
            setTimeout(() => window.location.reload(), 500);
        } catch (error) {
            alert(`Failed to activate database: ${error.message}`);
        }
    }
}

async function handleDeleteDatabase(event) {
    const dbName = event.target.closest('button').dataset.dbName;
    if (confirm(`Are you sure you want to delete the configuration for "${dbName}"? This does NOT delete the files on disk.`)) {
        try {
            await apiRequest(API_ENDPOINT_DATABASE_ACTION(dbName, 'delete'), 'DELETE');
            loadAndDisplayDatabases();
        } catch (error) {
            alert(`Failed to delete database: ${error.message}`);
        }
    }
}

createNewDbBtn?.addEventListener('click', () => {
    createDbForm.reset();
    createDbError.textContent = '';
    openModal(createDbModal);
});
closeCreateDbModalBtn?.addEventListener('click', () => closeModal(createDbModal));
cancelCreateDbBtn?.addEventListener('click', () => closeModal(createDbModal));

createDbForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const name = createDbNameInput.value.trim();
    if (!name) {
        createDbError.textContent = "Database name is required.";
        return;
    }
    
    try {
        await apiRequest(API_ENDPOINT_DATABASES, 'POST', { name });
        closeModal(createDbModal);
        loadAndDisplayDatabases();
    } catch (error) {
        createDbError.textContent = `Error: ${error.message}`;
    }
});

// --- Upload Modal Logic ---
uploadDocumentBtn?.addEventListener("click", () => {
    fileInput.value = ''; fileListPreview.innerHTML = ''; uploadSubmitBtn.disabled = true;
    openModal(uploadModal);
});
closeUploadModalBtn?.addEventListener("click", () => closeModal(uploadModal));
fileInput?.addEventListener("change", () => {
    fileListPreview.innerHTML = ''; uploadProgressArea.innerHTML = ''; uploadOverallStatus.innerHTML = ''; 
    if (fileInput.files.length > 0) {
        Array.from(fileInput.files).forEach(file => {
            const fileDiv = document.createElement('div');
            fileDiv.className = 'text-xs p-1 bg-gray-100 dark:bg-gray-700 rounded truncate';
            fileDiv.textContent = file.name;
            fileListPreview.appendChild(fileDiv);
        });
        uploadSubmitBtn.disabled = false;
    } else {
        uploadSubmitBtn.disabled = true;
    }
});
uploadForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!fileInput.files.length) return;
    uploadSubmitBtn.disabled = true;
    uploadProgressArea.innerHTML = '';
    let successCount = 0, errorCount = 0;
    const guidance = extractionGuidanceInput.value.trim();

    for (let i = 0; i < fileInput.files.length; i++) {
        const file = fileInput.files[i];
        
        const formData = new FormData();
        formData.append("file", file);
        formData.append("guidance", guidance);
        formData.append("client_id", clientId);

        try {
            const response = await apiRequest(API_ENDPOINT_UPLOAD, "POST", formData, true);
            const data = await response.json(); // Wait for the 202 response to be parsed
            showUserStatus(uploadOverallStatus, `File '${file.name}' submitted for processing (Task ID: ${data.task_id}).`, "info", 6000);
            successCount++;
        } catch (error) {
            showUserStatus(uploadOverallStatus, `Failed to submit '${file.name}': ${error.message}`, "error", 0);
            errorCount++;
        }
    }
    
    if (errorCount === 0) {
        showUserStatus(uploadOverallStatus, "All files submitted. Processing in background.", "success", 5000);
    }
    
    // Refresh graph data after a short delay to allow backend processing to start
    setTimeout(fetchGraphData, 3000);
    uploadSubmitBtn.disabled = false;
});


// --- Node & Edge Edit Modal Logic ---
function setupEditModal(modal, closeBtn, cancelBtn, saveBtn, saveFn) {
    closeBtn?.addEventListener("click", () => closeModal(modal));
    cancelBtn?.addEventListener("click", () => closeModal(modal));
    saveBtn?.addEventListener("click", saveFn);
}
// Placeholder save functions, to be filled by open...Modal functions
let saveNodeChanges = async () => {};
let saveEdgeChanges = async () => {};
setupEditModal(nodeEditModal, closeNodeEditModalBtn, cancelNodeEditBtn, saveNodeChangesBtn, () => saveNodeChanges());
setupEditModal(edgeEditModal, closeEdgeEditModalBtn, cancelEdgeEditBtn, saveEdgeChangesBtn, () => saveEdgeChanges());

function openNodeEditModal(nodeData, visJsCallbackForToolbarEdit) {
    modalNodeId.value = nodeData.id;
    modalNodeLabel.value = nodeData.original_label || nodeData.label || '';
    modalNodeProperties.value = JSON.stringify(nodeData.properties || {}, null, 2);
    saveNodeChanges = async () => {
        const payload = { label: modalNodeLabel.value.trim(), properties: JSON.parse(modalNodeProperties.value) };
        try {
            await apiRequest(`${API_ENDPOINT_NODE}${nodeData.id}`, 'PUT', payload);
            const updatedNodeVisData = { ...nodeData, ...payload, original_label: payload.label };
            if (visJsCallbackForToolbarEdit) visJsCallbackForToolbarEdit(updatedNodeVisData);
            else nodesDataSet.update(updatedNodeVisData);
            closeModal(nodeEditModal);
            updateSelectionInfoPanel({ nodes: [nodeData.id], edges: [] });
        } catch (error) { alert(`Failed to update node: ${error.message}`); if (visJsCallbackForToolbarEdit) visJsCallbackForToolbarEdit(null); }
    };
    openModal(nodeEditModal);
}
function openEdgeEditModal(edgeData, visJsCallbackForToolbarEdit) {
    modalEdgeId.value = edgeData.id;
    modalEdgeLabel.value = edgeData.label || '';
    modalEdgeProperties.value = JSON.stringify(edgeData.properties || {}, null, 2);
    saveEdgeChanges = async () => {
        const payload = { label: modalEdgeLabel.value.trim() || null, properties: JSON.parse(modalEdgeProperties.value) };
         try {
            await apiRequest(`${API_ENDPOINT_EDGE}${edgeData.id}`, 'PUT', payload);
            const updatedEdgeVisData = { ...edgeData, ...payload };
            if (visJsCallbackForToolbarEdit) visJsCallbackForToolbarEdit(updatedEdgeVisData);
            else edgesDataSet.update(updatedEdgeVisData);
            closeModal(edgeEditModal);
            updateSelectionInfoPanel({ nodes: [], edges: [edgeData.id] });
        } catch (error) { alert(`Failed to update edge: ${error.message}`); if (visJsCallbackForToolbarEdit) visJsCallbackForToolbarEdit(null); }
    };
    openModal(edgeEditModal);
}
editSelectedNodeBtn?.addEventListener("click", () => {
    const nodeId = editSelectedNodeBtn.dataset.nodeId;
    if (nodeId && nodesDataSet.get(nodeId)) openNodeEditModal(nodesDataSet.get(nodeId));
});
editSelectedEdgeBtn?.addEventListener("click", () => {
    const edgeId = editSelectedEdgeBtn.dataset.edgeId;
    if (edgeId && edgesDataSet.get(edgeId)) openEdgeEditModal(edgesDataSet.get(edgeId));
});

// --- Graph Search ---
async function handleGraphSearch(event) {
    event.preventDefault(); if (!network) return;
    const query = graphSearchInput.value.trim();
    if (!query) { showUserStatus(graphActionStatus, "Please enter a search query.", "warning"); return; }
    
    showUserStatus(graphActionStatus, "Searching...", "info", 0);
    graphLoadingOverlay.classList.remove('hidden');
    clearSearch(false); // Clear previous results but not input
    
    try {
        const params = new URLSearchParams({ q: query });
        const results = await apiRequest(`${API_ENDPOINT_GRAPH_SEARCH}?${params.toString()}`);
        
        if (results.nodes.length === 0 && results.edges.length === 0) {
            showUserStatus(graphActionStatus, "No results found.", "info");
        } else {
            showUserStatus(graphActionStatus, `Found ${results.nodes.length} nodes and ${results.edges.length} edges.`, "success");
            populateSearchResults(results);
            searchResultsContainer.classList.remove("hidden");
        }
    } catch (error) {
        showUserStatus(graphActionStatus, `Search failed: ${error.message}`, "error");
    } finally {
        graphLoadingOverlay.classList.add('hidden');
    }
}

function populateSearchResults(results) {
    searchResultsList.innerHTML = "";
    results.nodes.forEach(node => {
        const item = document.createElement("button");
        item.className = "w-full text-left p-1.5 text-xs rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500";
        item.dataset.type = "node";
        item.dataset.id = node.id;
        item.innerHTML = `<i class="fas fa-circle mr-2 text-blue-400"></i> ${node.label.substring(0, 40)}${node.label.length > 40 ? '...' : ''}`;
        item.addEventListener("click", handleSearchResultClick);
        searchResultsList.appendChild(item);
    });
    results.edges.forEach(edge => {
        const item = document.createElement("button");
        item.className = "w-full text-left p-1.5 text-xs rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500";
        item.dataset.type = "edge";
        item.dataset.id = edge.id;
        item.innerHTML = `<i class="fas fa-long-arrow-alt-right mr-2 text-green-400"></i> Edge: ${edge.label || 'unlabeled'}`;
        item.addEventListener("click", handleSearchResultClick);
        searchResultsList.appendChild(item);
    });
}

function handleSearchResultClick(event) {
    const target = event.currentTarget;
    const { type, id } = target.dataset;
    
    if (type === 'node') {
        network.focus(id, { animation: true, scale: 1.2 });
        network.selectNodes([id]);
    } else if (type === 'edge') {
        const edge = edgesDataSet.get(id);
        if (edge) {
            network.fit({ nodes: [edge.from, edge.to], animation: true });
            network.selectEdges([id]);
        }
    }
}

function clearSearch(clearInput = true) {
    if (clearInput) graphSearchInput.value = "";
    searchResultsList.innerHTML = "";
    searchResultsContainer.classList.add("hidden");
    if (network) network.unselectAll();
    if(clearInput) showUserStatus(graphActionStatus, "", "info", 1);
}

graphSearchForm?.addEventListener("submit", handleGraphSearch);
graphSearchInput?.addEventListener("input", () => {
    clearSearchBtn.classList.toggle("hidden", !graphSearchInput.value);
    if (!graphSearchInput.value) clearSearch(true);
});
clearSearchBtn?.addEventListener("click", () => clearSearch(true));

// --- Graph Data Fetching & Rendering ---
async function fetchGraphData() {
    graphLoadingOverlay.classList.remove('hidden');
    showUserStatus(graphActionStatus, "Loading graph data...", "info", 0);
    try {
        const data = await apiRequest(API_ENDPOINT_GRAPH_DATA);
        nodesDataSet.clear();
        edgesDataSet.clear();
        nodesDataSet.add(data.nodes.map(n => ({ ...n, original_label: n.label })));
        edgesDataSet.add(data.edges);

        if (!network) renderGraph();
        else network.setData({ nodes: nodesDataSet, edges: edgesDataSet });
        
        applyCurrentLayout(); 
        showUserStatus(graphActionStatus, "Graph data loaded.", "success");
    } catch (error) {
        showUserStatus(graphActionStatus, `Error loading graph: ${error.message}`, "error");
        graphContainer.innerHTML = `<div class="p-4 text-red-600 dark:text-red-400">Error: ${error.message}</div>`;
    } finally {
        graphLoadingOverlay.classList.add('hidden');
    }
}

function getVisThemeColors() {
    const isDark = appSettings.theme === 'dark';
    return {
        nodeFontColor: isDark ? '#E5E7EB' : '#1F2937', edgeFontColor: isDark ? '#D1D5DB' : '#374151',
        edgeStrokeColor: isDark ? '#1F2937' : '#FFFFFF', nodeBorder: isDark ? '#60A5FA' : '#2563EB',
        nodeBg: isDark ? '#3B82F6' : '#93C5FD', nodeHighlightBorder: isDark ? '#93C5FD' : '#1D4ED8',
        nodeHighlightBg: isDark ? '#60A5FA' : '#BFDBFE', edgeColor: isDark ? '#3B82F6' : '#60A5FA'
    };
}

function renderGraph() {
    if (!graphContainer) return;
    graphContainer.innerHTML = ""; 
    const themeColors = getVisThemeColors();

    const options = {
        nodes: {
            shape: "dot", size: 15, font: { size: 13, color: themeColors.nodeFontColor },
            borderWidth: 2, borderWidthSelected: 3,
            color: { border: themeColors.nodeBorder, background: themeColors.nodeBg, 
                     highlight: { border: themeColors.nodeHighlightBorder, background: themeColors.nodeHighlightBg }},
            shadow: { enabled: true, color: 'rgba(0,0,0,0.15)', size: 6, x: 2, y: 2 },
        },
        edges: {
            color: {color: themeColors.edgeColor, highlight: themeColors.nodeHighlightBorder },
            width: 1.5, smooth: { type: 'continuous', roundness: 0.2 },
            arrows: { to: { enabled: true, scaleFactor: 0.6, type: 'arrow' } },
            font: { size: 10, color: themeColors.edgeFontColor, strokeWidth: 2, strokeColor: themeColors.edgeStrokeColor, align: 'middle' }
        },
        interaction: {
            tooltipDelay: 150, navigationButtons: false, keyboard: { enabled: true, bindToWindow: false },
            hover: true, dragNodes: true, dragView: true, zoomView: true
        },
        manipulation: { enabled: false },
        physics: {
            enabled: appSettings.physicsOnLoad, solver: 'barnesHut',
            barnesHut: { gravitationalConstant: -25000, centralGravity: 0.05, springLength: 110, springConstant: 0.02, damping: 0.09, avoidOverlap: 0.2 },
            stabilization: { iterations: 300, fit: true, updateInterval: 25 },
        },
        layout: { hierarchical: { enabled: false } }
    };
    network = new vis.Network(graphContainer, { nodes: nodesDataSet, edges: edgesDataSet }, options);
    network.on("click", updateSelectionInfoPanel);
    network.on("doubleClick", (params) => {
        if (editModeToggle.checked) {
            if (params.nodes.length > 0) openNodeEditModal(nodesDataSet.get(params.nodes[0]));
            else if (params.edges.length > 0) openEdgeEditModal(edgesDataSet.get(params.edges[0]));
        }
    });
}
function updateVisThemeOptions() {
    if (!network) return;
    const themeColors = getVisThemeColors();
    network.setOptions({
        nodes: { font: { color: themeColors.nodeFontColor }, color: { border: themeColors.nodeBorder, background: themeColors.nodeBg, highlight: { border: themeColors.nodeHighlightBorder, background: themeColors.nodeHighlightBg }}},
        edges: { color: {color: themeColors.edgeColor, highlight: themeColors.nodeHighlightBorder }, font: { color: themeColors.edgeFontColor, strokeColor: themeColors.edgeStrokeColor }}
    });
}

function updateSelectionInfoPanel(params) {
    nodeInfoPanel.style.display = "none"; edgeInfoPanel.style.display = "none";
    noSelectionMessage.style.display = "block"; 
    editSelectedNodeBtn.disabled = true; 
    editSelectedEdgeBtn.disabled = true;
    expandNeighborsBtn.disabled = true;


    const { nodes, edges } = params;
    if (nodes.length > 0) {
        const node = nodesDataSet.get(nodes[0]);
        if (node) {
            nodeIDDisplay.textContent = node.id; nodeLabelDisplay.textContent = node.original_label || node.label || "N/A";
            nodePropertiesDisplay.textContent = JSON.stringify(node.properties || {}, null, 2);
            nodeInfoPanel.style.display = "block"; noSelectionMessage.style.display = "none";
            editSelectedNodeBtn.disabled = !editModeToggle.checked; 
            editSelectedNodeBtn.dataset.nodeId = node.id;
            expandNeighborsBtn.disabled = false;
            expandNeighborsBtn.dataset.nodeId = node.id;

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

// --- Graph Controls (Sidebar) ---
editModeToggle?.addEventListener("change", (e) => {
    if (!network) return;
    const isChecked = e.target.checked;
    appSettings.editModeEnabled = isChecked;
    localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings));
    network.setOptions({ manipulation: { enabled: isChecked } });
    if (isChecked) network.enableEditMode(); else network.disableEditMode();
    editModeHint.textContent = isChecked ? "Edit tools active. Double-click to edit." : "Allows direct graph manipulation.";
    updateSelectionInfoPanel(network.getSelection()); 
});
togglePhysicsBtn?.addEventListener("click", () => {
    if (!network) return;
    appSettings.physicsOnLoad = !appSettings.physicsOnLoad;
    network.setOptions({ physics: { enabled: appSettings.physicsOnLoad } });
    updatePhysicsButtonText();
    if(settingPhysicsOnLoad) settingPhysicsOnLoad.checked = appSettings.physicsOnLoad;
    localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings));
});
fitGraphBtn?.addEventListener("click", () => network?.fit());
applyLayoutBtn?.addEventListener("click", () => {
    currentLayoutMethod = layoutSelect.value;
    applyCurrentLayout();
});
fuseEntitiesBtn?.addEventListener("click", async () => {
    if (!confirm("This will scan the entire graph for similar entities and try to merge them using an LLM. This may take some time. Continue?")) {
        return;
    }
    showUserStatus(graphActionStatus, "Starting entity fusion...", "info", 0);
    graphLoadingOverlay.classList.remove('hidden');
    try {
        const response = await apiRequest(API_ENDPOINT_GRAPH_FUSE, 'POST', {client_id: clientId});
        const data = await response.json();
        showUserStatus(graphActionStatus, `Fusion process started in background (Task ID: ${data.task_id}).`, "success", 10000);
        // Do not reload graph here, wait for completion message
    } catch (error) {
        showUserStatus(graphActionStatus, `Failed to start fusion: ${error.message}`, "error");
    } finally {
        graphLoadingOverlay.classList.add('hidden');
    }
});

findPathForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const startId = parseInt(startNodeInput.value, 10);
    const endId = parseInt(endNodeInput.value, 10);
    if (isNaN(startId) || isNaN(endId)) {
        showUserStatus(graphActionStatus, "Please enter valid Start and End Node IDs.", "warning");
        return;
    }
    showUserStatus(graphActionStatus, `Finding path from ${startId} to ${endId}...`, "info", 0);
    graphLoadingOverlay.classList.remove("hidden");
    try {
        const pathData = await apiRequest(API_ENDPOINT_PATH, 'POST', { start_node_id: startId, end_node_id: endId });
        network.unselectAll();
        const nodeIdsInPath = pathData.nodes.map(n => n.node_id);
        const edgeIdsInPath = pathData.relationships.map(r => r.relationship_id);
        network.selectNodes(nodeIdsInPath);
        network.selectEdges(edgeIdsInPath);
        network.fit({ nodes: nodeIdsInPath, animation: true });
        showUserStatus(graphActionStatus, `Path found with ${nodeIdsInPath.length} nodes.`, "success");
    } catch (error) {
        showUserStatus(graphActionStatus, `Pathfinding failed: ${error.message}`, "error");
    } finally {
        graphLoadingOverlay.classList.add("hidden");
    }
});

expandNeighborsBtn?.addEventListener("click", async (e) => {
    const nodeId = parseInt(e.currentTarget.dataset.nodeId, 10);
    if (isNaN(nodeId)) return;
    
    showUserStatus(graphActionStatus, `Expanding neighbors for node ${nodeId}...`, "info", 0);
    graphLoadingOverlay.classList.remove("hidden");
    try {
        const neighborData = await apiRequest(API_ENDPOINT_NEIGHBORS(nodeId));
        const newNodes = neighborData.nodes.map(n => ({...n, id: n.node_id, label: n.label, group: n.label, properties: n.properties, original_label: n.label }));
        const newEdges = neighborData.relationships.map(r => ({...r, id: r.relationship_id, from: r.source_node_id, to: r.target_node_id, label: r.type, properties: r.properties}));

        nodesDataSet.update(newNodes);
        edgesDataSet.update(newEdges);
        showUserStatus(graphActionStatus, `Added ${newNodes.length} neighbors for node ${nodeId}.`, "success");
    } catch (error) {
        showUserStatus(graphActionStatus, `Failed to expand neighbors: ${error.message}`, "error");
    } finally {
        graphLoadingOverlay.classList.add("hidden");
    }
});


function applyCurrentLayout() {
    if (!network) return;
    const selectedLayoutText = layoutSelect.options[layoutSelect.selectedIndex].text;
    showUserStatus(graphActionStatus, `Applying ${selectedLayoutText} layout...`, "info", 0);
    graphLoadingOverlay.classList.remove('hidden');

    let layoutOptions = { hierarchical: { enabled: false } };
    let physicsEnabledForLayout = true;
    switch (currentLayoutMethod) {
        case "hierarchicalRepulsion":
            layoutOptions.hierarchical = { enabled: true, direction: "UD", sortMethod: "directed", levelSeparation: 180, nodeSpacing: 130};
            network.setOptions({ physics: { solver: 'hierarchicalRepulsion' }});
            break;
        case "hierarchicalDirected":
            layoutOptions.hierarchical = { enabled: true, direction: "LR", sortMethod: "directed", levelSeparation: 220 };
            physicsEnabledForLayout = false;
            break;
        case "barnesHut":
        default:
            network.setOptions({ physics: { solver: 'barnesHut' }});
            break;
    }
    network.setOptions({ layout: layoutOptions, physics: { enabled: physicsEnabledForLayout } });
    if (!physicsEnabledForLayout) network.stabilize();
    
    appSettings.physicsOnLoad = physicsEnabledForLayout;
    updatePhysicsButtonText();
    if(settingPhysicsOnLoad) settingPhysicsOnLoad.checked = appSettings.physicsOnLoad;
    localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings));

    setTimeout(() => {
        network.fit();
        graphLoadingOverlay.classList.add('hidden');
        showUserStatus(graphActionStatus, `${selectedLayoutText} layout applied.`, "success");
    }, 100);
}

// --- RAG Chat Logic ---
chatForm?.addEventListener("submit", handleChatSubmit);

function addChatMessage(message, sender, isThinking = false) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `chat-message ${sender === 'user' ? 'chat-user' : 'chat-ai'}`;
    
    if (isThinking) {
        messageDiv.innerHTML = `<i class="fas fa-spinner fa-spin"></i>`;
        messageDiv.id = "thinking-indicator";
    } else if (sender === 'ai') {
        messageDiv.innerHTML = markdownConverter.makeHtml(message);
    } else {
        messageDiv.textContent = message;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function handleChatSubmit(event) {
    event.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;

    addChatMessage(query, 'user');
    chatInput.value = '';
    addChatMessage('', 'ai', true); // Show thinking indicator

    try {
        const response = await apiRequest(API_ENDPOINT_CHAT, 'POST', { query });
        document.getElementById("thinking-indicator")?.remove(); // Remove thinking indicator
        addChatMessage(response.answer, 'ai');
    } catch (error) {
        document.getElementById("thinking-indicator")?.remove();
        addChatMessage(`Error: ${error.message}`, 'ai');
    }
}


// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    loadAppSettings();
    setupWebSocket();
    fetchGraphData();
});