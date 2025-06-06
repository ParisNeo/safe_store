// webui/static_assets/script.js

// --- Configuration ---
const API_BASE_URL = ""; // Assuming API calls are relative to the current host
const API_ENDPOINT_UPLOAD = `${API_BASE_URL}/upload-file/`;
const API_ENDPOINT_GRAPH_DATA = `${API_BASE_URL}/graph-data/`;
const API_ENDPOINT_NODE = `${API_BASE_URL}/graph/node/`;
const API_ENDPOINT_EDGE = `${API_BASE_URL}/graph/edge/`;
// Placeholder for DB management endpoints
const API_ENDPOINT_DATABASES = `${API_BASE_URL}/api/databases`; // GET, POST
const API_ENDPOINT_DATABASE_ACTION = (dbId, action) => `${API_BASE_URL}/api/databases/${dbId}/${action}`; // activate, delete, etc.

const MAX_FILE_SIZE_MB = 10;

// --- DOM Elements ---
// Navbar & General UI
const themeToggleBtn = document.getElementById("theme-toggle-btn");
const themeIconSun = document.getElementById("theme-icon-sun");
const themeIconMoon = document.getElementById("theme-icon-moon");
const settingsBtn = document.getElementById("settings-btn");
const uploadDocumentBtn = document.getElementById("upload-document-btn");
const graphLoadingOverlay = document.getElementById("graph-loading-overlay");

// Graph & Sidebar
const graphContainer = document.getElementById("graph-container");
const graphActionStatus = document.getElementById("graph-action-status");
const nodeInfoPanel = document.getElementById("node-info-panel");
const edgeInfoPanel = document.getElementById("edge-info-panel");
const nodeIDDisplay = document.getElementById("node-id-display");
const nodeLabelDisplay = document.getElementById("node-label-display");
const nodePropertiesDisplay = document.getElementById("node-properties-display");
const editSelectedNodeBtn = document.getElementById("edit-selected-node-btn");
const edgeIDDisplay = document.getElementById("edge-id-display");
const edgeFromDisplay = document.getElementById("edge-from-display");
const edgeToDisplay = document.getElementById("edge-to-display");
const edgeTypeDisplay = document.getElementById("edge-type-display");
const edgePropertiesDisplay = document.getElementById("edge-properties-display");
const editSelectedEdgeBtn = document.getElementById("edit-selected-edge-btn");
const noSelectionMessage = document.getElementById("no-selection");

// Graph Controls (Sidebar)
const editModeToggle = document.getElementById("edit-mode-toggle");
const editModeHint = document.getElementById("edit-mode-hint");
const togglePhysicsBtn = document.getElementById("toggle-physics-btn");
const physicsBtnText = document.getElementById("physics-btn-text");
const fitGraphBtn = document.getElementById("fit-graph-btn");
const layoutSelect = document.getElementById("layout-select");
const applyLayoutBtn = document.getElementById("apply-layout-btn");

// Upload Modal
const uploadModal = document.getElementById("upload-modal");
const closeUploadModalBtn = document.getElementById("close-upload-modal-btn");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const fileListPreview = document.getElementById("file-list-preview");
const uploadSubmitBtn = document.getElementById("upload-submit-btn");
const uploadProgressArea = document.getElementById("upload-progress-area");
const uploadOverallStatus = document.getElementById("upload-overall-status");

// Node Edit Modal
const nodeEditModal = document.getElementById("node-edit-modal");
const closeNodeEditModalBtn = document.getElementById("close-node-edit-modal-btn");
const cancelNodeEditBtn = document.getElementById("cancel-node-edit-btn");
const modalNodeId = document.getElementById("modal-node-id");
const modalNodeLabel = document.getElementById("modal-node-label");
const modalNodeProperties = document.getElementById("modal-node-properties");
const saveNodeChangesBtn = document.getElementById("save-node-changes-btn");

// Edge Edit Modal
const edgeEditModal = document.getElementById("edge-edit-modal");
const closeEdgeEditModalBtn = document.getElementById("close-edge-edit-modal-btn");
const cancelEdgeEditBtn = document.getElementById("cancel-edge-edit-btn");
const modalEdgeId = document.getElementById("modal-edge-id");
const modalEdgeFrom = document.getElementById("modal-edge-from");
const modalEdgeTo = document.getElementById("modal-edge-to");
const modalEdgeLabel = document.getElementById("modal-edge-label");
const modalEdgeProperties = document.getElementById("modal-edge-properties");
const saveEdgeChangesBtn = document.getElementById("save-edge-changes-btn");

// Settings Modal
const settingsModal = document.getElementById("settings-modal");
const closeSettingsModalBtn = document.getElementById("close-settings-modal-btn");
const modalThemeToggle = document.getElementById("modal-theme-toggle");
const settingPhysicsOnLoad = document.getElementById("setting-physics-on-load");
const saveSettingsBtn = document.getElementById("save-settings-btn");
const activeDatabaseNameDisplay = document.getElementById("active-database-name");
const databaseListContainer = document.getElementById("database-list-container");
const createNewDbBtn = document.getElementById("create-new-db-btn");
const loadExistingDbBtn = document.getElementById("load-existing-db-btn"); // Placeholder functionality

// DB Config Modal
const databaseConfigModal = document.getElementById("database-config-modal");
const dbConfigModalTitle = document.getElementById("db-config-modal-title");
const closeDbConfigModalBtn = document.getElementById("close-db-config-modal-btn");
const dbConfigForm = document.getElementById("db-config-form");
const dbNameInput = document.getElementById("db-name-input");
const dbPathInput = document.getElementById("db-path-input");
const cancelDbConfigBtn = document.getElementById("cancel-db-config-btn");
const saveDbConfigBtn = document.getElementById("save-db-config-btn");
let currentDbConfigTarget = null; // 'new' or db_id for editing

// --- Global State & Vis.js Instances ---
let network;
let nodesDataSet = new vis.DataSet();
let edgesDataSet = new vis.DataSet();
let appSettings = {
    theme: 'light',
    physicsOnLoad: true,
    editModeEnabled: false,
    activeDatabaseId: 'default' // Placeholder
};
let isEditingText = false; // For keyboard interaction fix
let currentLayoutMethod = "default"; // Store the current layout method

// --- Helper Functions ---
function showUserStatus(element, message, type = "success", duration = 4000, isModal = false) {
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

function parseJsonProperties(jsonString, defaultValue = {}) {
    try {
        return jsonString.trim() ? JSON.parse(jsonString) : defaultValue;
    } catch (e) {
        console.error("Invalid JSON:", e, "Input:", jsonString);
        return null;
    }
}

async function apiRequest(endpoint, method = 'GET', body = null, isFormData = false) {
    const options = { method };
    if (!isFormData && body) {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify(body);
    } else if (isFormData && body) {
        options.body = body; // FormData sets its own Content-Type
    }

    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
            throw new Error(errorData.detail || `Request failed with status ${response.status}`);
        }
        return response.status === 204 ? null : await response.json();
    } catch (error) {
        console.error(`API call to ${endpoint} failed:`, error);
        throw error;
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
    // If Vis.js options depend on theme, update them here:
    // updateVisThemeOptions(); 
}
themeToggleBtn?.addEventListener("click", () => applyTheme(appSettings.theme === 'dark' ? 'light' : 'dark'));
modalThemeToggle?.addEventListener("change", (e) => applyTheme(e.target.checked ? 'dark' : 'light'));

// --- Settings Management ---
function loadAppSettings() {
    const storedSettings = localStorage.getItem('graphExplorerSettings');
    if (storedSettings) {
        try {
            const parsed = JSON.parse(storedSettings);
            // Merge, prioritizing stored settings but keeping defaults for new keys
            appSettings = { ...appSettings, ...parsed };
        } catch (e) { console.error("Failed to parse stored settings:", e); }
    }
    applyTheme(appSettings.theme);
    if(settingPhysicsOnLoad) settingPhysicsOnLoad.checked = appSettings.physicsOnLoad;
    if(editModeToggle) {
        editModeToggle.checked = appSettings.editModeEnabled;
        // Trigger the change event handler to apply the state to Vis.js and UI hints
        editModeToggle.dispatchEvent(new Event('change')); 
    }    
    updatePhysicsButtonText();
    // Load databases and set active one (placeholder)
    loadAndDisplayDatabases(); 
}

function saveAppSettings() {
    appSettings.physicsOnLoad = settingPhysicsOnLoad.checked;
    // Theme is saved implicitly by applyTheme
    localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings));
    showUserStatus(graphActionStatus, "Settings applied.", "success", 2000);
    closeModal(settingsModal);
    if (network) network.setOptions({ physics: { enabled: appSettings.physicsOnLoad } });
    updatePhysicsButtonText();
}
settingsBtn?.addEventListener("click", () => openModal(settingsModal));
closeSettingsModalBtn?.addEventListener("click", () => closeModal(settingsModal));
saveSettingsBtn?.addEventListener("click", saveAppSettings);

function updatePhysicsButtonText() {
    if(physicsBtnText) physicsBtnText.textContent = appSettings.physicsOnLoad ? "Stop Physics" : "Start Physics";
}

// --- Database Management (Frontend Placeholders) ---
let availableDatabases = []; // Store full DB objects from backend {id: "...", name: "...", path: "...", active: true/false}

async function loadAndDisplayDatabases() {
    try {
        // availableDatabases = await apiRequest(API_ENDPOINT_DATABASES); // UNCOMMENT WHEN BACKEND READY
        // For now, mock data:
        availableDatabases = [
            { id: 'default', name: 'Default Main DB', path: 'webui_store.db', active: true },
            { id: 'project_alpha', name: 'Project Alpha', path: 'alpha_store.db', active: false },
        ];
        appSettings.activeDatabaseId = availableDatabases.find(db => db.active)?.id || 'default';
        renderDatabaseList();
        if(activeDatabaseNameDisplay) activeDatabaseNameDisplay.textContent = availableDatabases.find(db => db.id === appSettings.activeDatabaseId)?.name || 'N/A';
    } catch (error) {
        console.error("Failed to load databases:", error);
        if(databaseListContainer) databaseListContainer.innerHTML = `<p class="text-xs italic text-red-500">Error loading databases: ${error.message}</p>`;
    }
}

function renderDatabaseList() {
    if (!databaseListContainer) return;
    databaseListContainer.innerHTML = ''; // Clear
    if (availableDatabases.length === 0) {
        databaseListContainer.innerHTML = `<p class="text-xs italic dark:text-gray-400">No databases configured.</p>`;
        return;
    }
    availableDatabases.forEach(db => {
        const div = document.createElement('div');
        div.className = `p-1.5 rounded-md flex justify-between items-center text-xs ${db.id === appSettings.activeDatabaseId ? 'bg-blue-100 dark:bg-blue-800' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`;
        div.innerHTML = `
            <span class="font-medium dark:text-gray-200">${db.name} (${db.path})</span>
            <div class="space-x-1">
                ${db.id !== appSettings.activeDatabaseId ? `<button data-db-id="${db.id}" class="activate-db-btn px-2 py-0.5 bg-green-500 hover:bg-green-600 text-white rounded text-xs">Activate</button>` : '<span class="px-2 py-0.5 text-green-600 dark:text-green-400 text-xs font-semibold">Active</span>'}
                <button data-db-id="${db.id}" data-db-name="${db.name}" data-db-path="${db.path}" class="edit-db-btn px-1.5 py-0.5 bg-yellow-500 hover:bg-yellow-600 text-white rounded text-xs"><i class="fas fa-edit"></i></button>
                <button data-db-id="${db.id}" class="delete-db-btn px-1.5 py-0.5 bg-red-500 hover:bg-red-600 text-white rounded text-xs"><i class="fas fa-trash"></i></button>
            </div>
        `;
        databaseListContainer.appendChild(div);
    });

    // Add event listeners
    document.querySelectorAll('.activate-db-btn').forEach(btn => btn.addEventListener('click', handleActivateDatabase));
    document.querySelectorAll('.edit-db-btn').forEach(btn => btn.addEventListener('click', handleEditDatabase));
    document.querySelectorAll('.delete-db-btn').forEach(btn => btn.addEventListener('click', handleDeleteDatabase));
}

async function handleActivateDatabase(event) {
    const dbId = event.target.dataset.dbId;
    showUserStatus(graphActionStatus, `Activating database ${dbId}...`, "info", 0);
    try {
        // await apiRequest(API_ENDPOINT_DATABASE_ACTION(dbId, 'activate'), 'PUT'); // UNCOMMENT
        appSettings.activeDatabaseId = dbId; // Mock activation
        availableDatabases.forEach(db => db.active = (db.id === dbId)); // Mock active state update
        saveAppSettings(); // Save new active DB
        renderDatabaseList();
        if(activeDatabaseNameDisplay) activeDatabaseNameDisplay.textContent = availableDatabases.find(db => db.id === dbId)?.name || 'N/A';
        showUserStatus(graphActionStatus, `Database ${dbId} activated. Reloading graph...`, "success");
        fetchGraphData(); // Reload graph for the new DB
    } catch (error) {
        showUserStatus(graphActionStatus, `Failed to activate ${dbId}: ${error.message}`, "error");
    }
}

function handleEditDatabase(event) {
    const btn = event.target.closest('button');
    currentDbConfigTarget = btn.dataset.dbId;
    dbConfigModalTitle.textContent = "Edit Database";
    dbNameInput.value = btn.dataset.dbName;
    dbPathInput.value = btn.dataset.dbPath;
    openModal(databaseConfigModal);
}

async function handleDeleteDatabase(event) {
    const dbId = event.target.closest('button').dataset.dbId;
    const db = availableDatabases.find(d => d.id === dbId);
    if (confirm(`Are you sure you want to delete database "${db?.name}"? This might also delete the .db file.`)) {
        showUserStatus(graphActionStatus, `Deleting database ${dbId}...`, "info", 0);
        try {
            // await apiRequest(API_ENDPOINT_DATABASE_ACTION(dbId, 'delete'), 'DELETE'); // UNCOMMENT
            availableDatabases = availableDatabases.filter(d => d.id !== dbId); // Mock deletion
            if(appSettings.activeDatabaseId === dbId && availableDatabases.length > 0) {
                appSettings.activeDatabaseId = availableDatabases[0].id; // Activate first available
                availableDatabases[0].active = true;
            } else if (availableDatabases.length === 0) {
                appSettings.activeDatabaseId = null;
            }
            saveAppSettings();
            renderDatabaseList();
             if(activeDatabaseNameDisplay) activeDatabaseNameDisplay.textContent = availableDatabases.find(d => d.active)?.name || 'N/A';
            showUserStatus(graphActionStatus, `Database ${dbId} deleted.`, "success");
        } catch (error) {
            showUserStatus(graphActionStatus, `Failed to delete ${dbId}: ${error.message}`, "error");
        }
    }
}

createNewDbBtn?.addEventListener('click', () => {
    currentDbConfigTarget = 'new';
    dbConfigModalTitle.textContent = "Create New Database";
    dbConfigForm.reset();
    openModal(databaseConfigModal);
});

closeDbConfigModalBtn?.addEventListener('click', () => closeModal(databaseConfigModal));
cancelDbConfigBtn?.addEventListener('click', () => closeModal(databaseConfigModal));

dbConfigForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const name = dbNameInput.value.trim();
    const path = dbPathInput.value.trim();
    if (!name || !path) { alert("Name and Path are required."); return; }

    const payload = { name, path };
    let endpoint = API_ENDPOINT_DATABASES;
    let method = 'POST';

    if (currentDbConfigTarget !== 'new') { // Editing existing
        endpoint = API_ENDPOINT_DATABASE_ACTION(currentDbConfigTarget, 'update'); // Or just /api/databases/{id}
        method = 'PUT';
    }
    
    showUserStatus(graphActionStatus, `Saving database configuration...`, "info", 0);
    try {
        // const result = await apiRequest(endpoint, method, payload); // UNCOMMENT
        // Mock success:
        const result = { id: currentDbConfigTarget === 'new' ? `db_${Date.now()}` : currentDbConfigTarget, ...payload };
        if (currentDbConfigTarget === 'new') {
            availableDatabases.push({ ...result, active: false });
        } else {
            const index = availableDatabases.findIndex(db => db.id === result.id);
            if (index > -1) availableDatabases[index] = { ...availableDatabases[index], ...result };
        }
        loadAndDisplayDatabases(); // Re-render list
        showUserStatus(graphActionStatus, `Database "${name}" configuration saved.`, "success");
        closeModal(databaseConfigModal);
    } catch (error) {
        showUserStatus(graphActionStatus, `Failed to save DB config: ${error.message}`, "error");
    }
});


// --- Upload Modal & Multi-File Logic ---
uploadDocumentBtn?.addEventListener("click", () => {
    fileInput.value = ''; // Clear previous selection
    fileListPreview.innerHTML = '';
    uploadSubmitBtn.disabled = true;
    openModal(uploadModal);
});
closeUploadModalBtn?.addEventListener("click", () => closeModal(uploadModal));

fileInput?.addEventListener("change", () => {
    fileListPreview.innerHTML = ''; // Clear previous list
    uploadProgressArea.innerHTML = ''; // Clear old progress bars
    uploadOverallStatus.innerHTML = ''; 
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
    const files = fileInput.files;
    if (!files.length) {
        showUserStatus(uploadOverallStatus, "Please select files to upload.", "warning");
        return;
    }

    uploadSubmitBtn.disabled = true;
    uploadProgressArea.innerHTML = ''; // Clear previous progress

    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const fileId = `file-progress-${i}`;
        const progressDiv = document.createElement('div');
        progressDiv.id = fileId;
        progressDiv.className = 'text-xs p-1.5 rounded-md bg-gray-100 dark:bg-gray-700 flex justify-between items-center';
        progressDiv.innerHTML = `<span>${file.name}</span><span class="status-icon"><i class="fas fa-hourglass-start text-blue-500"></i> Queued</span>`;
        uploadProgressArea.appendChild(progressDiv);

        if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
            document.querySelector(`#${fileId} .status-icon`).innerHTML = `<i class="fas fa-times-circle text-red-500"></i> Size Exceeded`;
            errorCount++;
            continue;
        }

        document.querySelector(`#${fileId} .status-icon`).innerHTML = `<i class="fas fa-spinner fa-spin text-blue-500"></i> Uploading...`;
        const formData = new FormData();
        formData.append("file", file);

        try {
            await apiRequest(API_ENDPOINT_UPLOAD, "POST", formData, true);
            document.querySelector(`#${fileId} .status-icon`).innerHTML = `<i class="fas fa-check-circle text-green-500"></i> Processed`;
            successCount++;
        } catch (error) {
            document.querySelector(`#${fileId} .status-icon`).innerHTML = `<i class="fas fa-exclamation-circle text-red-500"></i> Error: ${error.message.substring(0,30)}`;
            errorCount++;
        }
    }

    let overallMessage = `${successCount} file(s) processed successfully.`;
    if (errorCount > 0) overallMessage += ` ${errorCount} file(s) failed.`;
    showUserStatus(uploadOverallStatus, overallMessage, errorCount > 0 ? (successCount > 0 ? "warning" : "error") : "success", 0);
    
    if (successCount > 0) fetchGraphData(); // Refresh graph if any file succeeded
    uploadSubmitBtn.disabled = false;
    // Do not close modal automatically, let user see results.
});


// --- Node & Edge Edit Modals Logic (largely same as before, ensure Tailwind compatibility) ---
// (Functions openNodeEditModal, openEdgeEditModal, and their save handlers are similar to previous JS, adapted for new modal IDs and structure if needed)
// Minor change: using apiRequest helper
editSelectedNodeBtn?.addEventListener("click", () => {
    const nodeId = editSelectedNodeBtn.dataset.nodeId;
    if (nodeId && nodesDataSet.get(nodeId)) openNodeEditModal(nodesDataSet.get(nodeId));
});
closeNodeEditModalBtn?.addEventListener("click", () => closeModal(nodeEditModal));
cancelNodeEditBtn?.addEventListener("click", () => closeModal(nodeEditModal));
saveNodeChangesBtn?.addEventListener("click", async () => {
    const nodeId = modalNodeId.value;
    const newLabel = modalNodeLabel.value.trim();
    const newPropsStr = modalNodeProperties.value;
    if (!newLabel) { alert("Node label cannot be empty."); return; }
    const newProps = parseJsonProperties(newPropsStr);
    if (newProps === null) { alert("Invalid JSON in properties."); return; }

    showUserStatus(graphActionStatus, "Saving node...", "info", 0);
    try {
        await apiRequest(`${API_ENDPOINT_NODE}${nodeId}`, 'PUT', { label: newLabel, properties: newProps });
        nodesDataSet.update({ id: nodeId, label: newLabel, original_label: newLabel, properties: newProps }); // Update local dataset
        showUserStatus(graphActionStatus, `Node "${newLabel}" updated.`, "success");
        closeModal(nodeEditModal);
        updateSelectionInfoPanel({ nodes: [nodeId], edges: [] });
    } catch (error) {
        showUserStatus(graphActionStatus, `Failed to update node: ${error.message}`, "error");
    }
});

editSelectedEdgeBtn?.addEventListener("click", () => {
    const edgeId = editSelectedEdgeBtn.dataset.edgeId;
    if (edgeId && edgesDataSet.get(edgeId)) openEdgeEditModal(edgesDataSet.get(edgeId));
});
closeEdgeEditModalBtn?.addEventListener("click", () => closeModal(edgeEditModal));
cancelEdgeEditBtn?.addEventListener("click", () => closeModal(edgeEditModal));
saveEdgeChangesBtn?.addEventListener("click", async () => {
    const edgeId = modalEdgeId.value;
    const newLabel = modalEdgeLabel.value.trim();
    const newPropsStr = modalEdgeProperties.value;
    const newProps = parseJsonProperties(newPropsStr);
    if (newProps === null) { alert("Invalid JSON in properties."); return; }

    showUserStatus(graphActionStatus, "Saving edge...", "info", 0);
    try {
        await apiRequest(`${API_ENDPOINT_EDGE}${edgeId}`, 'PUT', { label: newLabel || null, properties: newProps });
        edgesDataSet.update({ id: edgeId, label: newLabel, properties: newProps });
        showUserStatus(graphActionStatus, `Edge updated.`, "success");
        closeModal(edgeEditModal);
        updateSelectionInfoPanel({ nodes: [], edges: [edgeId] });
    } catch (error) {
        showUserStatus(graphActionStatus, `Failed to update edge: ${error.message}`, "error");
    }
});

function openNodeEditModal(nodeData, visJsCallbackForToolbarEdit) {
    modalNodeId.value = nodeData.id;
    modalNodeLabel.value = nodeData.original_label || nodeData.label || '';
    modalNodeProperties.value = JSON.stringify(nodeData.properties || {}, null, 2);
    saveNodeChangesBtn.onclick = async () => { // Rewire onclick for this specific instance
        const nodeId = modalNodeId.value;
        const newLabel = modalNodeLabel.value.trim();
        const newPropsStr = modalNodeProperties.value;
        if (!newLabel) { alert("Node label cannot be empty."); return; }
        const newProps = parseJsonProperties(newPropsStr);
        if (newProps === null) { alert("Invalid JSON in properties."); return; }
        
        showUserStatus(graphActionStatus, "Saving node...", "info", 0);
        try {
            await apiRequest(`${API_ENDPOINT_NODE}${nodeId}`, 'PUT', { label: newLabel, properties: newProps });
            const updatedNodeVisData = { id: nodeId, label: newLabel, original_label: newLabel, properties: newProps };
            if (visJsCallbackForToolbarEdit) visJsCallbackForToolbarEdit(updatedNodeVisData); // For Vis toolbar
            else nodesDataSet.update(updatedNodeVisData); // For sidebar button edit
            
            showUserStatus(graphActionStatus, `Node "${newLabel}" updated.`, "success");
            closeModal(nodeEditModal);
            updateSelectionInfoPanel({ nodes: [nodeId], edges: [] });
        } catch (error) {
            showUserStatus(graphActionStatus, `Failed to update node: ${error.message}`, "error");
            if (visJsCallbackForToolbarEdit) visJsCallbackForToolbarEdit(null); // Signal error to Vis
        }
    };
    openModal(nodeEditModal);
}

function openEdgeEditModal(edgeData, visJsCallbackForToolbarEdit) {
    modalEdgeId.value = edgeData.id;
    modalEdgeFrom.value = edgeData.from;
    modalEdgeTo.value = edgeData.to;
    modalEdgeLabel.value = edgeData.label || '';
    modalEdgeProperties.value = JSON.stringify(edgeData.properties || {}, null, 2);
    saveEdgeChangesBtn.onclick = async () => {
        const edgeId = modalEdgeId.value;
        const newLabel = modalEdgeLabel.value.trim();
        const newPropsStr = modalEdgeProperties.value;
        const newProps = parseJsonProperties(newPropsStr);
        if (newProps === null) { alert("Invalid JSON in properties."); return; }

        showUserStatus(graphActionStatus, "Saving edge...", "info", 0);
        try {
            await apiRequest(`${API_ENDPOINT_EDGE}${edgeId}`, 'PUT', { label: newLabel || null, properties: newProps });
            const updatedEdgeVisData = { id: edgeId, label: newLabel, properties: newProps, from: edgeData.from, to: edgeData.to };
            if (visJsCallbackForToolbarEdit) visJsCallbackForToolbarEdit(updatedEdgeVisData);
            else edgesDataSet.update(updatedEdgeVisData);

            showUserStatus(graphActionStatus, `Edge updated.`, "success");
            closeModal(edgeEditModal);
            updateSelectionInfoPanel({ nodes: [], edges: [edgeId] });
        } catch (error) {
            showUserStatus(graphActionStatus, `Failed to update edge: ${error.message}`, "error");
             if (visJsCallbackForToolbarEdit) visJsCallbackForToolbarEdit(null);
        }
    };
    openModal(edgeEditModal);
}

// --- Graph Data Fetching & Rendering ---
async function fetchGraphData() {
    if(graphLoadingOverlay) graphLoadingOverlay.classList.remove('hidden');
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
        if (graphContainer) graphContainer.innerHTML = `<div class="p-4 text-red-600 dark:text-red-400">Error loading graph: ${error.message}</div>`;
    } finally {
        if(graphLoadingOverlay) graphLoadingOverlay.classList.add('hidden');
    }
}

function getVisThemeColors() {
    const isDark = appSettings.theme === 'dark';
    return {
        nodeFontColor: isDark ? '#E5E7EB' : '#1F2937', // gray-200 : gray-800
        edgeFontColor: isDark ? '#D1D5DB' : '#374151', // gray-300 : gray-700
        edgeStrokeColor: isDark ? '#1F2937' : '#FFFFFF', // gray-800 : white (for edge label background)
        nodeBorder: isDark ? '#60A5FA' : '#2563EB', // blue-400 : blue-600
        nodeBg: isDark ? '#3B82F6' : '#93C5FD',     // blue-500 : blue-300
        nodeHighlightBorder: isDark ? '#93C5FD' : '#1D4ED8', // blue-300 : blue-700
        nodeHighlightBg: isDark ? '#60A5FA' : '#BFDBFE',     // blue-400 : blue-200
        edgeColor: isDark ? '#3B82F6' : '#60A5FA' // blue-500 : blue-400
    };
}

function renderGraph() {
    if (!graphContainer) { console.error("Graph container not found!"); return; }
    graphContainer.innerHTML = ""; 
    const themeColors = getVisThemeColors();

    const options = {
        nodes: {
            shape: "dot", size: 15,
            font: { size: 13, color: themeColors.nodeFontColor },
            borderWidth: 2, borderWidthSelected: 3,
            color: { border: themeColors.nodeBorder, background: themeColors.nodeBg, 
                     highlight: { border: themeColors.nodeHighlightBorder, background: themeColors.nodeHighlightBg }},
            shadow: { enabled: true, color: 'rgba(0,0,0,0.15)', size: 6, x: 2, y: 2 },
        },
        edges: {
            color: {color: themeColors.edgeColor, highlight: themeColors.nodeHighlightBorder, hover: themeColors.nodeHighlightBorder },
            width: 1.5, hoverWidth: 0.5, selectionWidth:1,
            smooth: { type: 'continuous', roundness: 0.2 }, // continuous is often smoother
            arrows: { to: { enabled: true, scaleFactor: 0.6, type: 'arrow' } },
            font: { size: 10, color: themeColors.edgeFontColor, strokeWidth: 2, strokeColor: themeColors.edgeStrokeColor, align: 'middle' }
        },
        interaction: {
            tooltipDelay: 150,
            navigationButtons: false, 
            keyboard: { enabled: true, speed: {x:10,y:10,zoom:0.03}, bindToWindow: false },
            hover: true, dragNodes: true, dragView: true, zoomView: true
        },
        manipulation: {
            enabled: false, initiallyActive: false,
            addNode: async (nodeData, callback) => { /* ... (same as before, ensure API calls + status) ... */ 
                nodeData.label = prompt("Node label:", "New Node");
                if (!nodeData.label) { callback(null); return; }
                try {
                    const newNode = await apiRequest(API_ENDPOINT_NODE, 'POST', { label: nodeData.label, properties: {} });
                    callback({ ...newNode, original_label: newNode.label }); // Use data from backend
                    showUserStatus(graphActionStatus, `Node "${newNode.label}" added.`, "success");
                } catch (e) { showUserStatus(graphActionStatus, `Add node failed: ${e.message}`, "error"); callback(null); }
            },
            editNode: (nodeData, callback) => { openNodeEditModal(nodeData, callback); },
            deleteNode: async (data, callback) => { /* ... (same as before) ... */ 
                const node = nodesDataSet.get(data.nodes[0]);
                if (!confirm(`Delete node "${node?.original_label || data.nodes[0]}"?`)) { callback(null); return; }
                try {
                    await apiRequest(`${API_ENDPOINT_NODE}${data.nodes[0]}`, 'DELETE');
                    callback(data); // Vis handles dataset removal
                    showUserStatus(graphActionStatus, "Node deleted.", "success");
                } catch (e) { showUserStatus(graphActionStatus, `Delete node failed: ${e.message}`, "error"); callback(null); }
            },
            addEdge: async (edgeData, callback) => { /* ... (same as before) ... */
                edgeData.label = prompt("Edge label (optional):", "");
                 try {
                    const newEdge = await apiRequest(API_ENDPOINT_EDGE, 'POST', { from_node_id: edgeData.from, to_node_id: edgeData.to, label: edgeData.label || null, properties: {} });
                    callback({ ...newEdge, from: newEdge.from_node_id, to: newEdge.to_node_id}); // Use data from backend, map from/to
                    showUserStatus(graphActionStatus, `Edge added.`, "success");
                } catch (e) { showUserStatus(graphActionStatus, `Add edge failed: ${e.message}`, "error"); callback(null); }
            },
            editEdge: (edgeData, callback) => { openEdgeEditModal(edgeData, callback); },
            deleteEdge: async (data, callback) => { /* ... (same as before) ... */
                const edge = edgesDataSet.get(data.edges[0]);
                if (!confirm(`Delete edge "${edge?.label || 'selected'}"?`)) { callback(null); return; }
                try {
                    await apiRequest(`${API_ENDPOINT_EDGE}${data.edges[0]}`, 'DELETE');
                    callback(data);
                    showUserStatus(graphActionStatus, "Edge deleted.", "success");
                } catch (e) { showUserStatus(graphActionStatus, `Delete edge failed: ${e.message}`, "error"); callback(null); }
            },
        },
        physics: {
            enabled: appSettings.physicsOnLoad, 
            solver: 'barnesHut',
            barnesHut: { gravitationalConstant: -25000, centralGravity: 0.05, springLength: 110, springConstant: 0.02, damping: 0.09, avoidOverlap: 0.2 },
            stabilization: { iterations: 300, fit: true, updateInterval: 25 },
        },
        layout: { randomSeed: undefined, improvedLayout: true, hierarchical: { enabled: false } }
    };
    network = new vis.Network(graphContainer, { nodes: nodesDataSet, edges: edgesDataSet }, options);
    network.on("click", updateSelectionInfoPanel);
    network.on("doubleClick", (params) => {
        if (editModeToggle.checked) {
            if (params.nodes.length > 0) openNodeEditModal(nodesDataSet.get(params.nodes[0]));
            else if (params.edges.length > 0) openEdgeEditModal(edgesDataSet.get(params.edges[0]));
        }
    });
    
    // Keyboard interaction fix
    document.querySelectorAll('.input-prevent-graph-interaction').forEach(input => {
        input.addEventListener('focus', () => { isEditingText = true; network.setOptions({ keyboard: { enabled: false } }); });
        input.addEventListener('blur', () => { isEditingText = false; network.setOptions({ keyboard: { enabled: true } }); });
    });
    // This is an alternative: instead of stopPropagation on graph container, disable vis.js keyboard when input is focused.
}

function updateVisThemeOptions() { // Call this when theme changes if Vis options are theme-dependent
    if (!network) return;
    const themeColors = getVisThemeColors();
    network.setOptions({
        nodes: {
            font: { color: themeColors.nodeFontColor },
            color: { border: themeColors.nodeBorder, background: themeColors.nodeBg, 
                     highlight: { border: themeColors.nodeHighlightBorder, background: themeColors.nodeHighlightBg }},
        },
        edges: {
            color: {color: themeColors.edgeColor, highlight: themeColors.nodeHighlightBorder, hover: themeColors.nodeHighlightBorder },
            font: { color: themeColors.edgeFontColor, strokeColor: themeColors.edgeStrokeColor }
        }
    });
}
// Override applyTheme to include Vis.js theme update
const originalApplyTheme = applyTheme;
applyTheme = (theme) => {
    originalApplyTheme(theme);
    updateVisThemeOptions();
};


function updateSelectionInfoPanel(params) {
    // (Largely same as before - ensure DOM IDs are correct)
    nodeInfoPanel.style.display = "none";
    edgeInfoPanel.style.display = "none";
    noSelectionMessage.style.display = "block";
    editSelectedNodeBtn.disabled = true;
    editSelectedEdgeBtn.disabled = true;

    const { nodes: clickedNodes, edges: clickedEdges } = params;
    if (clickedNodes.length > 0) {
        const node = nodesDataSet.get(clickedNodes[0]);
        if (node) {
            nodeIDDisplay.textContent = node.id;
            nodeLabelDisplay.textContent = node.original_label || node.label || "N/A";
            nodePropertiesDisplay.textContent = JSON.stringify(node.properties || {}, null, 2);
            nodeInfoPanel.style.display = "block"; noSelectionMessage.style.display = "none";
            editSelectedNodeBtn.disabled = !editModeToggle.checked;
            editSelectedNodeBtn.dataset.nodeId = node.id;
        }
    } else if (clickedEdges.length > 0) {
        const edge = edgesDataSet.get(clickedEdges[0]);
        if (edge) {
            edgeIDDisplay.textContent = edge.id;
            edgeFromDisplay.textContent = String(edge.from);
            edgeToDisplay.textContent = String(edge.to);
            edgeTypeDisplay.textContent = edge.label || "No Type";
            edgePropertiesDisplay.textContent = JSON.stringify(edge.properties || {}, null, 2);
            edgeInfoPanel.style.display = "block"; noSelectionMessage.style.display = "none";
            editSelectedEdgeBtn.disabled = !editModeToggle.checked;
            editSelectedEdgeBtn.dataset.edgeId = edge.id;
        }
    }
}

// --- Graph Controls (Sidebar) ---
editModeToggle?.addEventListener("change", (e) => {
    if (network) {
        const isChecked = e.target.checked;
        appSettings.editModeEnabled = isChecked;
        localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings)); // Persist change        
        network.setOptions({ manipulation: { enabled: isChecked } });
        if (isChecked) network.enableEditMode(); else network.disableEditMode();
        editModeHint.textContent = isChecked ? "Edit tools active. Use toolbar or double-click." : "Allows direct graph manipulation.";
        // Update button states based on selection + edit mode
        updateSelectionInfoPanel(network.getSelection()); 
        showUserStatus(graphActionStatus, `Edit mode ${isChecked ? "enabled" : "disabled"}.`, "info", 2000);
    }
});

togglePhysicsBtn?.addEventListener("click", () => {
    if (network) {
        // Toggle the state in appSettings first
        appSettings.physicsOnLoad = !appSettings.physicsOnLoad;
        network.setOptions({ physics: { enabled: appSettings.physicsOnLoad } });
        updatePhysicsButtonText(); // Update button text based on new state
        if(settingPhysicsOnLoad) settingPhysicsOnLoad.checked = appSettings.physicsOnLoad; // Sync settings modal toggle
        localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings)); // Persist
        showUserStatus(graphActionStatus, `Physics ${appSettings.physicsOnLoad ? "enabled" : "disabled"}.`, "info");
    }
});

fitGraphBtn?.addEventListener("click", () => network?.fit());
applyLayoutBtn?.addEventListener("click", () => {
    currentLayoutMethod = layoutSelect.value; // Update global current layout
    applyCurrentLayout();
});

function applyCurrentLayout() { // Renamed from applyLayout to avoid conflict
    if (!network) return;
    const selectedLayoutText = layoutSelect.options[layoutSelect.selectedIndex].text;
    showUserStatus(graphActionStatus, `Applying ${selectedLayoutText} layout...`, "info", 0);
    if(graphLoadingOverlay) graphLoadingOverlay.classList.remove('hidden');

    let layoutOptions = { hierarchical: { enabled: false } };
    let physicsEnabledForLayout = true; // Default to physics on

    switch (currentLayoutMethod) {
        case "hierarchicalRepulsion":
            layoutOptions.hierarchical = { enabled: true, direction: "UD", sortMethod: "directed", levelSeparation: 180, nodeSpacing: 130, blockShifting: true, edgeMinimization: true, parentCentralization: true};
            physicsEnabledForLayout = true; // Hierarchical repulsion solver needs physics
            network.setOptions({ physics: { solver: 'hierarchicalRepulsion', hierarchicalRepulsion: { avoidOverlap: 0.3, nodeDistance: 140}}});
            break;
        case "hierarchicalDirected":
            layoutOptions.hierarchical = { enabled: true, direction: "LR", sortMethod: "directed", levelSeparation: 220, nodeSpacing: 110, treeSpacing: 220 };
            physicsEnabledForLayout = false; // Pure hierarchical often better with physics off post-layout
            break;
        case "barnesHut":
            network.setOptions({ physics: { solver: 'barnesHut' }}); // Ensure barnesHut is active
            break;
        case "default":
        default:
            network.setOptions({ physics: { solver: 'barnesHut' }}); // Ensure barnesHut is active
            break;
    }
    
    // Important: Set layout options THEN physics options
    network.setOptions({ layout: layoutOptions });
    network.setOptions({ physics: { enabled: physicsEnabledForLayout } }); // Apply physics state for layout

    if (!physicsEnabledForLayout) {
        network.stabilize(); // If physics is off, stabilize once
    }
    // network.fit(); // Fit after layout and stabilization
    
    // Update global physics state based on what the layout function decided
    appSettings.physicsOnLoad = physicsEnabledForLayout;
    updatePhysicsButtonText();
    if(settingPhysicsOnLoad) settingPhysicsOnLoad.checked = appSettings.physicsOnLoad;
    localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings));

    // Using a timeout to allow rendering then fit nicely
    setTimeout(() => {
        network.fit();
        if(graphLoadingOverlay) graphLoadingOverlay.classList.add('hidden');
        showUserStatus(graphActionStatus, `${selectedLayoutText} layout applied.`, "success");
    }, 100); // Small delay for visual updates
}


// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    loadAppSettings();  // Load theme, settings (including active DB if stored)
    // loadAndDisplayDatabases(); // This is now called within loadAppSettings after fetching activeDB
    fetchGraphData();   // Fetch graph for the (now known) active DB
});