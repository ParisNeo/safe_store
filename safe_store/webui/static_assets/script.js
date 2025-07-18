// --- DOM Elements ---
const sidebar = document.getElementById("sidebar");
const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
const mainContent = document.getElementById("main-content");
const accordions = document.querySelectorAll(".accordion-header");
const themeToggleBtn = document.getElementById("theme-toggle-btn");
const themeIconSun = document.getElementById("theme-icon-sun");
const themeIconMoon = document.getElementById("theme-icon-moon");
const themeText = document.getElementById("theme-text");
const settingsBtn = document.getElementById("settings-btn");
const uploadDocumentBtn = document.getElementById("upload-document-btn");
const databasesBtn = document.getElementById("databases-btn");
const graphLoadingOverlay = document.getElementById("graph-loading-overlay");
const graphContainer = document.getElementById("graph-container");
const graphActionStatus = document.getElementById("graph-action-status");
const taskProgressContainer = document.getElementById("task-progress-container");

// Selection & Info Panel
const selectionAccordion = document.getElementById("selection-accordion");
const nodeInfoPanel = document.getElementById("node-info-panel");
const edgeInfoPanel = document.getElementById("edge-info-panel");
const nodeIDDisplay = document.getElementById("node-id-display");
const nodeLabelDisplay = document.getElementById("node-label-display");
const nodePropertiesDisplay = document.getElementById("node-properties-display");
const editSelectedNodeBtn = document.getElementById("edit-selected-node-btn");
const expandNeighborsBtn = document.getElementById("expand-neighbors-btn");
const setStartNodeBtn = document.getElementById("set-start-node-btn");
const setEndNodeBtn = document.getElementById("set-end-node-btn");
const edgeIDDisplay = document.getElementById("edge-id-display");
const edgeTypeDisplay = document.getElementById("edge-type-display");
const edgePropertiesDisplay = document.getElementById("edge-properties-display");
const editSelectedEdgeBtn = document.getElementById("edit-selected-edge-btn");
const noSelectionMessage = document.getElementById("no-selection");

// Controls
const graphSearchForm = document.getElementById("graph-search-form");
const graphSearchInput = document.getElementById("graph-search-input");
const editModeBtn = document.getElementById("edit-mode-btn");
const editModeBtnText = document.getElementById("edit-mode-btn-text");
const togglePhysicsBtn = document.getElementById("toggle-physics-btn");
const physicsBtnText = document.getElementById("physics-btn-text");
const fuseEntitiesBtn = document.getElementById("fuse-entities-btn");
const fitGraphBtn = document.getElementById("fit-graph-btn");
const findPathForm = document.getElementById("find-path-form");
const startNodeInput = document.getElementById("start-node-input");
const endNodeInput = document.getElementById("end-node-input");
const directedPathToggle = document.getElementById("directed-path-toggle");

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

// Chat Modal
const chatBtn = document.getElementById("chat-btn");
const chatModal = document.getElementById("chat-modal");
const closeChatModalBtn = document.getElementById("close-chat-modal-btn");
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSubmitBtn = document.getElementById("chat-submit-btn");

// Add/Edit Modals
const nodeAddModal = document.getElementById("node-add-modal");
const cancelNodeAddBtn = document.getElementById("cancel-node-add-btn");
const nodeAddForm = document.getElementById("node-add-form");
const modalAddNodeLabel = document.getElementById("modal-add-node-label");
const edgeAddModal = document.getElementById("edge-add-modal");
const cancelEdgeAddBtn = document.getElementById("cancel-edge-add-btn");
const edgeAddForm = document.getElementById("edge-add-form");
const modalAddEdgeLabel = document.getElementById("modal-add-edge-label");
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

// Settings & DB Modals
const settingsModal = document.getElementById("settings-modal");
const closeSettingsModalBtn = document.getElementById("close-settings-modal-btn");
const modalThemeToggle = document.getElementById("modal-theme-toggle");
const settingPhysicsOnLoad = document.getElementById("setting-physics-on-load");
const saveSettingsBtn = document.getElementById("save-settings-btn");
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


// --- API Configuration ---
const API_BASE_URL = "";
const API_ENDPOINTS = {
    UPLOAD: `${API_BASE_URL}/upload-file/`,
    GRAPH_DATA: `${API_BASE_URL}/graph-data/`,
    GRAPH_SEARCH: (q) => `${API_BASE_URL}/graph/search/?q=${encodeURIComponent(q)}`,
    GRAPH_FUSE: `${API_BASE_URL}/graph/fuse/`,
    NODE: (id) => `${API_BASE_URL}/graph/node/${id ? id : ''}`,
    EDGE: (id) => `${API_BASE_URL}/graph/edge/${id ? id : ''}`,
    NEIGHBORS: (nodeId) => `${API_BASE_URL}/graph/node/${nodeId}/neighbors`,
    PATH: `${API_BASE_URL}/graph/path`,
    CHAT: `${API_BASE_URL}/api/chat/rag`,
    DATABASES: `${API_BASE_URL}/api/databases`,
    DATABASE_ACTION: (dbName, action) => `${API_BASE_URL}/api/databases/${encodeURIComponent(dbName)}/${action}`,
};

// --- Global State & Vis.js Instances ---
let network;
let nodesDataSet = new vis.DataSet();
let edgesDataSet = new vis.DataSet();
let appSettings = {
    theme: 'light',
    physicsOnLoad: true,
    editModeEnabled: false,
    sidebarCollapsed: false
};
let isPhysicsActive = true;
let socket = null;
let sessionId = null;
const markdownConverter = new showdown.Converter();

// --- Helper Functions ---
function showUserStatus(element, message, type = "success", duration = 4000) {
    if (!element) return;
    let iconClass = "fa-check-circle text-green-500";
    if (type === "error") {
        iconClass = "fa-times-circle text-red-500";
    } else if (type === "info") {
        iconClass = "fa-info-circle text-blue-500";
    } else if (type === "warning") {
        iconClass = "fa-exclamation-triangle text-yellow-500";
    }
    element.innerHTML = `<i class="fas ${iconClass} mr-2"></i><span>${message}</span>`;
    if (duration > 0) {
        setTimeout(() => {
            if (element.innerHTML.includes(message)) element.innerHTML = "";
        }, duration);
    }
}

async function apiRequest(endpoint, method = 'GET', body = null, isFormData = false) {
    const options = {
        method
    };
    if (!isFormData && body) {
        options.headers = {
            'Content-Type': 'application/json'
        };
        options.body = JSON.stringify(body);
    } else if (isFormData && body) {
        options.body = body;
    }
    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({
                detail: `HTTP error ${response.status}`
            }));
            throw new Error(errorData.detail || `Request failed`);
        }
        return response.status === 204 || response.status === 202 ? response : await response.json();
    } catch (error) {
        console.error(`API call to ${endpoint} failed:`, error);
        throw error;
    }
}

function openModal(modal) {
    modal?.classList.remove("hidden");
}

function closeModal(modal) {
    modal?.classList.add("hidden");
}

// --- Socket.IO & Progress ---
function setupSocketIO() {
    socket = io({
        path: '/sio'
    });
    socket.on('connect', () => {
        sessionId = socket.id;
        console.log("Socket.IO connected:", sessionId);
        chatSubmitBtn.disabled = false;
    });
    socket.on('progress_update', handleProgressUpdate);
    socket.on('disconnect', () => {
        console.log("Socket.IO disconnected.");
        sessionId = null;
        chatSubmitBtn.disabled = true;
    });
    socket.on('connect_error', (error) => console.error("Socket.IO connection error:", error));
}

function handleProgressUpdate(data) {
    const {
        task_id,
        filename,
        progress,
        message
    } = data;
    let el = document.getElementById(`progress-${task_id}`);
    const container = task_id.startsWith('fuse_') ? taskProgressContainer : uploadProgressArea;

    if (!el) {
        el = document.createElement('div');
        el.id = `progress-${task_id}`;
        el.className = 'p-2 my-1 bg-bg-tertiary rounded-md text-xs';
        const fileIdentifier = filename || (task_id.startsWith('fuse_') ? "Entity Fusion" : "Processing Task");
        el.innerHTML = `<div class="font-semibold mb-1 truncate">${fileIdentifier}</div><div id="progress-message-${task_id}" class="text-text-secondary mb-1 truncate">Initializing...</div><div class="progress-bar-container"><div id="progress-bar-${task_id}" class="progress-bar" style="width: 0%;"></div></div>`;
        container.appendChild(el);
    }
    document.getElementById(`progress-message-${task_id}`).textContent = message;
    const progressBar = document.getElementById(`progress-bar-${task_id}`);
    progressBar.style.width = `${progress * 100}%`;
    if (progress >= 1.0) {
        progressBar.classList.remove('bg-accent-primary');
        progressBar.classList.add(message.toLowerCase().includes("error") ? 'bg-red-600' : 'bg-green-600');
        setTimeout(() => el.remove(), 8000);
    }
}

// --- Theme & Settings Management ---
function applyTheme(theme) {
    appSettings.theme = theme;
    if (theme === 'dark') {
        document.documentElement.classList.add('dark');
        themeIconSun.classList.add('hidden');
        themeIconMoon.classList.remove('hidden');
        themeText.textContent = "Dark";
    } else {
        document.documentElement.classList.remove('dark');
        themeIconSun.classList.remove('hidden');
        themeIconMoon.classList.add('hidden');
        themeText.textContent = "Light";
    }
    if (modalThemeToggle) modalThemeToggle.checked = theme === 'dark';
    updateVisThemeOptions();
}

function loadAppSettings() {
    const s = localStorage.getItem('graphExplorerSettings');
    if (s) {
        try {
            appSettings = { ...appSettings,
                ...JSON.parse(s)
            };
        } catch (e) {
            console.error("Bad settings:", e);
        }
    }
    applyTheme(appSettings.theme);
    applySidebarState(appSettings.sidebarCollapsed);
    if (settingPhysicsOnLoad) settingPhysicsOnLoad.checked = appSettings.physicsOnLoad;
    isPhysicsActive = appSettings.physicsOnLoad;
    if (editModeBtn) {
        updateEditModeButton();
    }
    updatePhysicsButtonText();
}

function saveAppSettings() {
    appSettings.physicsOnLoad = settingPhysicsOnLoad.checked;
    appSettings.editModeEnabled = appSettings.editModeEnabled;
    appSettings.sidebarCollapsed = !sidebar.classList.contains("w-96");
    localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings));
    updatePhysicsButtonText();
}
themeToggleBtn?.addEventListener("click", () => applyTheme(appSettings.theme === 'dark' ? 'light' : 'dark'));
modalThemeToggle?.addEventListener("change", (e) => applyTheme(e.target.checked ? 'dark' : 'light'));
settingsBtn?.addEventListener("click", () => openModal(settingsModal));
closeSettingsModalBtn?.addEventListener("click", () => closeModal(settingsModal));
saveSettingsBtn?.addEventListener("click", () => {
    saveAppSettings();
    closeModal(settingsModal);
});

// --- Sidebar ---
function applySidebarState(collapsed) {
    if (collapsed) {
        sidebar.classList.remove("w-96");
        sidebar.classList.add("w-16");
    } else {
        sidebar.classList.add("w-96");
        sidebar.classList.remove("w-16");
    }
    setTimeout(() => network?.redraw(), 350);
}
sidebarToggleBtn?.addEventListener("click", () => {
    appSettings.sidebarCollapsed = !appSettings.sidebarCollapsed;
    applySidebarState(appSettings.sidebarCollapsed);
    localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings));
});
accordions.forEach(button => {
    button.addEventListener('click', () => {
        const content = button.nextElementSibling;
        button.classList.toggle('active');
        content.classList.toggle('open');
    });
});

function openAccordion(accordionHeader) {
    if (accordionHeader && !accordionHeader.classList.contains('active')) {
        accordionHeader.click();
    }
}

// --- Modals Listeners ---
databasesBtn?.addEventListener("click", () => {
    loadAndDisplayDatabases();
    openModal(databasesModal);
});
closeDatabasesModalBtn?.addEventListener("click", () => closeModal(databasesModal));

chatBtn?.addEventListener("click", () => openModal(chatModal));
closeChatModalBtn?.addEventListener("click", () => closeModal(chatModal));

uploadDocumentBtn?.addEventListener("click", () => {
    uploadForm.reset();
    fileListPreview.innerHTML = '';
    uploadProgressArea.innerHTML = '';
    uploadOverallStatus.innerHTML = '';
    uploadSubmitBtn.disabled = true;
    openModal(uploadModal);
});
closeUploadModalBtn?.addEventListener("click", () => closeModal(uploadModal));

async function loadAndDisplayDatabases() {
    try {
        const dbs = await apiRequest(API_ENDPOINTS.DATABASES);
        databaseListContainer.innerHTML = '';
        dbs.forEach(db => {
            const div = document.createElement('div');
            div.className = `p-3 rounded-md flex items-center ${db.is_active ? 'bg-bg-active' : 'bg-bg-tertiary'}`;
            div.innerHTML = `<div class="font-semibold">${db.name}${db.is_active ? '<span class="text-green-500 text-sm ml-2">(Active)</span>' : ''}</div><div class="ml-auto space-x-2">${!db.is_active ? `<button data-db-name="${db.name}" class="activate-db-btn control-button text-xs py-1 px-2 bg-green-500 text-white">Activate</button>` : ''}<button data-db-name="${db.name}" class="delete-db-btn control-button text-xs py-1 px-2 bg-red-600 text-white" ${dbs.length<=1?'disabled':''}>Delete</button></div>`;
            databaseListContainer.appendChild(div);
        });
        document.querySelectorAll('.activate-db-btn').forEach(b => b.addEventListener('click', handleActivateDatabase));
        document.querySelectorAll('.delete-db-btn').forEach(b => b.addEventListener('click', handleDeleteDatabase));
    } catch (e) {
        databaseListContainer.innerHTML = `<p class="text-red-500">Error: ${e.message}</p>`;
    }
}
async function handleActivateDatabase(e) {
    if (confirm(`Activate "${e.target.dataset.dbName}" and reload the application?`)) {
        try {
            await apiRequest(API_ENDPOINTS.DATABASE_ACTION(e.target.dataset.dbName, 'activate'), 'PUT');
            window.location.reload();
        } catch (err) {
            alert(`Failed to activate: ${err.message}`);
        }
    }
}
async function handleDeleteDatabase(e) {
    const btn = e.target.closest('button');
    if (confirm(`Delete configuration for "${btn.dataset.dbName}"? This does NOT delete files.`)) {
        try {
            await apiRequest(API_ENDPOINTS.DATABASE_ACTION(btn.dataset.dbName, 'delete'), 'DELETE');
            loadAndDisplayDatabases();
        } catch (err) {
            alert(`Failed to delete: ${err.message}`);
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
createDbForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = createDbNameInput.value.trim();
    if (!name) return;
    try {
        await apiRequest(API_ENDPOINTS.DATABASES, 'POST', {
            name
        });
        closeModal(createDbModal);
        loadAndDisplayDatabases();
    } catch (err) {
        createDbError.textContent = `Error: ${err.message}`;
    }
});

fileInput?.addEventListener("change", () => {
    fileListPreview.innerHTML = '';
    if (fileInput.files.length) {
        Array.from(fileInput.files).forEach(f => {
            fileListPreview.innerHTML += `<div class="text-xs p-1 truncate">${f.name}</div>`;
        });
        uploadSubmitBtn.disabled = false;
    } else {
        uploadSubmitBtn.disabled = true;
    }
});
uploadForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files.length || !sessionId) {
        showUserStatus(uploadOverallStatus, "No files selected or not connected to server.", "error");
        return;
    }
    uploadSubmitBtn.disabled = true;
    showUserStatus(uploadOverallStatus, `Uploading ${fileInput.files.length} file(s)...`, "info", 0);
    const formData = new FormData();
    Array.from(fileInput.files).forEach(file => formData.append("files", file));
    formData.append("guidance", extractionGuidanceInput.value.trim());
    formData.append("sid", sessionId);
    try {
        await apiRequest(API_ENDPOINTS.UPLOAD, "POST", formData, true);
        showUserStatus(uploadOverallStatus, "Upload successful. Processing started.", "success", 6000);
        setTimeout(fetchGraphData, 5000);
    } catch (error) {
        showUserStatus(uploadOverallStatus, `Upload failed: ${error.message}`, "error", 0);
    } finally {
        uploadSubmitBtn.disabled = false;
    }
});

// --- Graph Data & Rendering ---
async function fetchGraphData() {
    graphLoadingOverlay.classList.remove('hidden');
    showUserStatus(graphActionStatus, "Loading graph...", "info", 0);
    try {
        const data = await apiRequest(API_ENDPOINTS.GRAPH_DATA);
        nodesDataSet.clear();
        edgesDataSet.clear();
        nodesDataSet.add(data.nodes.map(n => ({ ...n,
            original_label: n.label
        })));
        edgesDataSet.add(data.edges);
        if (!network) renderGraph();
        else network.setData({
            nodes: nodesDataSet,
            edges: edgesDataSet
        });
        showUserStatus(graphActionStatus, "Graph loaded.", "success");
    } catch (error) {
        showUserStatus(graphActionStatus, `Error loading graph: ${error.message}`, "error");
    } finally {
        graphLoadingOverlay.classList.add('hidden');
    }
}

function getVisThemeColors() {
    const isDark = appSettings.theme === 'dark';
    return {
        nodeFont: isDark ? '#f1f5f9' : '#0f172a', // text-primary
        edgeFont: isDark ? '#94a3b8' : '#64748b', // text-secondary
        edgeStroke: 'transparent',
        nodeBorder: isDark ? '#3b82f6' : '#93c5fd', // border-accent
        nodeBg: isDark ? '#2563eb' : '#bfdbfe', 
        nodeHighlightBorder: isDark ? '#93c5fd' : '#1d4ed8',
        nodeHighlightBg: isDark ? '#60a5fa' : '#dbeafe', 
        edgeColor: isDark ? '#334155' : '#cbd5e1', // slate-700 / slate-300
    };
}


function renderGraph() {
    if (!graphContainer) return;
    const c = getVisThemeColors();
    isPhysicsActive = appSettings.physicsOnLoad;
    const options = {
        nodes: {
            shape: "dot",
            size: 15,
            font: {
                size: 13,
                color: c.nodeFont
            },
            borderWidth: 2,
            color: {
                border: c.nodeBorder,
                background: c.nodeBg,
                highlight: {
                    border: c.nodeHighlightBorder,
                    background: c.nodeHighlightBg
                }
            }
        },
        edges: {
            color: {
                color: c.edgeColor,
                highlight: c.nodeHighlightBorder
            },
            width: 1.5,
            smooth: {
                type: 'continuous'
            },
            arrows: {
                to: {
                    enabled: true,
                    scaleFactor: 0.6
                }
            },
            font: {
                size: 10,
                color: c.edgeFont,
                strokeWidth: 0,
            }
        },
        interaction: {
            tooltipDelay: 150,
            navigationButtons: false,
            keyboard: true,
            hover: true
        },
        physics: {
            enabled: isPhysicsActive,
            solver: 'barnesHut',
            barnesHut: {
                gravitationalConstant: -25000,
                centralGravity: 0.05,
                springLength: 110
            },
            stabilization: {
                iterations: 300,
                fit: true
            }
        },
        manipulation: {
            enabled: false,
            addNode: (nodeData, callback) => openNodeAddModal(nodeData, callback),
            editNode: (nodeData, callback) => openNodeEditModal(nodeData, callback),
            addEdge: (edgeData, callback) => openEdgeAddModal(edgeData, callback),
            editEdge: {
                editWithoutDrag: (edgeData, callback) => openEdgeEditModal(edgeData, callback)
            },
            deleteNode: async (data, callback) => {
                if (confirm(`Delete ${data.nodes.length} node(s)?`)) {
                    try {
                        for (const id of data.nodes) await apiRequest(API_ENDPOINTS.NODE(id), 'DELETE');
                        callback(data);
                    } catch (e) {
                        alert(`Failed to delete: ${e}`);
                        callback(null);
                    }
                } else callback(null);
            },
            deleteEdge: async (data, callback) => {
                if (confirm(`Delete ${data.edges.length} edge(s)?`)) {
                    try {
                        for (const id of data.edges) await apiRequest(API_ENDPOINTS.EDGE(id), 'DELETE');
                        callback(data);
                    } catch (e) {
                        alert(`Failed to delete: ${e}`);
                        callback(null);
                    }
                } else callback(null);
            },
        }
    };
    network = new vis.Network(graphContainer, {
        nodes: nodesDataSet,
        edges: edgesDataSet
    }, options);
    network.on("click", updateSelectionInfoPanel);
    network.on("doubleClick", (p) => {
        if (appSettings.editModeEnabled) {
            if (p.nodes.length) openNodeEditModal(nodesDataSet.get(p.nodes[0]), (d) => nodesDataSet.update(d));
            else if (p.edges.length) openEdgeEditModal(edgesDataSet.get(p.edges[0]), (d) => edgesDataSet.update(d));
        }
    });
    if (isPhysicsActive) network.on("stabilizationIterationsDone", () => {
        isPhysicsActive = false;
        updatePhysicsButtonText();
        network.setOptions({
            physics: false
        });
    });
}

function updateVisThemeOptions() {
    if (!network) return;
    const c = getVisThemeColors();
    network.setOptions({
        nodes: {
            font: {
                color: c.nodeFont
            },
            color: {
                border: c.nodeBorder,
                background: c.nodeBg,
                highlight: {
                    border: c.nodeHighlightBorder,
                    background: c.nodeHighlightBg
                }
            }
        },
        edges: {
            color: {
                color: c.edgeColor,
                highlight: c.nodeHighlightBorder
            },
            font: {
                color: c.edgeFont,
            }
        }
    });
}

function updateSelectionInfoPanel(params) {
    nodeInfoPanel.style.display = "none";
    edgeInfoPanel.style.display = "none";
    noSelectionMessage.style.display = "block";
    
    editSelectedNodeBtn.disabled = true;
    expandNeighborsBtn.disabled = true;
    setStartNodeBtn.disabled = true;
    setEndNodeBtn.disabled = true;
    editSelectedEdgeBtn.disabled = true;
    
    const {
        nodes,
        edges
    } = params;
    if (nodes.length > 0) {
        openAccordion(selectionAccordion.querySelector('.accordion-header'));
        const node = nodesDataSet.get(nodes[0]);
        if (node) {
            nodeIDDisplay.textContent = node.id;
            nodeLabelDisplay.textContent = node.original_label || node.label || "N/A";
            nodePropertiesDisplay.textContent = JSON.stringify(node.properties || {}, null, 2);
            nodeInfoPanel.style.display = "block";
            noSelectionMessage.style.display = "none";
            
            editSelectedNodeBtn.disabled = !appSettings.editModeEnabled;
            editSelectedNodeBtn.dataset.nodeId = node.id;
            expandNeighborsBtn.disabled = false;
            expandNeighborsBtn.dataset.nodeId = node.id;
            setStartNodeBtn.disabled = false;
            setStartNodeBtn.dataset.nodeId = node.id;
            setEndNodeBtn.disabled = false;
            setEndNodeBtn.dataset.nodeId = node.id;
        }
    } else if (edges.length > 0) {
        openAccordion(selectionAccordion.querySelector('.accordion-header'));
        const edge = edgesDataSet.get(edges[0]);
        if (edge) {
            edgeIDDisplay.textContent = edge.id;
            edgeTypeDisplay.textContent = edge.label || "No Type";
            edgePropertiesDisplay.textContent = JSON.stringify(edge.properties || {}, null, 2);
            edgeInfoPanel.style.display = "block";
            noSelectionMessage.style.display = "none";
            
            editSelectedEdgeBtn.disabled = !appSettings.editModeEnabled;
            editSelectedEdgeBtn.dataset.edgeId = edge.id;
        }
    }
}

// --- Add/Edit Modal Logic ---
let visJsAddNodeCallback, visJsAddEdgeCallback;
let tempNodeData, tempEdgeData;

function openNodeAddModal(nodeData, callback) {
    visJsAddNodeCallback = callback;
    tempNodeData = nodeData;
    openModal(nodeAddModal);
    modalAddNodeLabel.focus();
}

function openEdgeAddModal(edgeData, callback) {
    visJsAddEdgeCallback = callback;
    tempEdgeData = edgeData;
    openModal(edgeAddModal);
    modalAddEdgeLabel.focus();
}
nodeAddForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
        const newNode = await apiRequest(API_ENDPOINTS.NODE(), 'POST', {
            label: modalAddNodeLabel.value.trim(),
            properties: {}
        });
        tempNodeData.id = newNode.id;
        tempNodeData.label = newNode.label;
        tempNodeData.properties = newNode.properties;
        tempNodeData.original_label = newNode.label;
        if (visJsAddNodeCallback) visJsAddNodeCallback(tempNodeData);
    } catch (error) {
        alert(`Failed to add node: ${error}`);
        if (visJsAddNodeCallback) visJsAddNodeCallback(null);
    } finally {
        closeModal(nodeAddModal);
    }
});
edgeAddForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = {
        from_node_id: tempEdgeData.from,
        to_node_id: tempEdgeData.to,
        label: modalAddEdgeLabel.value.trim() || null,
        properties: {}
    };
    try {
        const newEdge = await apiRequest(API_ENDPOINTS.EDGE(), 'POST', payload);
        tempEdgeData.id = newEdge.id;
        tempEdgeData.label = newEdge.label;
        tempEdgeData.properties = newEdge.properties;
        if (visJsAddEdgeCallback) visJsAddEdgeCallback(tempEdgeData);
    } catch (error) {
        alert(`Failed to add edge: ${error}`);
        if (visJsAddEdgeCallback) visJsAddEdgeCallback(null);
    } finally {
        closeModal(edgeAddModal);
    }
});
cancelNodeAddBtn?.addEventListener('click', () => {
    if (visJsAddNodeCallback) visJsAddNodeCallback(null);
    closeModal(nodeAddModal);
});
cancelEdgeAddBtn?.addEventListener('click', () => {
    if (visJsAddEdgeCallback) visJsAddEdgeCallback(null);
    closeModal(edgeAddModal);
});

let saveNodeChanges = async () => {};
let saveEdgeChanges = async () => {};

function setupEditModal(modal, closeBtn, cancelBtn, saveBtn, saveFn) {
    closeBtn?.addEventListener("click", () => closeModal(modal));
    cancelBtn?.addEventListener("click", () => closeModal(modal));
    saveBtn?.addEventListener("click", saveFn);
}
setupEditModal(nodeEditModal, closeNodeEditModalBtn, cancelNodeEditBtn, saveNodeChangesBtn, () => saveNodeChanges());
setupEditModal(edgeEditModal, closeEdgeEditModalBtn, cancelEdgeEditBtn, saveEdgeChangesBtn, () => saveEdgeChanges());

function openNodeEditModal(nodeData, visJsCallback) {
    modalNodeId.value = nodeData.id;
    modalNodeLabel.value = nodeData.original_label || nodeData.label || '';
    modalNodeProperties.value = JSON.stringify(nodeData.properties || {}, null, 2);
    saveNodeChanges = async () => {
        try {
            const payload = {
                label: modalNodeLabel.value.trim(),
                properties: JSON.parse(modalNodeProperties.value)
            };
            const updatedNode = await apiRequest(API_ENDPOINTS.NODE(nodeData.id), 'PUT', payload);
            const updatedVisData = { ...updatedNode,
                original_label: updatedNode.label,
                id: updatedNode.id
            };
            if (visJsCallback) visJsCallback(updatedVisData);
            else nodesDataSet.update(updatedVisData);
            closeModal(nodeEditModal);
            updateSelectionInfoPanel({
                nodes: [nodeData.id],
                edges: []
            });
        } catch (error) {
            alert(`Failed to update node: ${error}`);
            if (visJsCallback) visJsCallback(null);
        }
    };
    openModal(nodeEditModal);
}

function openEdgeEditModal(edgeData, visJsCallback) {
    modalEdgeId.value = edgeData.id;
    modalEdgeLabel.value = edgeData.label || '';
    modalEdgeProperties.value = JSON.stringify(edgeData.properties || {}, null, 2);
    saveEdgeChanges = async () => {
        try {
            const payload = {
                label: modalEdgeLabel.value.trim() || null,
                properties: JSON.parse(modalEdgeProperties.value)
            };
            const updatedEdge = await apiRequest(API_ENDPOINTS.EDGE(edgeData.id), 'PUT', payload);
            const updatedVisData = { ...updatedEdge,
                from: updatedEdge.from_node_id,
                to: updatedEdge.to_node_id,
                id: updatedEdge.id
            };
            if (visJsCallback) visJsCallback(updatedVisData);
            else edgesDataSet.update(updatedVisData);
            closeModal(edgeEditModal);
            updateSelectionInfoPanel({
                nodes: [],
                edges: [edgeData.id]
            });
        } catch (error) {
            alert(`Failed to update edge: ${error}`);
            if (visJsCallback) visJsCallback(null);
        }
    };
    openModal(edgeEditModal);
}
editSelectedNodeBtn?.addEventListener("click", () => {
    const id = editSelectedNodeBtn.dataset.nodeId;
    if (id && nodesDataSet.get(id)) openNodeEditModal(nodesDataSet.get(id), (d) => nodesDataSet.update(d));
});
editSelectedEdgeBtn?.addEventListener("click", () => {
    const id = editSelectedEdgeBtn.dataset.edgeId;
    if (id && edgesDataSet.get(id)) openEdgeEditModal(edgesDataSet.get(id), (d) => edgesDataSet.update(d));
});

// --- Graph Controls ---
function updateEditModeButton() {
    if (!editModeBtn || !editModeBtnText) return;
    const isEnabled = appSettings.editModeEnabled;
    editModeBtnText.textContent = isEnabled ? "Disable Editing" : "Enable Editing";
    editModeBtn.classList.toggle('bg-green-500', isEnabled);
    editModeBtn.classList.toggle('dark:bg-green-600', isEnabled);
    editModeBtn.classList.toggle('text-white', isEnabled);
}
editModeBtn?.addEventListener("click", () => {
    if (!network) return;
    appSettings.editModeEnabled = !appSettings.editModeEnabled;
    localStorage.setItem('graphExplorerSettings', JSON.stringify(appSettings));
    network.setOptions({
        manipulation: {
            enabled: appSettings.editModeEnabled
        }
    });
    updateEditModeButton();
    updateSelectionInfoPanel(network.getSelection());
});

function updatePhysicsButtonText() {
    if (physicsBtnText) physicsBtnText.textContent = isPhysicsActive ? "Stop Physics" : "Start Physics";
}
togglePhysicsBtn?.addEventListener("click", () => {
    if (!network) return;
    isPhysicsActive = !isPhysicsActive;
    network.setOptions({
        physics: {
            enabled: isPhysicsActive
        }
    });
    updatePhysicsButtonText();
});
fitGraphBtn?.addEventListener("click", () => network?.fit());
fuseEntitiesBtn?.addEventListener("click", async () => {
    if (!sessionId) {
        showUserStatus(graphActionStatus, "Not connected", "error");
        return;
    }
    if (!confirm("This will scan the graph and attempt to merge similar nodes. This can be a slow process. Continue?")) return;
    showUserStatus(graphActionStatus, "Starting fusion...", "info", 0);
    try {
        await apiRequest(API_ENDPOINTS.GRAPH_FUSE, 'POST', {
            sid: sessionId
        });
        showUserStatus(graphActionStatus, `Fusion task started.`, "success", 10000);
    } catch (e) {
        showUserStatus(graphActionStatus, `Failed: ${e.message}`, "error");
    }
});
findPathForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const start = parseInt(startNodeInput.value, 10),
        end = parseInt(endNodeInput.value, 10);
    if (isNaN(start) || isNaN(end)) {
        showUserStatus(graphActionStatus, "Valid start and end node IDs are required.", "warning");
        return;
    }
    showUserStatus(graphActionStatus, `Finding path...`, "info", 0);
    try {
        const pathData = await apiRequest(API_ENDPOINTS.PATH, 'POST', {
            start_node_id: start,
            end_node_id: end,
            directed: directedPathToggle.checked
        });
        network.unselectAll();
        network.selectNodes(pathData.nodes.map(n => n.node_id));
        network.selectEdges(pathData.relationships.map(r => r.relationship_id));
        network.fit({
            nodes: pathData.nodes.map(n => n.node_id),
            animation: true
        });
        showUserStatus(graphActionStatus, `Path found with ${pathData.nodes.length} nodes.`, "success");
    } catch (err) {
        showUserStatus(graphActionStatus, `Path failed: ${err.message}`, "error");
    }
});
expandNeighborsBtn?.addEventListener("click", async (e) => {
    const id = parseInt(e.currentTarget.dataset.nodeId, 10);
    if (isNaN(id)) return;
    showUserStatus(graphActionStatus, `Expanding node ${id}...`, "info", 0);
    try {
        const d = await apiRequest(API_ENDPOINTS.NEIGHBORS(id));
        const newNodes = d.nodes.map(n => ({ ...n,
            id: n.node_id,
            label: n.label,
            group: n.label,
            properties: n.properties,
            original_label: n.label
        }));
        const newEdges = d.relationships.map(r => ({ ...r,
            id: r.relationship_id,
            from: r.source_node_id,
            to: r.target_node_id,
            label: r.type,
            properties: r.properties
        }));
        nodesDataSet.update(newNodes);
        edgesDataSet.update(newEdges);
        showUserStatus(graphActionStatus, `Added ${newNodes.length} neighbors.`, "success");
    } catch (err) {
        showUserStatus(graphActionStatus, `Failed to expand: ${err.message}`, "error");
    }
});
graphSearchForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = graphSearchInput.value.trim();
    if (!q) return;
    showUserStatus(graphActionStatus, "Searching...", "info", 0);
    try {
        const res = await apiRequest(API_ENDPOINTS.GRAPH_SEARCH(q));
        showUserStatus(graphActionStatus, `Found ${res.nodes.length} matching nodes. Highlighting results.`, "success");
        network.unselectAll();
        if (res.nodes.length > 0) {
            network.selectNodes(res.nodes.map(n => n.id));
            network.fit({
                nodes: res.nodes.map(n => n.id),
                animation: true
            });
        }
    } catch (err) {
        showUserStatus(graphActionStatus, `Search failed: ${err.message}`, "error");
    }
});

setStartNodeBtn?.addEventListener("click", (e) => {
    const nodeId = e.currentTarget.dataset.nodeId;
    if (nodeId) {
        startNodeInput.value = nodeId;
        showUserStatus(graphActionStatus, `Node ${nodeId} set as path start.`, "info");
        openAccordion(findPathForm.closest('.accordion-item').querySelector('.accordion-header'));
    }
});

setEndNodeBtn?.addEventListener("click", (e) => {
    const nodeId = e.currentTarget.dataset.nodeId;
    if (nodeId) {
        endNodeInput.value = nodeId;
        showUserStatus(graphActionStatus, `Node ${nodeId} set as path end.`, "info");
        openAccordion(findPathForm.closest('.accordion-item').querySelector('.accordion-header'));
    }
});


// --- RAG Chat Logic ---
chatForm?.addEventListener("submit", handleChatSubmit);

function addChatMessage(message, sender, isThinking = false) {
    const d = document.createElement("div");
    d.className = `chat-message ${sender === 'user' ? 'chat-user' : 'chat-ai'}`;
    if (isThinking) {
        d.innerHTML = `<i class="fas fa-spinner fa-spin"></i>`;
        d.id = "thinking-indicator";
    } else if (sender === 'ai') {
        d.innerHTML = markdownConverter.makeHtml(message);
    } else {
        d.textContent = message;
    }
    chatMessages.appendChild(d);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
async function handleChatSubmit(e) {
    e.preventDefault();
    const q = chatInput.value.trim();
    if (!q) return;
    addChatMessage(q, 'user');
    chatInput.value = '';
    addChatMessage('', 'ai', true);
    try {
        const r = await apiRequest(API_ENDPOINTS.CHAT, 'POST', {
            query: q
        });
        document.getElementById("thinking-indicator")?.remove();
        addChatMessage(r.answer, 'ai');
    } catch (err) {
        document.getElementById("thinking-indicator")?.remove();
        addChatMessage(`Error: ${err.message}`, 'ai');
    }
}

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    loadAppSettings();
    setupSocketIO();
    fetchGraphData();
});