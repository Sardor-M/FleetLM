// Browser compute node: detects WebGPU, registers with the orchestrator over
// an outbound WebSocket, and (Phase 4) will run whole-model inference via
// WebLLM. Currently a protocol demo on the layer-shard path.

// ── State ──────────────────────────────────────────────────────────
// Derive the WebSocket URL from wherever this page was actually served,
// so it works on localhost, LAN, or behind TLS alike.
const WS_URL = (location.protocol === "https:" ? "wss://" : "ws://")
    + location.host + "/nodes/ws";

let ws = null;
let nodeId = crypto.randomUUID();
let gpuDevice = null;
let gpuInfo = {};
let heartbeatInterval = null;
let assignedLayers = null;

// ── Logging ────────────────────────────────────────────────────────
function log(msg, level = "info") {
    const el = document.getElementById("log");
    const time = new Date().toLocaleTimeString();
    const cls = { info: "log-info", ok: "log-ok", warn: "log-warn", error: "log-err" }[level] || "log-info";
    el.innerHTML += `<div class="${cls}">[${time}] ${msg}</div>`;
    el.scrollTop = el.scrollHeight;
}

// ── WebGPU Detection ───────────────────────────────────────────────
async function detectGPU() {
    const supportEl = document.getElementById("webgpu-support");

    if (!navigator.gpu) {
        supportEl.innerHTML = '<span class="status-dot dot-red"></span> Not available';
        log("WebGPU not supported in this browser", "error");
        return false;
    }

    try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter) {
            supportEl.innerHTML = '<span class="status-dot dot-red"></span> No adapter';
            log("No WebGPU adapter found", "error");
            return false;
        }

        // adapter.info shipped in Chrome 128; requestAdapterInfo() was
        // removed in early 2024 — support both, tolerate neither.
        const info = adapter.info
            || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
        gpuInfo = {
            gpu_name: info.description || info.device || info.vendor || "Unknown GPU",
            architecture: info.architecture || "unknown",
            vendor: info.vendor || "unknown",
        };

        gpuDevice = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
            }
        });

        const maxBuf = gpuDevice.limits.maxBufferSize;
        gpuInfo.gpu_vram_mb = Math.round(maxBuf / (1024 * 1024));

        supportEl.innerHTML = '<span class="status-dot dot-green"></span> Available';
        document.getElementById("gpu-name").textContent = gpuInfo.gpu_name;
        document.getElementById("gpu-arch").textContent = gpuInfo.architecture;
        document.getElementById("gpu-buffer").textContent = `${gpuInfo.gpu_vram_mb} MB`;
        document.getElementById("node-id").textContent = nodeId.slice(0, 8);

        log(`GPU detected: ${gpuInfo.gpu_name} (${gpuInfo.gpu_vram_mb} MB)`, "ok");
        return true;

    } catch (e) {
        supportEl.innerHTML = '<span class="status-dot dot-red"></span> Error';
        log(`WebGPU error: ${e.message}`, "error");
        return false;
    }
}

// ── WebSocket Connection ───────────────────────────────────────────
function connectWS() {
    ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        document.getElementById("ws-status").innerHTML =
            '<span class="status-dot dot-green"></span> Connected';
        log("Connected to orchestrator", "ok");

        ws.send(JSON.stringify({
            type: "register",
            node_id: nodeId,
            gpu_name: gpuInfo.gpu_name || "unknown",
            gpu_vram_mb: gpuInfo.gpu_vram_mb || 0,
            runtime: "webgpu",
            mode: "layer_shard",
        }));

        heartbeatInterval = setInterval(sendHeartbeat, 5000);
    };

    ws.onmessage = (event) => {
        if (typeof event.data === "string") {
            handleMessage(JSON.parse(event.data));
        } else {
            handleBinaryData(event.data);
        }
    };

    ws.onclose = () => {
        document.getElementById("ws-status").innerHTML =
            '<span class="status-dot dot-yellow"></span> Disconnected';
        log("Disconnected from orchestrator", "warn");
        clearInterval(heartbeatInterval);

        // Auto-reconnect after 5s while the node is running
        setTimeout(() => {
            if (document.getElementById("btn-stop").disabled === false) {
                log("Reconnecting...", "info");
                connectWS();
            }
        }, 5000);
    };

    ws.onerror = () => log("WebSocket error", "error");
}

function handleMessage(msg) {
    switch (msg.type) {
        case "layer_assignment":
            assignedLayers = [msg.start_layer, msg.end_layer];
            document.getElementById("layers").textContent = `${msg.start_layer} - ${msg.end_layer}`;
            document.getElementById("node-status").textContent = "Loading weights...";
            log(`Assigned layers ${msg.start_layer}-${msg.end_layer} of ${msg.model_id}`, "info");

            // Simulated weight load (Phase 4: real weight download + WebGPU buffers)
            setTimeout(() => {
                ws.send(JSON.stringify({
                    type: "layers_loaded",
                    node_id: nodeId,
                    start_layer: msg.start_layer,
                    end_layer: msg.end_layer,
                }));
                document.getElementById("node-status").textContent = "Ready";
                log(`Layers ${msg.start_layer}-${msg.end_layer} loaded into WebGPU`, "ok");
            }, 2000);
            break;

        case "session_end":
            log(`Session ${msg.session_id} ended`, "info");
            document.getElementById("node-status").textContent = "Ready";
            break;

        default:
            log(`Unknown message type: ${msg.type}`, "warn");
    }
}

function handleBinaryData(buffer) {
    log(`Received binary frame: ${buffer.byteLength} bytes`, "info");
}

// ── Heartbeat ──────────────────────────────────────────────────────
function sendHeartbeat() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: "heartbeat",
            node_id: nodeId,
            cpu_usage: 0,
            gpu_usage: 0,
            ram_usage: 0,
            active_sessions: 0,
        }));
    }
}

// ── Controls ───────────────────────────────────────────────────────
async function startNode() {
    document.getElementById("btn-start").disabled = true;
    document.getElementById("btn-stop").disabled = false;

    const gpuOk = await detectGPU();
    if (!gpuOk) {
        log("Cannot start: WebGPU not available. Try Chrome 113+ with a GPU.", "error");
        document.getElementById("btn-start").disabled = false;
        document.getElementById("btn-stop").disabled = true;
        return;
    }

    connectWS();
}

function stopNode() {
    document.getElementById("btn-start").disabled = false;
    document.getElementById("btn-stop").disabled = true;

    clearInterval(heartbeatInterval);
    if (ws) {
        ws.close();
        ws = null;
    }
    document.getElementById("ws-status").innerHTML =
        '<span class="status-dot dot-gray"></span> Disconnected';
    document.getElementById("node-status").textContent = "Stopped";
    log("Node stopped", "warn");
}

// ── Init ───────────────────────────────────────────────────────────
detectGPU();
