// Cluster dashboard: polls /health and renders fleet state.

const EMPTY_ROW =
    '<tr><td colspan="8" style="color:#666; text-align:center;">' +
    'No nodes connected. Run <code>python -m node_agent</code> or open ' +
    '<a href="/compute">/compute</a> in Chrome.</td></tr>';

function nodeRow(n) {
    const model = n.model
        ? n.model.split("/").pop()
        : (n.layers ? `layers ${n.layers[0]}-${n.layers[1]}` : "-");
    return `
        <tr>
            <td>${n.id}</td>
            <td>${n.gpu}</td>
            <td>${n.vram_mb} MB</td>
            <td>${n.runtime}</td>
            <td>${n.mode === "whole_model" ? "whole model" : "layer shard"}</td>
            <td>${model}</td>
            <td><span class="badge badge-${n.status}">${n.status}</span></td>
            <td>${n.cpu}%</td>
        </tr>
    `;
}

async function refresh() {
    try {
        const resp = await fetch("/health");
        const data = await resp.json();

        document.getElementById("total-nodes").textContent = data.nodes.total_nodes;
        document.getElementById("ready-nodes").textContent = data.nodes.ready_nodes;
        document.getElementById("active-sessions").textContent = data.active_sessions;

        const tbody = document.getElementById("nodes-table");
        tbody.innerHTML = data.nodes.nodes.length === 0
            ? EMPTY_ROW
            : data.nodes.nodes.map(nodeRow).join("");
    } catch (e) {
        // server might be restarting; keep the last rendered state
    }
}

refresh();
setInterval(refresh, 3000);
