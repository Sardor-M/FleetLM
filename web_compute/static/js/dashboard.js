// Cluster dashboard: polls /health and renders fleet state.

const EMPTY_ROW =
    '<tr><td colspan="7" class="empty">No nodes connected - see ' +
    '<a href="/compute">how to contribute a machine</a>.</td></tr>';

function nodeRow(n) {
    const model = n.model ? n.model.split("/").pop() : "-";
    return `
        <tr>
            <td>${n.id}</td>
            <td>${n.gpu}</td>
            <td>${n.vram_mb} MB</td>
            <td>${n.runtime}</td>
            <td>${model}</td>
            <td><span class="badge badge-${n.status}">${n.status}</span></td>
            <td>${n.cpu}%</td>
            <td>${n.active}</td>
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
