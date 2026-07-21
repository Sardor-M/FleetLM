// Browser capability probe.
//
// Browser nodes cannot serve yet: running a whole model in a tab needs a
// WebGPU inference runtime (WebLLM), which is a roadmap item. Until then this
// page reports honestly what this browser could contribute, and points at the
// native agent, which works today.

const els = {};
["webgpu-support", "gpu-name", "gpu-arch", "gpu-buffer", "verdict"].forEach((id) => {
    els[id] = document.getElementById(id);
});

function setSupport(dotClass, text) {
    els["webgpu-support"].innerHTML = `<span class="status-dot ${dotClass}"></span> ${text}`;
}

function setVerdict(text, ok) {
    els.verdict.textContent = text;
    els.verdict.className = ok ? "verdict verdict-ok" : "verdict verdict-warn";
}

async function probe() {
    if (!navigator.gpu) {
        setSupport("dot-red", "Not available");
        setVerdict(
            "This browser has no WebGPU. Chrome 113+ or Safari 18+ would be needed " +
            "once browser nodes ship. The native agent works here today.",
            false,
        );
        return;
    }

    try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter) {
            setSupport("dot-red", "No adapter");
            setVerdict("WebGPU is present but no GPU adapter was offered.", false);
            return;
        }

        // adapter.info shipped in Chrome 128; requestAdapterInfo() was removed
        // in early 2024 - support both, require neither.
        const info = adapter.info
            || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});

        const maxBufferMb = Math.round(adapter.limits.maxBufferSize / (1024 * 1024));

        setSupport("dot-green", "Available");
        els["gpu-name"].textContent =
            info.description || info.device || info.vendor || "Unknown GPU";
        els["gpu-arch"].textContent = info.architecture || "unknown";
        els["gpu-buffer"].textContent = `${maxBufferMb} MB`;

        // A 4-bit model needs roughly 0.6 GB per billion parameters, and the
        // largest single buffer bounds what a WebGPU runtime can hold at once.
        const suggestion =
            maxBufferMb >= 4096 ? "an 8B 4-bit model"
            : maxBufferMb >= 2048 ? "a 3B 4-bit model"
            : maxBufferMb >= 512 ? "a 1B 4-bit model"
            : "small models only";
        setVerdict(
            `This GPU looks capable of ${suggestion} once browser nodes ship. ` +
            `Today, run the native agent to contribute.`,
            true,
        );
    } catch (e) {
        setSupport("dot-red", "Error");
        setVerdict(`WebGPU probe failed: ${e.message}`, false);
    }
}

probe();
