import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NOTICES_EVENT = "yue2_comfy.notices";
const NOTICE_LIFE_MS = 15000;
const REFUSAL_LIFE_MS = 30000;
const SEVERITY = { warn: "warn", error: "error", info: "info" };

function nodeTitle(id) {
    const node = app.graph?.getNodeById?.(Number(id));
    return node?.title || "YuE2";
}

function showNotices(detail) {
    const issues = Array.isArray(detail?.issues) ? detail.issues : [];
    const refusal = detail?.kind === "refusal";
    const title = nodeTitle(detail?.node);
    for (const issue of issues) {
        const message = String(issue?.message ?? "").trim();
        if (!message) continue;
        const toast = app.extensionManager?.toast;
        if (typeof toast?.add !== "function") {
            console.warn("[YuE2] " + title + ": " + message);
            continue;
        }
        toast.add({
            severity: refusal ? "error" : SEVERITY[issue?.level] || "info",
            summary: refusal ? title + " stopped" : title,
            detail: message,
            life: refusal ? REFUSAL_LIFE_MS : NOTICE_LIFE_MS,
        });
    }
}

app.registerExtension({
    name: "yue2.notices",
    setup() {
        api.addEventListener(NOTICES_EVENT, (event) => showNotices(event.detail));
    },
});
