import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

export function installStyle(id, css) {
    if (document.getElementById(id)) return;
    const style = document.createElement("style");
    style.id = id;
    style.textContent = css;
    document.head.appendChild(style);
}

export function element(tag, className, text) {
    const made = document.createElement(tag);
    if (className) made.className = className;
    if (text !== undefined) made.textContent = text;
    return made;
}

export async function ask(url, body) {
    const response = await api.fetchApi(url, {
        method: body ? "POST" : "GET",
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
    });
    let payload = {};
    try {
        payload = await response.json();
    } catch (error) {
        payload = {};
    }
    return { ok: response.ok, status: response.status, payload };
}

const BASE_STYLE_ID = "yue2-controls-style";
const BASE_STYLE = `
.yue2-buttons { display: flex; width: 100%; height: 100%; gap: 6px;
    font-family: system-ui, sans-serif; }
.yue2-buttons button { flex: 1 1 0; min-width: 0; font: inherit; font-size: 12px;
    padding: 0 8px; border-radius: 6px; cursor: pointer;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    color: var(--input-text, #ddd); background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-buttons button:hover { background: var(--comfy-menu-bg, #353535); }
.yue2-buttons button:active { background: #3B7DD8; border-color: #3B7DD8; color: #fff; }
.yue2-buttons button[disabled] { opacity: 0.5; cursor: not-allowed; }
`;

export const BUTTON_H = 24;
const BUTTON_MARGIN = 4;

export function buttonRow(node, name, buttons) {
    installStyle(BASE_STYLE_ID, BASE_STYLE);

    const row = document.createElement("div");
    row.className = "yue2-buttons";
    const made = buttons.map((spec) => {
        const button = document.createElement("button");
        button.textContent = spec.label;
        if (spec.tooltip) button.title = spec.tooltip;
        button.addEventListener("click", () => spec.onClick(button));
        row.appendChild(button);
        return button;
    });

    const widget = node.addDOMWidget(name, "yue2_buttons", row, {
        hideOnZoom: false,
        margin: BUTTON_MARGIN,
        hideInPanel: true,
        getValue: () => undefined,
        setValue: () => {},
        getMinHeight: () => BUTTON_H + 2 * BUTTON_MARGIN,
        getMaxHeight: () => BUTTON_H + 2 * BUTTON_MARGIN,
    });
    widget.serialize = false;
    widget.serializeValue = () => undefined;
    return { widget, buttons: made };
}

export function panelWidget(node, name, element, height, grow = false) {
    const widget = node.addDOMWidget(name, "yue2_panel", element, {
        hideOnZoom: false,
        margin: BUTTON_MARGIN,
        hideInPanel: true,
        getValue: () => undefined,
        setValue: () => {},
        getMinHeight: () => height() + 2 * BUTTON_MARGIN,
        getMaxHeight: () => (grow ? undefined : height() + 2 * BUTTON_MARGIN),
    });
    widget.serialize = false;
    widget.serializeValue = () => undefined;
    return widget;
}

export function widgetNamed(node, name) {
    return node?.widgets?.find((w) => w.name === name);
}

export function graphChanged() {
    try {
        const tracker = app.extensionManager?.workflow?.activeWorkflow?.changeTracker;
        if (tracker?.captureCanvasState) tracker.captureCanvasState();
        else tracker?.checkState?.();
    } catch (error) {
        console.error("[YuE2] the workflow was not told that its graph changed:", error);
    }
}

export function setWidgetValue(node, name, value) {
    const widget = widgetNamed(node, name);
    if (!widget || widget.value === value) return;
    widget.value = value;
    widget.callback?.(value, app.canvas, node);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    graphChanged();
}

export function showWidget(node, name, shown) {
    const widget = widgetNamed(node, name);
    if (!widget) return;
    widget.hidden = !shown;
    widget.options = widget.options || {};
    widget.options.hidden = !shown;
    if (shown) delete widget.computeSize;
    else widget.computeSize = () => [0, -4];
}

export function sourceOf(node, name) {
    const link = node.inputs?.find((input) => input.name === name)?.link;
    if (link === null || link === undefined) return null;
    const record = app.graph?.links?.get?.(link) ?? app.graph?.links?.[link];
    return record ? app.graph.getNodeById(record.origin_id) : null;
}

const WINDOW_STYLE_ID = "yue2-window-style";
const WINDOW_STYLE = `
.yue2-back { position: fixed; inset: 0; z-index: 1300; display: flex;
    align-items: center; justify-content: center; background: rgba(0, 0, 0, 0.55);
    font-family: system-ui, sans-serif; }
.yue2-panel { width: min(920px, 95vw); height: calc(100vh - 32px); display: flex;
    flex-direction: column; overflow: hidden;
    background: var(--comfy-menu-bg, #353535); color: var(--input-text, #ddd);
    border: 1px solid var(--border-color, #4e4e4e); border-radius: 10px;
    padding: 16px 20px; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    box-sizing: border-box; }
`;

const ASKED_STYLE_ID = "yue2-asked-style";
const ASKED_STYLE = `
.yue2-panel.yue2-asked { width: min(460px, 92vw); height: auto; max-height: calc(100vh - 64px);
    gap: 10px; padding: 18px 20px; }
.yue2-asked-title { font-size: 14px; font-weight: 600; }
.yue2-asked-text { font-size: 13px; line-height: 1.5; white-space: pre-line; overflow: auto;
    color: var(--descrip-text, #bbb); }
.yue2-asked-row { display: flex; gap: 8px; justify-content: flex-end; margin-top: 4px; }
.yue2-asked button { font: inherit; font-size: 12px; padding: 5px 14px; border-radius: 6px;
    cursor: pointer; color: var(--input-text, #ddd); background: var(--comfy-input-bg, #2b2b2b);
    border: 1px solid var(--border-color, #4e4e4e); }
.yue2-asked button:hover { background: var(--comfy-menu-bg, #353535); }
.yue2-asked button.yue2-asked-go { background: #3B7DD8; border-color: #3B7DD8; color: #fff; }
.yue2-asked button.yue2-asked-danger { background: #B4433A; border-color: #B4433A; color: #fff; }
`;

export function confirmed(text, { title = "", ok = "OK", cancel = "Cancel",
                                  danger = false } = {}) {
    installStyle(ASKED_STYLE_ID, ASKED_STYLE);
    return new Promise((resolve) => {
        let answer = false;
        const made = frame({ onClose: () => resolve(answer) });
        made.panel.classList.add("yue2-asked");
        if (title) made.panel.appendChild(element("div", "yue2-asked-title", title));
        made.panel.appendChild(element("div", "yue2-asked-text", String(text)));
        const row = element("div", "yue2-asked-row");
        const no = element("button", "", cancel);
        const yes = element("button", "yue2-asked-go" + (danger ? " yue2-asked-danger" : ""), ok);
        no.addEventListener("click", () => made.close());
        yes.addEventListener("click", () => {
            answer = true;
            made.close();
        });
        row.appendChild(no);
        row.appendChild(yes);
        made.panel.appendChild(row);
        yes.focus();
    });
}

export function warned(text, { title = "", ok = "OK" } = {}) {
    installStyle(ASKED_STYLE_ID, ASKED_STYLE);
    return new Promise((resolve) => {
        const made = frame({ onClose: () => resolve() });
        made.panel.classList.add("yue2-asked");
        made.panel.appendChild(element("div", "yue2-asked-title", title || "That did not work"));
        made.panel.appendChild(element("div", "yue2-asked-text", String(text)));
        const row = element("div", "yue2-asked-row");
        const yes = element("button", "yue2-asked-go", ok);
        yes.addEventListener("click", () => made.close());
        row.appendChild(yes);
        made.panel.appendChild(row);
        yes.focus();
    });
}

const OPEN = [];

export function frame({ onClose, sticky } = {}) {
    installStyle(WINDOW_STYLE_ID, WINDOW_STYLE);
    const back = element("div", "yue2-back");
    const panel = element("div", "yue2-panel");
    back.appendChild(panel);
    const me = { close };
    OPEN.push(me);

    function close() {
        const at = OPEN.indexOf(me);
        if (at >= 0) OPEN.splice(at, 1);
        document.removeEventListener("keydown", onKey, true);
        back.remove();
        onClose?.();
    }

    function onKey(event) {
        if (event.key !== "Escape" || OPEN[OPEN.length - 1] !== me) return;
        event.stopPropagation();
        me.onEscape ? me.onEscape() : close();
    }

    panel.addEventListener("keydown", (event) => event.stopPropagation());
    document.addEventListener("keydown", onKey, true);
    back.addEventListener("pointerdown", (event) => {
        if (event.target === back && !sticky) close();
    });
    document.body.appendChild(back);
    return { back, panel, close, handle: me };
}
