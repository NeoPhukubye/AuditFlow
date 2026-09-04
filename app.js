// AuditFlow web frontend — vanilla ES module, no build step.
// Submits a ticket to the configured AuditFlow backend and renders the result.

const DEFAULT_BASE = ""; // production backend URL is injected via web/config.json

let apiBase = DEFAULT_BASE;

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};

async function loadConfig() {
  try {
    const res = await fetch("/config.json", { cache: "no-store" });
    if (res.ok) {
      const cfg = await res.json();
      apiBase = cfg.apiBase || DEFAULT_BASE;
    }
  } catch {
    // config.json absent -> keep default (empty). Site still loads as docs.
  }
  refreshStatus();
}

function setStatus(msg, kind = "muted") {
  const s = $("api-status");
  s.textContent = msg;
  s.className = kind;
}

async function refreshStatus() {
  if (!apiBase) {
    setStatus("backend not configured", "muted");
    return;
  }
  setStatus("checking backend…", "muted");
  try {
    const res = await fetch(`${apiBase}/health`, { cache: "no-store" });
    const data = await res.json();
    if (data.gemini_configured) {
      setStatus("backend online · ready", "ok");
    } else {
      setStatus("backend online (set GEMINI_API_KEY to triage)", "warn");
    }
    $("config-hint").classList.add("hidden");
  } catch {
    setStatus("backend unreachable", "err");
  }
}

function esc(html) {
  const d = document.createElement("div");
  d.textContent = html;
  return d.innerHTML;
}

function renderDraft(draft) {
  if (!draft) return el("p", "muted", "No draft available.");
  const c = el("div", "result-grid");
  c.append(
    el("div", null, `${draft.category} • priority: ${draft.priority}`),
    el("p", "muted", draft.root_cause_summary),
    el("p", null, "Reply:"),
    el("pre", "reply", draft.proposed_reply),
  );
  const ul = el("ul");
  (draft.actions_taken || []).forEach((a) => ul.append(el("li", null, a)));
  c.append(el("p", null, "Actions:"), ul);
  return c;
}

function renderResult(data) {
  const box = el("div", "result-grid");

  if (data.success) {
    box.append(el("span", "badge-approved", "APPROVED"));
    box.append(el("p", null, `Triage succeeded in ${data.attempts} attempt(s).`));
    box.append(renderDraft(data.final_draft));
  } else {
    if (data.escalated) box.append(el("span", "badge-escalated", "ESCALATED to human review"));
    else box.append(el("span", "badge-rejected", "REJECTED"));
    box.append(el("p", null, data.error || "Pipeline did not approve the draft."));
    box.append(el("p", null, `Attempts: ${data.attempts ?? 0}`));
    box.append(renderDraft(data.last_draft));
    const ul = el("ul");
    (data.audit_notes && data.audit_notes.critique_points || []).forEach((c) => ul.append(el("li", null, c)));
    box.append(el("p", null, "Critique:"), ul);
  }

  const kb = el("ul", "kb-list");
  (data.kb_sources || []).forEach((s) => kb.append(el("li", null, `[KB] ${s}`)));
  if (kb.children.length) box.append(el("p", null, "KB sources:"), kb);

  return box;
}

async function submit(e) {
  e.preventDefault();
  if (!apiBase) {
    setStatus("backend not configured — set API_URL secret", "err");
    return;
  }
  const btn = el("button", null, "Running…");
  btn.disabled = true;
  const submitBtn = $("ticket-form").querySelector("button");
  submitBtn.replaceWith(btn);
  $("result").innerHTML = "";
  $("result-card").hidden = false;

  const payload = {
    ticket_id: "web-" + Date.now(),
    source: $("source").value,
    subject: $("subject").value,
    description: $("description").value,
    customer_email: $("customer_email").value || undefined,
  };

  try {
    const res = await fetch(`${apiBase}/webhooks/ticket`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (res.ok) {
      $("result").replaceChildren(renderResult(data));
    } else {
      $("result").replaceChildren(
        el("div", "result-grid", null),
        el("span", res.status >= 500 ? "badge-rejected" : "badge-escalated",
          `${res.status} ${res.statusText}`),
        el("p", null, data.detail || "Request failed."),
      );
    }
  } catch (err) {
    $("result").replaceChildren(
      el("span", "badge-rejected", "ERROR"),
      el("p", null, `Could not reach backend: ${err.message}`),
    );
  } finally {
    btn.replaceWith(submitBtn);
  }
}

$("ticket-form").addEventListener("submit", submit);
$("refresh-status").addEventListener("click", refreshStatus);
loadConfig();
