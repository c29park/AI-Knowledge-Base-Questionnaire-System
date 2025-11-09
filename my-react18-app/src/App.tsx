import React, { useCallback, useId, useMemo, useRef, useState, useEffect } from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import type { PieLabelRenderProps } from "recharts";

// ====== Tunables / Endpoints ======
const MAX_BYTES = 2 * 1024 * 1024; // keep for now; set to Infinity to remove limit
const ENDPOINTS = {
  kbUpload: "/api/upload/kb",
  qUpload: "/api/upload/questions",
  kbList: "/api/kb",
  qList: "/api/questions",
  kbCreatePrimary: "/api/kb",               // preferred: POST single item
  kbUpdatePrimary: (id: string) => `/api/kb/${encodeURIComponent(id)}`, // preferred: PUT single item
  kbDeletePrimary: "/api/kb",               // preferred: DELETE { ids: [] }
  kbAppend: "/api/upload/kb?append=1",      // fallback: POST array of items
  kbDeleteAlt: "/api/kb/delete",            // fallback: POST { ids: [] }

  // NEW: answers + jobs
  answersList: "/api/answers",
  answersGenerate: "/api/answers/generate",
  jobStatus: (id: string) => `/api/jobs/${encodeURIComponent(id)}`,
  answersClear: "/api/answers/clear",
} as const;

type KBItem = { 
  id: string; 
  category: string; 
  question: string; 
  answer: string;
  created_at?: string;
  updated_at?: string;
};
type QItem  = { id: string; text: string };

// very forgiving shape for answers.json
type AnswerItem = {
  id: string;
  question: string;
  answer: string;
  context?: string;
  used_categories?: string[];
  fused_parent_ids?: string[];
  generated_at?: string;
};

function pretty(text: string, cap = 120_000) {
  if (text.length <= cap) return text;
  return text.slice(0, cap) + "\n… (truncated)";
}

function looksLikeJsonFile(file: File) {
  const nameOk = file.name.toLowerCase().endsWith(".json");
  const typeOk = file.type === "application/json" || file.type === "text/json";
  return nameOk || typeOk;
}

function fmt(dt?: string) {
  if (!dt) return "—";
  try { return new Date(dt).toLocaleString(); } catch { return dt; }
}

function Spinner({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" role="status" aria-label="Loading">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" strokeWidth="4" fill="none" />
      <path d="M12 2 a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" fill="none">
        <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="0.8s" repeatCount="indefinite" />
      </path>
    </svg>
  );
}

/* ------------------ Helpers for resilient API calls & IDs ------------------ */
async function tryRequests(reqs: Array<() => Promise<Response>>): Promise<Response> {
  let lastErr: any;
  for (const mk of reqs) {
    try {
      const res = await mk();
      if (res.ok) return res;
      lastErr = new Error(`${res.status} ${res.statusText}`);
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr ?? new Error("All requests failed");
}


// STRICT kb_<natural number>
const KB_ID_RE = /^kb_(\d+)$/i;

function getNextKbId(rows: {id: string}[]){
  let max = 0;
  for (const {id} of rows){
    const m = KB_ID_RE.exec(String(id));
    if(m){
      const n = parseInt(m[1], 10);
      if (!Number.isNaN(n) && n > max) max=n;
    }
  }
  return `kb_${max+1}`;
}

// Build the category list for one answer row
function rowCategories(r: AnswerItem): string[] {
  const s = new Set<string>();
  const isGlobal = (c?: string) => !!c && c.trim().toUpperCase() === "GLOBAL";

  // from /answers.json if present
  if (Array.isArray(r.used_categories)) {
    for (const c of r.used_categories) {
      const v = String(c || "").trim();
      if (v && !isGlobal(v)) s.add(v);
    }
  }
  // from parsed context blocks ([n] Category ... Q: ... A: ...)
  for (const b of parseContextBlocks(r.context)) {
    const v = String(b.category || "").trim();
    if (v && !isGlobal(v)) s.add(v);
  }
  return Array.from(s);
}

// All unique categories across results (sorted, stable)
function allCategories(rows: AnswerItem[]): string[] {
  const s = new Set<string>();
  for (const r of rows) for (const c of rowCategories(r)) s.add(c);
  return Array.from(s).sort((a, b) => a.localeCompare(b));
}

/* ------------------ Component ------------------ */
export default function App() {
  const [phase, setPhase] = useState<"kb" | "questions">("kb");
  const endpoint = phase === "kb" ? ENDPOINTS.kbUpload : ENDPOINTS.qUpload;

  const [file, setFile] = useState<File | null>(null);
  const [rawText, setRawText] = useState<string>("");
  const [data, setData] = useState<unknown>(null);
  const [error, setError] = useState<string>("");
  const [status, setStatus] = useState<"idle" | "parsing" | "ready" | "submitting" | "success" | "error">("idle");
  const [dragActive, setDragActive] = useState(false);

  // Management screen state
  const [view, setView] = useState<"upload" | "management">("upload");
  const [kbRows, setKbRows] = useState<KBItem[]>([]);
  const [qRows, setQRows] = useState<QItem[]>([]);
  const [answersRows, setAnswersRows] = useState<AnswerItem[]>([]);
  const [activeTab, setActiveTab] = useState<"kb" | "questions" | "results">("kb");


  // View mode: "table" or "cards" (KB + Questions). Results is table-only.
  const [viewMode, setViewMode] = useState<"table" | "cards">("table");

  // Selection state for KB
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const selectedCount = selected.size;
  const canEdit = selectedCount === 1;
  // FAB / modals
  const [fabOpen, setFabOpen] = useState(false);
  const [showAddOne, setShowAddOne] = useState(false);
  const [showAddMany, setShowAddMany] = useState(false);
  const [showEdit, setShowEdit] = useState(false);

  // Add/Edit form state
  const emptyKB: KBItem = { id: "", category: "", question: "", answer: "" };
  const [kbForm, setKbForm] = useState<KBItem>(emptyKB);

  // Generate Answers job state
  
  const [genStatus, setGenStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [genError, setGenError] = useState<string>("");
  // Per-result context visibility
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const toggleContext = (key: string) =>
    setExpanded(prev => ({ ...prev, [key]: !prev[key] }));

  // RESULTS: filter & sort state
  const [selectedCats, setSelectedCats] = useState<string[]>([]);
  const [showCatPicker, setShowCatPicker] = useState(false);

  type SortKey = "generated_at" | "category" | "id";
  const [sortKey, setSortKey] = useState<SortKey>("id");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");



  const inputId = useId();
  const inputRef = useRef<HTMLInputElement | null>(null);

  const reset = () => {
    setFile(null);
    setRawText("");
    setData(null);
    setError("");
    setStatus("idle");
    if (inputRef.current) inputRef.current.value = "";
  };

  const goBackToKB = () => {
    if (phase === "questions" && (file || rawText)) {
      const ok = confirm("Going back will clear the selected questions file. Continue?");
      if (!ok) return;
    }
    reset();
    setPhase("kb");
  };

  const parseFile = useCallback(async (f: File) => {
    setError("");
    setStatus("parsing");

    if (!looksLikeJsonFile(f)) {
      setStatus("error");
      setError("Please choose a .json file.");
      return;
    }
    if (f.size > MAX_BYTES) {
      setStatus("error");
      setError(`File is too large (${(f.size / (1024 * 1024)).toFixed(2)} MiB). Max ${(MAX_BYTES / (1024 * 1024)).toFixed(0)} MiB.`);
      return;
    }

    try {
      const text = await f.text();
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch (e: any) {
        setStatus("error");
        setError(`Invalid JSON: ${e?.message ?? e}`);
        return;
      }
      setFile(f);
      setRawText(text);
      setData(parsed);
      setStatus("ready");
    } catch (e: any) {
      setStatus("error");
      setError(`Failed to read file: ${e?.message ?? e}`);
    }
  }, []);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) parseFile(f);
  };

  // Drag-n-drop
  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); setDragActive(true); };
  const onDragLeave = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (f) parseFile(f);
  };

  // Load data when switching to management
  useEffect(() => {
    if (view === "management") { void loadManagementData(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view]);

  // Delete answers.json when the tab/page is closed or reloaded
  useEffect(() => {
    const clearOnUnload = () => {
      const url = ENDPOINTS.answersClear;
      try {
        const ok = (navigator as any).sendBeacon?.(
          url,
          new Blob([JSON.stringify({ reason: "page-unload" })], { type: "application/json" })
        );
        if (!ok) {
          fetch(url, {
            method: "POST",
            keepalive: true,
            headers: { "Content-Type": "application/json" },
            body: "{}",
          }).catch(() => {});
        }
      } catch {
        fetch(url, {
          method: "POST",
          keepalive: true,
          headers: { "Content-Type": "application/json" },
          body: "{}",
        }).catch(() => {});
      }
    };

  window.addEventListener("pagehide", clearOnUnload);
  window.addEventListener("beforeunload", clearOnUnload);
  return () => {
    window.removeEventListener("pagehide", clearOnUnload);
    window.removeEventListener("beforeunload", clearOnUnload);
  };
}, []);


  async function loadManagementData() {
    try {
      const [kbRes, qRes, aRes] = await Promise.all([
        fetch(ENDPOINTS.kbList),
        fetch(ENDPOINTS.qList),
        fetch(ENDPOINTS.answersList),
      ]);

      const kbJson = kbRes.ok ? await kbRes.json() : [];
      const qJson  = qRes.ok ? await qRes.json()  : [];
      const aJson  = aRes.ok ? await aRes.json()  : [];

      setKbRows(Array.isArray(kbJson) ? kbJson : []);
      setQRows(Array.isArray(qJson) ? qJson : []);

      const normalized = normalizeAnswers(aJson);
      setAnswersRows(normalized);

      setSelected(new Set());
    } catch (e: any) {
      setError(`Failed to fetch uploaded data: ${e?.message ?? e}`);
    }
  }


  // Normalize answers to a safe shape
  function normalizeAnswers(raw: any): AnswerItem[] {
    if (!Array.isArray(raw)) return [];
    return raw.map((r: any) => {
      const id  = String(r?.id ?? r?.eval_id ?? r?.qid ?? "");
      const q   = String(r?.question ?? r?.input?.question ?? r?.text ?? "");
      const ans = String(r?.answer ?? r?.output?.answer ?? r?.generated_answer ?? "");
      const ctx =
        typeof r?.context === "string"
          ? r.context
          : (Array.isArray(r?.documents)
              ? r.documents.map((d: any) => d?.page_content).filter(Boolean).join("\n\n")
              : undefined);

      // NEW: accept several possible timestamp fields
      const ts = r?.generated_at ?? r?.timestamp ?? r?.ts ?? r?.created_at;
      return {
        id,
        question: q,
        answer: ans,
        context: ctx,
        used_categories: Array.isArray(r?.used_categories) ? r.used_categories : undefined,
        fused_parent_ids: Array.isArray(r?.fused_parent_ids) ? r.fused_parent_ids : undefined,
        generated_at: typeof ts === "string" ? ts : undefined,
      };
    }).filter(x => x.id || x.question || x.answer);
  }

  /* ------------------ CRUD with fallbacks + optimistic UI ------------------ */
  async function createKB(item: KBItem) {
    const res = await tryRequests([
      () => fetch(ENDPOINTS.kbCreatePrimary, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(item) }),
      () => fetch(ENDPOINTS.kbAppend,         { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify([item]) }),
    ]);
    return res;
  }

  async function updateKB(item: KBItem) {
    const res = await tryRequests([
      () => fetch(ENDPOINTS.kbUpdatePrimary(item.id), { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(item) }),
      async () => {
        await tryRequests([
          () => fetch(ENDPOINTS.kbDeletePrimary, { method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids: [item.id] }) }),
          () => fetch(ENDPOINTS.kbDeleteAlt,     { method: "POST",   headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids: [item.id] }) }),
        ]);
        return fetch(ENDPOINTS.kbAppend, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify([item]) });
      },
    ]);
    return res;
  }

  async function deleteKB(ids: string[]) {
    const res = await tryRequests([
      () => fetch(ENDPOINTS.kbDeletePrimary, { method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids }) }),
      () => fetch(ENDPOINTS.kbDeleteAlt,     { method: "POST",   headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids }) }),
    ]);
    return res;
  }

  // --- Generate Answers flow ---
  async function startGenerateAnswers() {
    try {
      setGenError("");
      setGenStatus("running");
      const res = await fetch(ENDPOINTS.answersGenerate, { method: "POST" });
      if (!res.ok) throw new Error(`Failed to start generation (${res.status})`);
      const { job_id } = await res.json();
      

      await pollJob(job_id);

      // ⬇️ Make results visible and switch to it
      await loadResults();
      setActiveTab("results");
      setGenStatus("done");
    } catch (e: any) {
      setGenError(e?.message ?? String(e));
      setGenStatus("error");
    }
  }


  async function pollJob(jobId: string) {
    // simple polling loop
    for (;;) {
      await new Promise(r => setTimeout(r, 1500));
      const j = await fetch(ENDPOINTS.jobStatus(jobId));
      if (!j.ok) throw new Error(`Job status failed (${j.status})`);
      const info = await j.json();
      if (info.status === "done") { setGenStatus("done"); return; }
      if (info.status === "error") {
        setGenStatus("error");
        throw new Error(`Generation failed (code ${info.returncode ?? "?"}). Check logs on server.`);
      }
      // else still running -> continue
    }
  }

  async function loadResults() {
    try {
      const aRes = await fetch(ENDPOINTS.answersList);
      const aJson = aRes.ok ? await aRes.json() : [];
      const list = normalizeAnswers(aJson);
      setAnswersRows(list);
      return list.length;                   // ⬅️ optional
    } catch (e: any) {
      setError(`Failed to fetch answers: ${e?.message ?? e}`);
      return 0;                             // ⬅️ optional
    }
  }

  async function clearAnswers() {
    try { await fetch(ENDPOINTS.answersClear, { method: "POST" }); } catch {}
    setAnswersRows([]);
    setGenStatus("idle");
    setGenError("");
    setExpanded({});            // reset toggles
    if (activeTab === "results") setActiveTab("kb");
  }


  // Clear selection when switching tab/view
  useEffect(() => { setSelected(new Set()); }, [activeTab, viewMode]);

  const toggleOne = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };
  const allIds = useMemo(() => kbRows.map(r => r.id), [kbRows]);
  const allChecked = selectedCount > 0 && selectedCount === allIds.length;
  const toggleAll = () => setSelected(prev => (prev.size === allIds.length ? new Set() : new Set(allIds)));

  /* ------------------ Upload flow ------------------ */
  const submit = async () => {
    if (!rawText) return;
    setStatus("submitting");
    setError("");
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: rawText,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`Request failed (${res.status}). ${text ? "\n\n" + text : "(No response body)"}`);
      }
      if (phase === "kb") {
        setTimeout(() => {
          reset();
          setPhase("questions");
        }, 500);
      } else {
        setStatus("success");
        setTimeout(async () => {
          await loadManagementData();
          setStatus("idle");
          setView("management");
          setActiveTab("kb");
          setViewMode("table");
        }, 800);
      }
    } catch (e: any) {
      setStatus("error");
      setError(
        `Submit failed: ${e?.message ?? e}\n\nTip: Make sure FastAPI is running on http://localhost:8000 and your Vite proxy forwards /api to it.`
      );
    }
  };

  /* ------------------ Counts / Stats ------------------ */
  const countSummary = useMemo(() => {
    if (!data) return "";
    let n = 0;
    if (Array.isArray(data)) n = (data as any[]).length;
    else if (data && typeof data === "object") {
      const obj = data as Record<string, unknown>;
      for (const k of ["pairs", "qa", "items", "data", "entries", "questions"]) {
        const v = (obj as any)[k]; if (Array.isArray(v)) { n = v.length; break; }
      }
    }
    if (phase === "kb") return `${n} ${n === 1 ? "pair" : "pairs"}`;
    return `${n} ${n === 1 ? "question" : "questions"}`;
  }, [data, phase]);

  const kbStats = useMemo(() => {
    const map = new Map<string, number>();
    for (const r of kbRows) {
      const cat = (r?.category ?? "Uncategorized") as string;
      map.set(cat, (map.get(cat) ?? 0) + 1);
    }
    const total = Array.from(map.values()).reduce((a, b) => a + b, 0);
    const rows = Array.from(map.entries()).map(([category, count]) => ({
      category,
      count,
      pct: total ? +((count / total) * 100).toFixed(1) : 0,
    }));
    return { total, rows };
  }, [kbRows]);

  const PURPLES = ["#7c3aed", "#6d28d9", "#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe", "#5b21b6", "#4c1d95"];

  const titleText = phase === "kb" ? "1. Upload your Knowledge Base" : "2. Upload your Security Questions";
  const descHtml =
    phase === "kb"
      ? <>Pick or drop your knowledge base <code>.json</code> file. Each item in the file must have its corresponding "id", "category", "question", and "answer" fields. Otherwise, the submission will fail. No data leaves your browser unless you click <em>Submit</em>.</>
      : <>Pick or drop your security questions <code>.json</code> file. Each item in the file must have its corresponding "id" and "text". Otherwise, submission will fail. No data leaves your browser unless you click <em>Submit</em>.</>;
  const detectedLabel = phase === "kb" ? "Questions & Answers:" : "Detected questions:";

  // All categories that exist in current answers
  const categories = useMemo(() => allCategories(answersRows), [answersRows]);

  // Filter by categories (OR semantics: match if a row has ANY of the selected categories)
  const filteredAnswers = useMemo(() => {
    if (!selectedCats.length) return answersRows;
    return answersRows.filter(r => {
      const cats = rowCategories(r);
      return cats.some(c => selectedCats.includes(c));
    });
  }, [answersRows, selectedCats]);

  function toTs(s?: string): number {
  if (!s) return 0;
  // Normalize "YYYY-MM-DD HH:mm:ss" → "YYYY-MM-DDTHH:mm:ss" so Date.parse is reliable
  const norm = s.includes("T") || s.endsWith("Z") ? s : s.replace(" ", "T");
  const t = Date.parse(norm);
  return Number.isNaN(t) ? 0 : t;
}

  function idNum(s?: string) {
    const m = String(s || "").match(/(\d+)$/);
    return m ? parseInt(m[1], 10) : NaN;
  }

  const visibleAnswers = useMemo(() => {
    const rows = [...filteredAnswers];
    rows.sort((a, b) => {
      switch (sortKey) {
        case "generated_at": {
          const da = toTs(a.generated_at);
          const db = toTs(b.generated_at);
          if (da !== db) return da - db;
          // tie-break by numeric ID if available
          const an = idNum(a.id), bn = idNum(b.id);
          if (!Number.isNaN(an) && !Number.isNaN(bn)) return an - bn;
          return String(a.id || "").localeCompare(String(b.id || ""));
        }
        case "id": {
          const an = idNum(a.id), bn = idNum(b.id);
          if (!Number.isNaN(an) && !Number.isNaN(bn)) return an - bn;
          return String(a.id || "").localeCompare(String(b.id || ""));
        }
        case "category":
        default: {
          const ca = rowCategories(a)[0] || "";
          const cb = rowCategories(b)[0] || "";
          return ca.localeCompare(cb);
        }
      }
    });
    return sortDir === "asc" ? rows : rows.reverse();
  }, [filteredAnswers, sortKey, sortDir]);



  /* ------------------ MANAGEMENT VIEW ------------------ */
  if (view === "management") {
    return (
      <div style={page}>
        <div style={card}>
          <h1 style={title}>Manage knowledge base & questions</h1>

          {/* Tabs */}
          <div style={tabsRow}>
            <button type="button" onClick={() => setActiveTab("kb")} style={activeTab === "kb" ? tabActive : tab}>
              Knowledge Base ({kbRows.length})
            </button>
            <button type="button" onClick={() => setActiveTab("questions")} style={activeTab === "questions" ? tabActive : tab}>
              Security Questions ({qRows.length})
            </button>
            {(answersRows.length > 0 || genStatus === "done" ) && (
              <button
                type="button"
                onClick={() => setActiveTab("results")}
                style={activeTab === "results" ? tabActive : tab}
              >
                Results ({answersRows.length})
              </button>
            )}
          </div>

          {/* View mode switch + contextual actions */}
          <div style={viewSwitchRow}>
            {activeTab !== "results" && (
              <>
                <span style={{ fontSize: 13, color: "#475569" }}>View:</span>
                <div style={segWrap}>
                  <button type="button" onClick={() => setViewMode("table")} style={viewMode === "table" ? segActive : seg}>Table</button>
                  <button type="button" onClick={() => setViewMode("cards")} style={viewMode === "cards" ? segActive : seg}>Cards</button>
                </div>
              </>
            )}

            {activeTab === "questions" && (
              <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
                <button
                  type="button"
                  onClick={startGenerateAnswers}
                  disabled={genStatus === "running"}
                  style={{ ...btnPrimary, background: "#7c3aed", opacity: genStatus === "running" ? 0.6 : 1 }}
                  title="Generate answers from your KB"
                >
                  {genStatus === "running" ? "Generating…" : "Generate Answers"}
                </button>
                {(answersRows.length > 0 || genStatus === "done") && (
                  <button
                    type="button"
                    onClick={async () => { await loadResults(); setActiveTab("results"); }}
                    style={{ ...btnGhost }}
                  >
                    View Results
                  </button>
                )}
              </div>
            )}

            {activeTab === "kb" && selectedCount > 0 && (
              <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
                <button
                  type="button"
                  onClick={() => {
                    if (canEdit) {
                      const id = Array.from(selected)[0];
                      const item = kbRows.find(r => r.id === id);
                      if (item) { setKbForm(item); setShowEdit(true); }
                    }
                  }}
                  disabled={!canEdit}
                  style={{ ...btnPrimary, background: canEdit ? "#7c3aed" : "#c4b5fd" }}
                  title={canEdit ? "Edit selected" : "Select exactly one to edit"}
                >
                  Edit
                </button>
                <button
                  type="button"
                  onClick={async () => {
                    const ids = Array.from(selected);
                    const ok = confirm(`Delete ${ids.length} entr${ids.length > 1 ? "ies" : "y"}?`);
                    if (!ok) return;
                    // optimistic UI
                    setKbRows(prev => prev.filter(r => !ids.includes(r.id)));
                    setSelected(new Set());
                    try {
                      await deleteKB(ids);
                    } catch (e: any) {
                      setError(e?.message ?? String(e));
                      // reload from server to reconcile
                      await loadManagementData();
                    }
                  }}
                  style={{ ...btnBase, background: "#fee2e2", color: "#991b1b", borderColor: "#fecaca" }}
                >
                  Delete ({selectedCount})
                </button>
              </div>
            )}
          </div>

          {/* Errors */}
          {error && (
            <div style={errorBox}>
              <strong>Oops:</strong>
              <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{error}</pre>
            </div>
          )}
          {genError && (
            <div style={{ ...errorBox, background: "#fff7ed", borderColor: "#fed7aa", color: "#9a3412" }}>
              <strong>Generate Answers error:</strong>
              <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{genError}</pre>
            </div>
          )}

          {/* Content */}
          {activeTab === "kb" ? (
            viewMode === "table" ? (
              <>
                {/* Stats above the table (KB only) */}
                <div style={chartCard}>
                  <div style={chartTitle}>KB by Category — {kbStats.total} total</div>
                  <div style={chartRow}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                        <Pie
                          data={kbStats.rows}
                          dataKey="count"
                          nameKey="category"
                          cx="50%"
                          cy="50%"
                          innerRadius={70}
                          outerRadius={110}
                          paddingAngle={1}
                          label={(p: PieLabelRenderProps) => {
                            const name  = String(p.name ?? "");
                            const value = typeof p.value === "number" ? p.value : Number(p.value);
                            const ratio = typeof p.percent === "number" ? p.percent : 0; // 0..1
                            if (!value || ratio < 0.06) return "";
                            return `${name}: ${value} (${Math.round(ratio * 100)}%)`;
                          }}
                          labelLine={false}
                        >
                          {kbStats.rows.map((_, i) => (
                            <Cell key={i} fill={PURPLES[i % PURPLES.length]} />
                          ))}
                        </Pie>
                        <Tooltip
                          formatter={(value: any, _name: any, { payload }: any) => {
                            const p = payload?.pct ?? 0;
                            return [`${value} (${p}%)`, "Count"];
                          }}
                        />
                        <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" style={{ fontWeight: 800, fill: "#6d28d9", fontSize: 18 }}>
                          {kbStats.total}
                        </text>
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Legend */}
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 8, marginTop: 10 }}>
                    {kbStats.rows.map((r, i) => (
                      <div key={r.category} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: "#334155" }}>
                        <span style={{ width: 10, height: 10, borderRadius: 3, background: PURPLES[i % PURPLES.length], border: "1px solid rgba(0,0,0,0.06)" }} />
                        <span style={{ fontWeight: 800, color: "#6d28d9" }}>{r.category}</span>
                        <span>— {r.count} ({r.pct}%)</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Table */}
                <div style={tableWrap}>
                  <table style={tbl}>
                    <thead>
                      <tr>
                        <th style={thCheck}>
                          <input type="checkbox" checked={allChecked} onChange={toggleAll} />
                        </th>
                        <th style={thId}>ID</th>
                        <th style={th}>Category</th>
                        <th style={th}>Question</th>
                        <th style={th}>Answer</th>
                        <th style={th}>Created</th>
                        <th style={th}>Updated</th>
                      </tr>
                    </thead>
                    <tbody>
                      {kbRows.map((r: any) => (
                        <tr key={r.id}>
                          <td style={tdCheck}>
                            <input 
                              type="checkbox" 
                              checked={selected.has(r.id)} 
                              onChange={() => toggleOne(r.id)} 
                            />
                          </td>
                          <td style={tdId}>
                            <span style={badge}>{r.id}</span>
                          </td>
                          <td style={td}><span style={chip}>{r.category}</span></td>
                          <td style={tdWrap}>{r.question}</td>
                          <td style={tdWrap}>{r.answer}</td>
                          <td style={td}>{fmt(r.created_at)}</td>
                          <td style={td}>{fmt(r.updated_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <>
                <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 8 }}>
                  <label style={{ fontSize: 13, color: "#334155", display: "flex", alignItems: "center", gap: 8 }}>
                    <input type="checkbox" checked={allChecked} onChange={toggleAll} />
                    Select all
                  </label>
                </div>
                <div style={cardGrid}>
                  {kbRows.map((r) => (
                    <div key={r.id} style={kbCard}>
                      <div style={cardHead}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <input type="checkbox" checked={selected.has(r.id)} onChange={() => toggleOne(r.id)} />
                          <span style={badge}>{r.id}</span>
                        </div>
                        <span style={chip}>{r.category}</span>
                      </div>
                      <div style={qaBlock}>
                        <div style={qaLabelQ}>Question</div>
                        <div style={qaText}>{r.question}</div>
                      </div>
                      <div style={qaBlock}>
                        <div style={qaLabelA}>Answer</div>
                        <div style={qaText}>{r.answer}</div>
                      </div>
                      <div style={{ marginTop:8, fontSize: 12, color: "#64748b"}}>
                        <div><strong>Created:</strong> {fmt(r.created_at)}</div>
                        <div><strong>Updated:</strong> {fmt(r.updated_at)}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )
          ) : activeTab === "questions" ? (
            viewMode === "table" ? (
              <div style={tableWrap}>
                <table style={tbl}>
                  <thead>
                    <tr>
                      <th style={th}>ID</th>
                      <th style={th}>Text</th>
                    </tr>
                  </thead>
                  <tbody>
                    {qRows.map((r) => (
                      <tr key={r.id}>
                        <td style={tdMono}><span style={badge}>{r.id}</span></td>
                        <td style={tdWrap}>{r.text}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={cardGrid}>
                {qRows.map((r) => (
                  <div key={r.id} style={questionCard}>
                    <div style={cardHead}>
                      <span style={badge}>{r.id}</span>
                    </div>
                    <div style={qaBlock}>
                      <div style={qaLabelQ}>Question</div>
                      <div style={qaText}>{r.text}</div>
                    </div>
                  </div>
                ))}
              </div>
            )
          ) : (
            // RESULTS TAB (table only)
            <div style={tableWrap}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "8px 12px",
                  gap: 8,
                  flexWrap: "wrap",
                }}
              >
                <div style={{ fontWeight: 800, color: "#7c3aed" }}>Generated Answers</div>

                <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: "auto", flexWrap: "wrap" }}>
                  {/* Sort controls */}
                  <label style={{ fontSize: 12, color: "#334155", display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ fontWeight: 800 }}>Sort</span>
                    <select
                      value={sortKey}
                      onChange={(e) => setSortKey(e.target.value as SortKey)}
                      style={{ border: "1px solid #ddd6fe", borderRadius: 8, padding: "6px 8px", fontWeight: 700, color: "#5b21b6" }}
                    >
                      <option value="generated_at">Generated time</option>
                      <option value="id">Question ID</option>        
                      <option value="category">Category</option>
                    </select>
                    <button
                      type="button"
                      onClick={() => setSortDir((d) => (d === "asc" ? "desc" : "asc"))}
                      style={{ ...btnBase, background: "white", borderColor: "#ddd6fe", color: "#6d28d9", borderRadius: 8 }}
                      title="Toggle sort direction"
                    >
                      {sortDir === "asc" ? "↑" : "↓"}
                    </button>
                  </label>

                  {/* Category filter */}
                  <div style={{ position: "relative" }}>
                    <button
                      type="button"
                      onClick={() => setShowCatPicker((v) => !v)}
                      style={{ ...btnBase, background: "white", borderColor: "#ddd6fe", color: "#6d28d9", borderRadius: 8 }}
                      aria-expanded={showCatPicker}
                    >
                      Filter Categories {selectedCats.length ? `(${selectedCats.length})` : ""}
                    </button>

                    {showCatPicker && (
                      <div style={catPanel}>
                        {categories.length === 0 ? (
                          <div style={{ fontSize: 12, color: "#64748b" }}>No categories found.</div>
                        ) : (
                          <div style={{ display: "grid", gap: 6, maxHeight: 220, overflow: "auto" }}>
                            {categories.map((c) => {
                              const checked = selectedCats.includes(c);
                              return (
                                <label key={c} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: "#334155" }}>
                                  <input
                                    type="checkbox"
                                    checked={checked}
                                    onChange={(e) => {
                                      setSelectedCats((prev) => (e.target.checked ? [...prev, c] : prev.filter((x) => x !== c)));
                                    }}
                                  />
                                  <span style={{ ...chip, padding: "2px 8px" }}>{c}</span>
                                </label>
                              );
                            })}
                          </div>
                        )}

                        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
                          <button type="button" onClick={() => setSelectedCats([])} style={{ ...btnGhost, padding: "6px 10px" }}>
                            Clear
                          </button>
                          <button
                            type="button"
                            onClick={() => setShowCatPicker(false)}
                            style={{ ...btnPrimary, background: "#7c3aed", padding: "6px 10px" }}
                          >
                            Done
                          </button>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Re-run */}
                  <button
                    type="button"
                    onClick={startGenerateAnswers}
                    disabled={genStatus === "running"}
                    style={{ ...btnPrimary, background: "#7c3aed", opacity: genStatus === "running" ? 0.6 : 1 }}
                    title="Re-run answer generation"
                  >
                    {genStatus === "running" ? "Regenerating..." : "Regenerate"}
                  </button>
                </div>
              </div>

              <table style={tbl}>
                <thead>
                  <tr>
                    <th style={th}>Question ID</th>
                    <th style={th}>Question</th>
                    <th style={th}>Answer</th>
                    <th style={th}>Categories</th>
                    <th style={th}>Generated</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleAnswers.map((r, idx) => {
                    const rowKey = `${r.id || "row"}-${idx}`;
                    const blocks = parseContextBlocks(r.context);
                    const isOpen = !!expanded[rowKey];
                    const cats = rowCategories(r);

                    return (
                      <React.Fragment key={rowKey}>
                        {/* Main row */}
                        <tr>
                          <td style={tdMono}><span style={badge}>{r.id || "—"}</span></td>
                          <td style={tdWrap}>{r.question || "—"}</td>
                          <td style={tdWrap}>{r.answer || "—"}</td>
                          <td style={{ ...td, minWidth: 180 }}>
                            {cats.length ? cats.map((c) => <span key={c} style={{ ...chip, marginRight: 6 }}>{c}</span>) : "—"}
                          </td>
                          <td style={td}>{fmt(r.generated_at)}</td>
                        </tr>

                        {/* Context row spans full width */}
                        <tr>
                          <td colSpan={5} style={tdCtxFull}>
                            <div style={ctxBar}>
                              <div style={{ fontWeight: 800, color: "#334155" }}>Context</div>
                              <button
                                type="button"
                                onClick={() => toggleContext(rowKey)}
                                style={ctxToggle}
                                aria-expanded={isOpen}
                              >
                                {isOpen ? "Hide" : "Show"}
                              </button>
                            </div>

                            {isOpen && (
                              <>
                                {blocks.length > 0 ? (
                                  blocks.map((b, i) => (
                                    <div key={i} style={ctxCard}>
                                      <div style={ctxHeader}>
                                        {Array.isArray(r.fused_parent_ids) && r.fused_parent_ids[i] && (
                                          <span style={badge}>{r.fused_parent_ids[i]}</span>
                                        )}
                                        {b.category && b.category.trim().toUpperCase() !== "GLOBAL" && (
                                          <span style={ctxTitle}>{b.category}</span>
                                        )}
                                      </div>
                                      {b.q && (
                                        <div style={ctxQA}>
                                          <span style={ctxLabel}>Q</span> {b.q}
                                        </div>
                                      )}
                                      {b.a && (
                                        <div style={ctxQA}>
                                          <span style={{ ...ctxLabel, background: "#7c3aed", color: "white" }}>A</span> {b.a}
                                        </div>
                                      )}
                                    </div>
                                  ))
                                ) : (
                                  "—"
                                )}
                              </>
                            )}
                          </td>
                        </tr>
                      </React.Fragment>
                    );
                  })}
                  {visibleAnswers.length === 0 && (
                    <tr>
                      <td style={td} colSpan={5}>
                        {selectedCats.length
                          ? "No results for the selected categories."
                          : <>No results yet. Click <em>Regenerate</em> in the Security Questions tab.</>}
                      </td>
                    </tr>
                  )}
                </tbody>



              </table>
            </div>
          )}

          <div style={{ ...row, justifyContent: "flex-end" }}>
            <button
              type="button"
              onClick={async () => {
                await clearAnswers();          // <— delete answers.json + reset Results UI
                setView("upload");
                setPhase("kb");
                reset();
              }}
              style={btnGhost}
            >
              ← Back to upload
            </button>
          </div>
        </div>

        {/* Floating Add Button (KB only) */}
        {activeTab === "kb" && (
          <div style={fabWrap}>
            {fabOpen && (
              <div style={fabMenu}>
                <button
                  type="button"
                  onClick={() => { setKbForm(emptyKB); setShowAddOne(true); setFabOpen(false); }}
                  style={fabItem}
                >
                  + Add entry
                </button>
                <button
                  type="button"
                  onClick={() => { setShowAddMany(true); setFabOpen(false); }}
                  style={fabItem}
                >
                  ⬆ Add multiple (.json)
                </button>
              </div>
            )}
            <button type="button" onClick={() => setFabOpen(v => !v)} style={fabBtn}>
              {fabOpen ? "×" : "+"}
            </button>
          </div>
        )}

        {/* Modals */}
        {showAddOne && (
          <Modal title="Add KB entry" onClose={() => setShowAddOne(false)}>
            <KBForm
              mode="create"
              value={kbForm}
              onChange={setKbForm}
              submitText="Create"
              onSubmit={async (v) => {
                try {
                  // AUTO-ID strictly as kb_<n>, where n = max(existing)+1
                  const id = getNextKbId(kbRows);
                  const payload: KBItem = { ...v, id };
                  await createKB(payload);
                  // optimistic: insert locally, then reconcile
                  setKbRows(prev => [...prev, payload]);
                  setShowAddOne(false);
                  await loadManagementData();
                } catch (e: any) {
                  alert(e?.message ?? String(e));
                }
              }}
            />
          </Modal>
        )}

        {showEdit && (
          <Modal title="Edit KB entry" onClose={() => setShowEdit(false)}>
            <KBForm
              mode="edit"
              value={kbForm}
              onChange={setKbForm}
              submitText="Save"
              onSubmit={async (v) => {
                try {
                  // keep original id; do not allow editing id
                  await updateKB(v);
                  // optimistic: update locally
                  setKbRows(prev => prev.map(row => row.id === v.id ? v : row));
                  setShowEdit(false);
                  await loadManagementData();
                  setSelected(new Set());
                } catch (e: any) {
                  alert(e?.message ?? String(e));
                }
              }}
            />
          </Modal>
        )}

        {showAddMany && (
          <Modal title="Append multiple KB entries (.json)" onClose={() => setShowAddMany(false)}>
            <AppendJson
              onSubmit={async (json) => {
                try {
                  let arr: any = [];
                  try { arr = JSON.parse(json); } catch { /* handled below */ }
                  if (!Array.isArray(arr)) {
                    alert("JSON must be an array of KB items."); return;
                  }
                  // Determine starting n from current rows
                  const startId = getNextKbId(kbRows);       // e.g., "kb_12"
                  const startN  = parseInt((KB_ID_RE.exec(startId)?.[1]) ?? "1", 10);

                  let n = startN;
                  const normalized: KBItem[] = arr.map((x: any) => ({
                    id: `kb_${n++}`,                         // sequential ids kb_<previous+1>...
                    category: String(x?.category ?? ""),
                    question: String(x?.question ?? ""),
                    answer: String(x?.answer ?? ""),
                  }));

                  // append via resilient endpoint(s)
                  await tryRequests([
                    () => fetch(ENDPOINTS.kbAppend,         { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(normalized) }),
                    () => fetch(ENDPOINTS.kbCreatePrimary,   { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(normalized) }),
                  ]);
                  // optimistic merge
                  setKbRows(prev => [...prev, ...normalized]);
                  setShowAddMany(false);
                  await loadManagementData();
                } catch (e: any) {
                  alert(e?.message ?? String(e));
                }
              }}
            />
          </Modal>
        )}

        {/* Generation overlay */}
        {genStatus === "running" && (
          <div style={fullscreenSlate} aria-live="polite" aria-busy="true">
            <div style={slateCard}>
              <Spinner />
              <span style={{ fontWeight: 700 }}>Generating answers…</span>
            </div>
          </div>
        )}

        <footer style={footer}>
          <p style={{ margin: 0, fontSize: 12, color: "#64748b" }}>
            Data shown here is read from your backend (GET <code>/api/kb</code>, <code>/api/questions</code>, <code>/api/answers</code>).
          </p>
        </footer>
      </div>
    );
  }

  /* ------------------ UPLOAD VIEW ------------------ */
  return (
    <div style={page}>
      <div style={card}>
        <h1 style={title}>{titleText}</h1>
        <p style={muted}>{descHtml}</p>

        <div
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          style={{
            ...dropzone,
            borderColor: dragActive ? "#2563eb" : "#cbd5e1",
            background: dragActive ? "#eff6ff" : "#f8fafc",
          }}
        >
          <div style={{ textAlign: "center" }}>
            <p style={{ margin: 0, fontWeight: 600 }}>Drag & drop JSON here</p>
            <p style={{ margin: 4, color: "#475569" }}>or</p>
            <div>
              <input
                ref={inputRef}
                id={inputId}
                type="file"
                accept=".json,application/json"
                onChange={onInputChange}
                style={{ display: "none" }}
              />
              <label htmlFor={inputId} style={{ ...btnPrimary, display: "inline-block", marginTop: 6 }}>
                Choose file
              </label>
            </div>
            <p style={{ marginTop: 8, fontSize: 12, color: "#64748b" }}>
              Max size {(MAX_BYTES / (1024 * 1024)).toFixed(0)} MiB
            </p>
          </div>
        </div>

        {status !== "idle" && (
          <section style={{ marginTop: 16 }}>
            {file && (
              <div style={fileInfo}>
                <div><strong>File:</strong> {file.name}</div>
                <div><strong>Size:</strong> {(file.size / 1024).toFixed(1)} KiB</div>
                {countSummary && <div><strong>{detectedLabel}</strong> {countSummary}</div>}
              </div>
            )}

            {error && (
              <div style={errorBox}>
                <strong>Oops:</strong>
                <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{error}</pre>
              </div>
            )}

            {rawText && (
              <details open style={detailsBox}>
                <summary style={{ cursor: "pointer", fontWeight: 600 }}>Preview</summary>
                <pre style={pre}>{pretty(JSON.stringify(data, null, 2))}</pre>
              </details>
            )}
          </section>
        )}

        <div style={row}>
          {phase === "questions" && (
            <button type="button" onClick={goBackToKB} disabled={status === "submitting"} style={btnGhost}>
              ← Back to Knowledge Base
            </button>
          )}

          <button onClick={submit} disabled={!rawText || status === "submitting"} style={{ ...btnPrimary, opacity: !rawText ? 0.6 : 1 }}>
            {status === "submitting" ? "Submitting…" : "Submit"}
          </button>
          <button onClick={reset} style={btnGhost}>Clear</button>
        </div>
      </div>

      {(status === "submitting") && (
        <div style={fullscreenSlate} aria-live="polite" aria-busy="true">
          <div style={slateCard}>
            <Spinner />
            <span style={{ fontWeight: 700 }}>
              {phase === "kb" ? "Uploading knowledge base…" : "Uploading security questions…"}
            </span>
          </div>
        </div>
      )}

      {status === "success" && phase === "questions" && (
        <div style={successSlate} role="dialog" aria-labelledby="allset-title">
          <div style={successCard}>
            <svg width="56" height="56" viewBox="0 0 24 24" aria-hidden="true">
              <circle cx="12" cy="12" r="10" fill="#22c55e" opacity="0.15" />
              <path d="M9.5 12.5l2 2 4-5" fill="none" stroke="#16a34a" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <div>
              <div id="allset-title" style={{ fontWeight: 800, fontSize: 20 }}>You’re all set!</div>
              <div style={{ fontSize: 13, color: "#475569" }}>Security questions uploaded successfully.</div>
            </div>
          </div>
        </div>
      )}

      <footer style={footer}>
        <p style={{ margin: 0, fontSize: 12, color: "#64748b" }}>
          Your file stays in your browser until you successfully submit it. Then it's sent to the server for processing.
        </p>
      </footer>
    </div>
  );
}

/* ---------- Reusable Modals / Forms ---------- */
function Modal(props: { title: string; onClose: () => void; children: React.ReactNode }) {
  return (
    <div style={fullscreenSlate}>
      <div style={modalCard}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
          <div style={{ fontWeight: 800, color: "#6d28d9", fontSize: 18 }}>{props.title}</div>
          <button onClick={props.onClose} style={{ ...btnBase, background: "#f1f5f9" }}>×</button>
        </div>
        {props.children}
      </div>
    </div>
  );
}

function KBForm(props: {
  mode: "create" | "edit";
  value: KBItem;
  onChange: (v: KBItem) => void;
  onSubmit: (v: KBItem) => void | Promise<void>;
  submitText: string;
}) {
  const v = props.value;
  const set = (k: keyof KBItem) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
    props.onChange({ ...v, [k]: e.target.value });

  return (
    <form
      onSubmit={async (e) => {
        e.preventDefault();
        // Require K, C, Q, A; ID handled automatically on create; immutable on edit
        if (!v.category || !v.question || !v.answer) { alert("Category, Question and Answer are required."); return; }
        await props.onSubmit(v);
      }}
      style={{ display: "grid", gap: 10 }}
    >
      {props.mode === "edit" && (
        <div style={{ fontSize: 12, color: "#475569" }}>
          <strong>ID:</strong> <code>{v.id}</code> (read-only)
        </div>
      )}
      {/* No ID input on create to avoid user needing to assign ID */}
      <label style={label}>Category<input style={input} value={v.category} onChange={set("category")} placeholder="Governance & Risk Management" /></label>
      <label style={label}>Question<textarea style={textarea} value={v.question} onChange={set("question")} placeholder="Does the organization…?" /></label>
      <label style={label}>Answer<textarea style={textarea} value={v.answer} onChange={set("answer")} placeholder="Yes, we…" /></label>

      <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 6 }}>
        <button type="submit" style={{ ...btnPrimary, background: "#7c3aed" }}>{props.submitText}</button>
      </div>
    </form>
  );
}

function AppendJson(props: { onSubmit: (json: string) => void | Promise<void> }) {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [text, setText] = useState("");
  const [err, setErr] = useState("");

  const handleFile = async (f: File) => {
    setErr("");
    try {
      if (!looksLikeJsonFile(f)) { setErr("Please choose a .json file."); return; }
      const t = await f.text();
      JSON.parse(t); // sanity parse
      setText(t);
    } catch (e: any) { setErr(`Invalid JSON: ${e?.message ?? e}`); }
  };

  return (
    <div style={{ display: "grid", gap: 10 }}>
      <input
        ref={fileRef}
        type="file"
        accept=".json,application/json"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) void handleFile(f); }}
      />
      {err && <div style={{ ...errorBox, marginTop: 0 }}><strong>Oops:</strong> {err}</div>}
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <button
          type="button"
          onClick={() => props.onSubmit(text)}
          disabled={!text}
          style={{ ...btnPrimary, background: "#7c3aed", opacity: text ? 1 : 0.6 }}
        >
          Append
        </button>
      </div>
    </div>
  );
}

// Parse "[1] Category\nQ: ...\nA: ..." blocks into structured parts
function parseContextBlocks(ctx?: string): Array<{ category?: string; q?: string; a?: string }> {
  if (!ctx) return [];
  const out: Array<{ category?: string; q?: string; a?: string }> = [];
  const re = /\[(\d+)\]\s*([^\n]*?)\s*\nQ:\s*([\s\S]*?)\nA:\s*([\s\S]*?)(?=\n\[\d+\]\s|$)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(ctx)) !== null) {
    out.push({
      category: (m[2] || "").trim() || undefined,
      q:        (m[3] || "").trim() || undefined,
      a:        (m[4] || "").trim() || undefined,
    });
  }
  // Fallback: nothing matched, show raw as one block
  return out.length ? out : [{ q: ctx }];
}


/* ====== Styles ====== */
const page: React.CSSProperties = {
  minHeight: "100dvh",
  background: "#0f172a",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  padding: 24,
  gap: 16,
  fontFamily:
    '"Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Apple Color Emoji", "Segoe UI Emoji"',
};

const card: React.CSSProperties = {
  width: "min(900px, 100%)",
  background: "white",
  borderRadius: 16,
  boxShadow: "0 10px 30px rgba(0,0,0,0.2)",
  padding: "24px 23px 23px",   
  margin: "0 auto",
};
const title: React.CSSProperties = {
  margin: 0,
  marginBottom: 8,
  paddingTop: 5,              
  lineHeight: 1.4,           
  fontSize: 38,
  fontWeight: 800,
  letterSpacing: 0.4,
  display: "block",            
  backgroundImage: "linear-gradient(90deg, #06b6d4, #6366f1 45%, #22c55e)",
  WebkitBackgroundClip: "text",
  backgroundClip: "text",
  color: "transparent",
  WebkitTextFillColor: "transparent",
};
const muted: React.CSSProperties = { marginTop: 4, color: "#475569" };
const dropzone: React.CSSProperties = {
  border: "2px dashed #cbd5e1",
  borderRadius: 14,
  padding: 40,
  background: "#f8fafc",
  transition: "all 160ms ease",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: 200,
  textAlign: "center",
};
const row: React.CSSProperties = { display: "flex", gap: 10, marginTop: 18, flexWrap: "wrap" };
const btnBase: React.CSSProperties = {
  border: "1px solid transparent",
  borderRadius: 10,
  padding: "10px 14px",
  fontWeight: 600,
  cursor: "pointer",
};

const btnPrimary: React.CSSProperties = { ...btnBase, background: "#2563eb", color: "white" };
const btnGhost: React.CSSProperties = { ...btnBase, background: "transparent", borderColor: "#cbd5e1", color: "#0f172a" };
const pre: React.CSSProperties = {
  margin: 0, padding: 12, overflow: "auto",
  background: "#0b1020", color: "#e2e8f0", borderRadius: 12, fontSize: 13, lineHeight: 1.4, maxHeight: 420,
};
const fileInfo: React.CSSProperties = {
  display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))",
  gap: 8, background: "#f1f5f9", borderRadius: 12, padding: 12, fontSize: 14,
};

const errorBox: React.CSSProperties = {
  background: "#fef2f2",
  border: "1px solid #fecaca",
  color: "#991b1b",
  padding: 12,
  borderRadius: 12,
  marginTop: 10,
  whiteSpace: "pre-wrap",
};

const detailsBox: React.CSSProperties = {
  background: "#f8fafc",
  border: "1px solid #e2e8f0",
  borderRadius: 12,
  padding: 12,
  marginTop: 10,
};

const successSlate: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(15,23,42,0.6)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 110,
};

const successCard: React.CSSProperties = {
  background: "white",
  borderRadius: 16,
  boxShadow: "0 24px 48px rgba(0,0,0,0.35)",
  padding: "20px 24px",
  display: "flex",
  alignItems: "center",
  gap: 14,
};

const footer: React.CSSProperties = { width: "min(900px, 100%)", margin: "0 auto", marginTop: 16, textAlign: "center" };

const fullscreenSlate: React.CSSProperties = {
  position: "fixed", inset: 0, background: "rgba(15,23,42,0.6)",
  display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100,
};
const slateCard: React.CSSProperties = {
  background: "white", borderRadius: 14, boxShadow: "0 20px 40px rgba(0,0,0,0.35)",
  padding: "18px 22px", display: "flex", alignItems: "center", gap: 12,
};

const tabsRow: React.CSSProperties = {
  display: "flex", gap: 8, borderBottom: "1px solid #e2e8f0", marginTop: 8,
};
const tab: React.CSSProperties = {
  border: "1px solid transparent",
  background: "transparent",
  color: "#0f172a",
  borderBottom: "2px solid transparent",
  borderRadius: 0,
  padding: "10px 12px",
  fontWeight: 600,
  cursor: "pointer",
};
const tabActive: React.CSSProperties = {
  ...tab,
  borderBottomColor: "#7c3aed",
  fontWeight: 800,
};

const tableWrap: React.CSSProperties = {
  overflowX: "auto",
  marginTop: 10,
  border: "1px solid #ddd6fe",
  borderRadius: 14,
  boxShadow: "0 8px 24px rgba(124,58,237,.15)",
  background: "white",
};

const tbl: React.CSSProperties = {
  width: "100%",
  borderCollapse: "separate",
  borderSpacing: 0,
  fontSize: 14,
  lineHeight: 1.5,
};

const th: React.CSSProperties = {
  textAlign: "left",
  padding: "12px 12px",
  position: "sticky" as const,
  top: 0,
  background:  "#7c3aed",
  color: "white",
  fontWeight: 800,
  letterSpacing: 0.2,
  borderBottom: "1px solid #6d28d9",
  zIndex: 1,
};

// --- Sticky checkbox + ID columns (so the ID doesn't look lost in tall rows) ---
const thCheck: React.CSSProperties = {
  ...th,
  width: 44,
  left: 0,
  position: "sticky",
  zIndex: 3,          // over body cells
};

// ID header/cell stick next to checkbox
const thId: React.CSSProperties = {
  ...th,
  left: 44,           // width of the checkbox col
  position: "sticky",
  zIndex: 3,
};

const td: React.CSSProperties = {
  verticalAlign: "top",
  padding: "12px 12px",
  borderBottom: "1px solid #f1f5f9",
  color: "#0f172a",
};

const tdId: React.CSSProperties = {
  ...td,
  ...{
    left: 44,
    position: "sticky",
    zIndex: 2,
    background: "#faf5ff",        // light violet band for the ID column
    borderRight: "1px dashed #e9d5ff",
    color: "#4c1d95",
    whiteSpace: "nowrap",
    display: "flex",
    alignItems: "flex-start",     // keep the pill pinned to the top
    paddingTop: 10,
    paddingBottom: 10,
  },
};


const tdCheck: React.CSSProperties = {
  ...td,
  width: 44,
  left: 0,
  position: "sticky",
  zIndex: 2,
  background: "white",
  textAlign: "center",
};
const tdMono: React.CSSProperties = {
  ...td,
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
  background: "#faf5ff",
  color: "#4c1d95",
  borderRight: "1px dashed #e9d5ff",
  whiteSpace: "nowrap",
};

const tdWrap: React.CSSProperties = {
  ...td,
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
};

const viewSwitchRow: React.CSSProperties = {
  display: "flex", alignItems: "center", gap: 10, marginTop: 12, marginBottom: 8,
};
const segWrap: React.CSSProperties = {
  display: "inline-flex", border: "1px solid #ddd6fe", borderRadius: 9999, overflow: "hidden",
};
const segBase: React.CSSProperties = {
  border: "none", padding: "6px 12px", cursor: "pointer", fontWeight: 700, fontSize: 13,
};
const seg: React.CSSProperties = {
  ...segBase, background: "transparent", color: "#6d28d9",
};
const segActive: React.CSSProperties = {
  ...segBase, background: "#ede9fe", color: "#5b21b6",
};

const cardGrid: React.CSSProperties = {
  marginTop: 12,
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
  gap: 12,
};
const kbCard: React.CSSProperties = {
  background: "white",
  border: "1px solid #ddd6fe",
  borderRadius: 14,
  boxShadow: "0 6px 16px rgba(124,58,237,.10)",
  padding: 14,
};
const questionCard: React.CSSProperties = {
  ...kbCard,
};
const cardHead: React.CSSProperties = {
  display: "flex", gap: 8, alignItems: "center", justifyContent: "space-between", marginBottom: 8,
};
const qaBlock: React.CSSProperties = { marginTop: 6 };
const qaLabelQ: React.CSSProperties = {
  fontSize: 12, fontWeight: 800, color: "#6d28d9", marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.4,
};
const qaLabelA: React.CSSProperties = {
  ...qaLabelQ, color: "#7c3aed",
};
const qaText: React.CSSProperties = { fontSize: 14, color: "#0f172a", whiteSpace: "pre-wrap", wordBreak: "break-word" };

const chip: React.CSSProperties = {
  display: "inline-block",
  padding: "2px 10px",
  borderRadius: 9999,
  background: "#ede9fe",
  color: "#6d28d9",
  fontWeight: 700,
  fontSize: 12,
};

const badge: React.CSSProperties = {
  display: "inline-block",
  padding: "2px 8px",
  borderRadius: 8,
  background: "#f5f3ff",
  color: "#6d28d9",
  fontWeight: 700,
  fontSize: 12,
  border: "1px solid #ddd6fe",
};

const chartCard: React.CSSProperties = {
  marginBottom: 12,
  border: "1px solid #ddd6fe",
  borderRadius: 14,
  boxShadow: "0 8px 24px rgba(124,58,237,.15)",
  background: "white",
  padding: 12,
};
const chartTitle: React.CSSProperties = {
  fontWeight: 800, color: "#7c3aed", marginBottom: 8,
};
const chartRow: React.CSSProperties = {
  width: "100%", height: 320,
};

const modalCard: React.CSSProperties = {
  background: "white",
  borderRadius: 16,
  boxShadow: "0 24px 48px rgba(0,0,0,0.35)",
  padding: 16,
  width: "min(560px, 92vw)",
};
const fabWrap: React.CSSProperties = {
  position: "fixed",
  right: 24,
  bottom: 24,
  display: "flex",
  flexDirection: "column",
  alignItems: "flex-end",
  gap: 8,
  zIndex: 200,
};
const fabBtn: React.CSSProperties = {
  width: 56,
  height: 56,
  borderRadius: "50%",
  border: "none",
  color: "white",
  background: "#7c3aed",
  fontSize: 28,
  fontWeight: 800,
  cursor: "pointer",
  boxShadow: "0 10px 20px rgba(124,58,237,.35)",
};
const fabMenu: React.CSSProperties = {
  display: "grid",
  gap: 6,
  marginBottom: 6,
};
const fabItem: React.CSSProperties = {
  ...btnBase,
  background: "white",
  borderColor: "#ddd6fe",
  color: "#6d28d9",
  fontWeight: 800,
};

const label: React.CSSProperties = {
  display: "grid",
  gap: 6,
  fontSize: 12,
  fontWeight: 800,
  color: "#334155",
};
const input: React.CSSProperties = {
  border: "1px solid #e2e8f0",
  borderRadius: 10,
  padding: "10px 12px",
  fontSize: 14,
  outline: "none",
};
const textarea: React.CSSProperties = {
  ...input,
  minHeight: 90,
  resize: "vertical",
};

const tdCtxFull: React.CSSProperties = {
  ...td,
  background: "#faf5ff",
  borderTop: "1px dashed #e9d5ff",
};

const ctxCard: React.CSSProperties = {
  background: "#f8fafc",
  border: "1px solid #e2e8f0",
  borderRadius: 12,
  padding: 10,
  marginTop: 8,
};



const ctxHeader: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  marginBottom: 6,
};

const ctxTitle: React.CSSProperties = {
  fontWeight: 800,
  color: "#334155",
};

const ctxQA: React.CSSProperties = {
  fontSize: 14,
  color: "#0f172a",
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
  marginTop: 4,
};

const ctxLabel: React.CSSProperties = {
  display: "inline-block",
  minWidth: 18,
  textAlign: "center",
  fontSize: 12,
  fontWeight: 800,
  marginRight: 6,
  padding: "1px 6px",
  borderRadius: 6,
  background: "#ede9fe",
  color: "#5b21b6",
};

const ctxBar: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 8,
  marginBottom: 6,
};

const ctxToggle: React.CSSProperties = {
  ...btnBase,
  padding: "6px 10px",
  fontSize: 12,
  fontWeight: 800,
  background: "white",
  color: "#6d28d9",
  borderColor: "#ddd6fe",
  borderWidth: 1,
  borderStyle: "solid",
  borderRadius: 8,
};

const catPanel: React.CSSProperties = {
  position: "absolute",
  right: 0,
  marginTop: 6,
  zIndex: 50,
  background: "white",
  border: "1px solid #ddd6fe",
  boxShadow: "0 10px 24px rgba(124,58,237,.18)",
  borderRadius: 12,
  padding: 10,
  minWidth: 220,
};