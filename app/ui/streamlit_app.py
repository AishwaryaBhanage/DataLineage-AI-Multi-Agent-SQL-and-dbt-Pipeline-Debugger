"""
DataLineage AI — Streamlit UI
Run with:  streamlit run app/ui/streamlit_app.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from app.workflows.debug_pipeline import run_pipeline
from app.core.config import MANIFEST_PATH

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataLineage AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── global font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, #7c3aed44, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 32px; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.hero-sub {
    color: #94a3b8; font-size: 15px; margin: 0;
}

/* ── section header ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 28px 0 14px 0;
}
.section-number {
    width: 28px; height: 28px; border-radius: 8px;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white; font-size: 13px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.section-title {
    font-size: 18px; font-weight: 600; color: #f1f5f9; margin: 0;
}

/* ── input card ── */
.input-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.input-label {
    font-size: 12px; font-weight: 600; letter-spacing: .06em;
    color: #64748b; text-transform: uppercase; margin-bottom: 8px;
}

/* ── stat pill ── */
.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
.stat-pill {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 999px; padding: 6px 16px;
    font-size: 13px; color: #e2e8f0; display: flex; align-items: center; gap: 6px;
}
.stat-pill strong { color: #a78bfa; }

/* ── tag chip ── */
.chip {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 500; margin: 2px;
    background: #1e293b; color: #94a3b8; border: 1px solid #334155;
}
.chip.red   { background:#2d1515; color:#fca5a5; border-color:#7f1d1d; }
.chip.blue  { background:#0c1a2e; color:#93c5fd; border-color:#1e3a5f; }
.chip.green { background:#0d2318; color:#86efac; border-color:#14532d; }
.chip.purple{ background:#1a0d2e; color:#c4b5fd; border-color:#4c1d95; }

/* ── error banner ── */
.error-banner {
    background: linear-gradient(135deg, #1a0a0a, #2d1515);
    border: 1px solid #7f1d1d;
    border-radius: 12px; padding: 18px 22px; margin-bottom: 16px;
}
.error-type-badge {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    background: #7f1d1d; color: #fecaca;
    font-size: 11px; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; margin-bottom: 10px;
}
.error-meta { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 12px; }
.error-meta-item { font-size: 13px; color: #94a3b8; }
.error-meta-item span { color: #f1f5f9; font-weight: 600; }

/* ── rule hit card ── */
.rule-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.rule-card::before {
    content: ""; position: absolute;
    left: 0; top: 0; bottom: 0; width: 4px;
}
.rule-card.high::before  { background: #22c55e; }
.rule-card.med::before   { background: #f59e0b; }
.rule-card.low::before   { background: #ef4444; }
.rule-rank {
    font-size: 11px; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; margin-bottom: 6px;
}
.rule-title { font-size: 15px; font-weight: 600; color: #f1f5f9; margin-bottom: 8px; }
.rule-evidence {
    font-size: 13px; color: #94a3b8;
    background: #1e293b; border-radius: 6px;
    padding: 8px 12px; margin-bottom: 8px;
}
.rule-fix {
    font-size: 13px; color: #60a5fa;
    display: flex; align-items: flex-start; gap: 6px;
}

/* ── confidence bar ── */
.conf-bar-wrap {
    display: flex; align-items: center; gap: 10px; margin-top: 6px;
}
.conf-bar-bg {
    flex: 1; height: 6px; background: #1e293b; border-radius: 999px; overflow: hidden;
}
.conf-bar-fill { height: 100%; border-radius: 999px; }
.conf-pct { font-size: 13px; font-weight: 700; min-width: 36px; }

/* ── root cause hero ── */
.rc-banner {
    background: linear-gradient(135deg, #0d1a0d, #1a2e0d);
    border: 1px solid #166534;
    border-radius: 14px; padding: 24px 28px; margin-bottom: 20px;
}
.rc-banner.warn {
    background: linear-gradient(135deg, #1a150d, #2e1f0d);
    border-color: #92400e;
}
.rc-label {
    font-size: 11px; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #4ade80; margin-bottom: 8px;
}
.rc-banner.warn .rc-label { color: #fbbf24; }
.rc-text { font-size: 20px; font-weight: 700; color: #f0fdf4; line-height: 1.35; }
.rc-banner.warn .rc-text { color: #fefce8; }
.rc-score {
    display: inline-block; margin-top: 12px;
    padding: 4px 14px; border-radius: 999px;
    background: #166534; color: #bbf7d0; font-size: 13px; font-weight: 700;
}
.rc-banner.warn .rc-score { background: #92400e; color: #fde68a; }

/* ── sql diff ── */
.diff-label {
    font-size: 11px; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; margin-bottom: 8px;
}
.diff-label.broken { color: #f87171; }
.diff-label.fixed  { color: #4ade80; }

/* ── checklist item ── */
.check-item {
    display: flex; align-items: flex-start; gap: 10px;
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
    font-size: 14px; color: #e2e8f0;
}
.check-icon { color: #4ade80; font-size: 16px; flex-shrink: 0; margin-top: 1px; }

/* ── upstream columns table ── */
.col-table { width: 100%; border-collapse: collapse; margin-top: 8px; }
.col-table th {
    text-align: left; font-size: 11px; font-weight: 700;
    letter-spacing: .07em; text-transform: uppercase;
    color: #64748b; padding: 6px 10px;
    border-bottom: 1px solid #1e293b;
}
.col-table td {
    padding: 6px 10px; font-size: 13px; color: #cbd5e1;
    border-bottom: 1px solid #0f172a;
}
.col-table tr:last-child td { border-bottom: none; }
.col-missing { color: #f87171 !important; font-weight: 600; }

/* ── sidebar styling ── */
[data-testid="stSidebar"] {
    background: #0a0f1e;
    border-right: 1px solid #1e293b;
}
.sidebar-logo {
    font-size: 22px; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}
.sidebar-caption { font-size: 12px; color: #475569; margin-bottom: 20px; }

/* ── run button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 15px !important;
    padding: 14px !important; letter-spacing: .02em;
    transition: opacity .2s;
}
.stButton > button[kind="primary"]:hover { opacity: .88; }

/* ── success / info tweaks ── */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def conf_color(s: float) -> str:
    return "#22c55e" if s >= .85 else ("#f59e0b" if s >= .65 else "#ef4444")

def conf_label(s: float) -> str:
    return "HIGH" if s >= .85 else ("MEDIUM" if s >= .65 else "LOW")

def section(num: str, title: str):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">{num}</div>
        <p class="section-title">{title}</p>
    </div>""", unsafe_allow_html=True)

def chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{text}</span>'

def conf_bar(score: float) -> str:
    color = conf_color(score)
    pct   = int(score * 100)
    return f"""
    <div class="conf-bar-wrap">
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct}%;background:{color}"></div>
        </div>
        <span class="conf-pct" style="color:{color}">{pct}%</span>
    </div>"""


def render_lineage_graph(lineage: dict, broken_model: str, query_is_valid: bool = False):
    nodes = lineage.get("nodes", [])
    edges = lineage.get("edges", [])
    if not nodes:
        st.info("No lineage data.")
        return

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for e in edges:
        G.add_edge(e["from"], e["to"])

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.shell_layout(G)

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(nodes) * 1.4)))
    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#0a0f1e")

    node_colors, node_edge_colors, node_sizes = [], [], []
    for n in G.nodes:
        if n == broken_model:
            # Green when valid (fixed), red when broken
            if query_is_valid:
                node_colors.append("#059669")
                node_edge_colors.append("#6ee7b7")
            else:
                node_colors.append("#ef4444")
                node_edge_colors.append("#fca5a5")
            node_sizes.append(3000)
        elif n in lineage.get("upstream", []):
            node_colors.append("#7c3aed")
            node_edge_colors.append("#c4b5fd")
            node_sizes.append(2500)
        else:
            node_colors.append("#059669")
            node_edge_colors.append("#6ee7b7")
            node_sizes.append(2200)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           ax=ax, alpha=0.92, linewidths=2,
                           edgecolors=node_edge_colors)
    nx.draw_networkx_labels(G, pos, font_color="white",
                            font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#475569", arrows=True,
                           arrowsize=22, arrowstyle="-|>", width=2.2,
                           connectionstyle="arc3,rad=0.08", ax=ax,
                           min_source_margin=30, min_target_margin=30)

    legend_handles = [
        mpatches.Patch(facecolor="#7c3aed", edgecolor="#c4b5fd", label="Upstream model"),
        mpatches.Patch(facecolor="#ef4444", edgecolor="#fca5a5", label="Broken model") if not query_is_valid
            else mpatches.Patch(facecolor="#059669", edgecolor="#6ee7b7", label="Fixed model ✅"),
        mpatches.Patch(facecolor="#059669", edgecolor="#6ee7b7", label="Healthy model"),
    ]
    legend = ax.legend(handles=legend_handles, loc="lower center",
                       facecolor="#0f172a", edgecolor="#1e293b",
                       labelcolor="white", fontsize=9,
                       ncol=3, bbox_to_anchor=(0.5, -0.06))
    for text in legend.get_texts():
        text.set_color("#cbd5e1")

    ax.axis("off")
    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-logo">⬡ DataLineage AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">AI-powered dbt / SQL debugger</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**⚙️ Settings**")
    use_llm = st.toggle("Enable Claude reasoning", value=True,
                        help="Disable to run rule engine only (no API key needed)")

    st.divider()
    st.markdown("**📂 Quick load**")
    if st.button("▶  Load sample failure", use_container_width=True):
        st.session_state["sample_loaded"] = True

    st.divider()
    st.markdown("""
    <div style="font-size:12px;color:#475569;line-height:1.8">
    <b style="color:#64748b">Stack</b><br>
    sqlglot · networkx · dbt-duckdb<br>
    FastAPI · Claude API<br><br>
    <b style="color:#64748b">Tests</b>&nbsp; 60 passing ✅
    </div>
    """, unsafe_allow_html=True)


# ── hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <p class="hero-title">⬡ DataLineage AI</p>
    <p class="hero-sub">
        Paste a broken dbt model + error message → get lineage reconstruction,
        root-cause analysis, and a corrected SQL in seconds.
    </p>
</div>
""", unsafe_allow_html=True)

# ── sample fill ───────────────────────────────────────────────────────────────

default_sql = default_error = default_manifest = ""
if st.session_state.get("sample_loaded"):
    try:
        default_sql      = Path("dbt_demo/models/customer_revenue.sql").read_text()
        default_error    = Path("data/sample_errors/customer_revenue_error.txt").read_text()
        default_manifest = str(Path("dbt_demo/target/manifest.json").resolve())
        st.success("Sample loaded — click **Analyze** to run.")
    except FileNotFoundError:
        st.warning("Sample files not found. Run `dbt run` inside dbt_demo/ first.")

# ── inputs ────────────────────────────────────────────────────────────────────

section("1", "Input")

col_sql, col_err = st.columns(2, gap="medium")
with col_sql:
    st.markdown('<div class="input-label">Broken SQL / dbt model</div>', unsafe_allow_html=True)
    sql_input = st.text_area("sql", value=default_sql, height=200,
                             placeholder="select customer_id, sum(amount)\nfrom {{ ref('stg_orders') }}\ngroup by customer_id",
                             label_visibility="collapsed")

with col_err:
    st.markdown('<div class="input-label">dbt / warehouse error message</div>', unsafe_allow_html=True)
    error_input = st.text_area("error", value=default_error, height=200,
                               placeholder='Runtime Error in model customer_revenue\nBinder Error: Referenced column "amount" not found',
                               label_visibility="collapsed")

st.markdown('<div class="input-label" style="margin-top:8px">manifest.json path or upload</div>', unsafe_allow_html=True)
m_col, u_col = st.columns([3, 1], gap="small")
with m_col:
    manifest_path_input = st.text_input("manifest", label_visibility="collapsed",
                                        value=default_manifest or str(Path(MANIFEST_PATH).resolve()),
                                        placeholder="./dbt_demo/target/manifest.json")
with u_col:
    uploaded = st.file_uploader("upload", type="json", label_visibility="collapsed")

manifest_path = manifest_path_input
if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(uploaded.read())
    tmp.flush()
    manifest_path = tmp.name
    st.success(f"Uploaded manifest ({uploaded.size // 1024} KB)")

st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("🚀  Analyze pipeline failure", type="primary", use_container_width=True)

# ── run ───────────────────────────────────────────────────────────────────────

if run_btn:
    if not sql_input.strip():
        st.error("Please paste a SQL model.")
        st.stop()
    if not error_input.strip():
        st.error("Please paste an error message.")
        st.stop()

    with st.spinner("Running pipeline…"):
        try:
            result = run_pipeline(sql=sql_input, error_text=error_input,
                                  manifest_path=manifest_path, use_llm=use_llm)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    # ── top status banner ─────────────────────────────────────────────────────
    if result.query_is_valid:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#052e16,#14532d);
                    border:1px solid #166534;border-radius:14px;
                    padding:22px 28px;margin-bottom:8px">
            <div style="font-size:11px;font-weight:700;letter-spacing:.1em;
                        color:#4ade80;text-transform:uppercase;margin-bottom:6px">
                ✅ Query is valid
            </div>
            <div style="font-size:19px;font-weight:700;color:#f0fdf4">
                No issues found — your SQL looks correct
            </div>
            <div style="font-size:13px;color:#86efac;margin-top:8px">
                All columns used in this query exist in the upstream models.
                The error message you pasted may be from a previous run before the fix was applied.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        top_hit  = result.rule_hits[0] if result.rule_hits else None
        top_conf = f"{top_hit.confidence:.0%}" if top_hit else "—"
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a0505,#2d1515);
                    border:1px solid #7f1d1d;border-radius:14px;
                    padding:22px 28px;margin-bottom:8px">
            <div style="font-size:11px;font-weight:700;letter-spacing:.1em;
                        color:#f87171;text-transform:uppercase;margin-bottom:6px">
                💥 Pipeline failure detected
            </div>
            <div style="font-size:19px;font-weight:700;color:#fef2f2">
                Issues found — see analysis below
            </div>
        </div>
        """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Model",      result.broken_model)
    s2.metric("Error type", result.parsed_error.error_type.value.replace("_"," ").title())
    s3.metric("Rule hits",  len(result.rule_hits) if not result.query_is_valid else 0)
    s4.metric("Status",     "✅ Valid" if result.query_is_valid else "❌ Broken")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2: parsed SQL ─────────────────────────────────────────────────────────
    section("2", "Parsed SQL Structure")
    p = result.parsed_sql

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill">Tables <strong>{len(p.tables)}</strong></div>
        <div class="stat-pill">Columns <strong>{len(p.columns)}</strong></div>
        <div class="stat-pill">Aggregations <strong>{len(p.aggregations)}</strong></div>
        <div class="stat-pill">dbt refs <strong>{len(p.dbt_refs)}</strong></div>
        <div class="stat-pill">CTEs <strong>{len(p.ctes)}</strong></div>
    </div>""", unsafe_allow_html=True)

    with st.expander("View all extracted entities"):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown("**Tables / refs**")
            st.markdown("".join(chip(t, "blue") for t in p.tables), unsafe_allow_html=True)
            st.markdown("<br>**Columns used**", unsafe_allow_html=True)
            missing_col = result.parsed_error.column
            for c in p.columns:
                kind = "red" if c == missing_col else ""
                st.markdown(chip(c, kind), unsafe_allow_html=True)
        with pc2:
            st.markdown("**Aggregations**")
            for a in p.aggregations:
                st.markdown(chip(a, "purple"), unsafe_allow_html=True)
        with pc3:
            st.markdown("**Column aliases**")
            for alias, expr in p.aliases.items():
                st.markdown(f"`{alias}` ← `{expr}`")

    # ── 3: error summary ──────────────────────────────────────────────────────
    section("3", "Error Analysis")
    pe = result.parsed_error
    all_errors = result.all_errors

    # Show count badge if multiple errors detected
    err_count = len([e for e in all_errors if e.error_type.value != "unknown"])
    if err_count > 1:
        st.markdown(
            f'<div style="margin-bottom:10px">'
            f'<span style="background:#7c1d1d;color:#fca5a5;border-radius:999px;'
            f'padding:4px 14px;font-size:12px;font-weight:700">'
            f'⚠ {err_count} distinct errors detected in this message</span></div>',
            unsafe_allow_html=True
        )

    # Render each sub-error as its own card
    # Skip the primary (full-text) entry — it's index 0 and duplicates the sub-errors
    display_errors = all_errors[1:] if len(all_errors) > 1 else all_errors
    shown = set()
    for err in display_errors:
        key = (err.error_type, err.column, err.relation)
        if key in shown or err.error_type.value == "unknown":
            continue
        shown.add(key)

        cand_html = "".join(chip(c, "green") for c in err.candidates) if err.candidates else ""
        # Use the segment text (specific to this sub-error), not the full message
        specific_text = err.raw_text.strip()[:200]

        col_display = err.column or "—"
        st.markdown(f"""
        <div class="error-banner" style="margin-bottom:10px">
            <div class="error-type-badge">{err.error_type.value.replace("_"," ")}</div>
            <div style="font-size:14px;font-weight:600;color:#fca5a5;margin-bottom:8px">
                {specific_text}
            </div>
            <div class="error-meta">
                <div class="error-meta-item">Column &nbsp;<span>{col_display}</span></div>
                <div class="error-meta-item">Model &nbsp;<span>{err.model or result.broken_model}</span></div>
                <div class="error-meta-item">Line &nbsp;<span>{err.line_number or "—"}</span></div>
            </div>
            {"<div style='margin-top:10px;font-size:11px;color:#64748b;margin-bottom:4px'>WAREHOUSE CANDIDATES</div>" + cand_html if cand_html else ""}
        </div>
        """, unsafe_allow_html=True)

    # ── 4: lineage graph ──────────────────────────────────────────────────────
    section("4", "Lineage Graph")

    # Always explain what the graph shows regardless of query validity
    lineage_note_color = "#052e16" if result.query_is_valid else "#0c1a2e"
    lineage_note_border = "#166534" if result.query_is_valid else "#1e3a5f"
    lineage_note_text = "#86efac" if result.query_is_valid else "#93c5fd"
    lineage_note_icon = "✅" if result.query_is_valid else "🔗"
    lineage_note_msg = (
        "Query is valid — lineage is shown for reference. "
        "This is the dependency chain your model sits in. "
        "No upstream columns are missing."
    ) if result.query_is_valid else (
        "Lineage is built from <code>manifest.json</code>, not from the error. "
        "It shows every model your SQL depends on. "
        "The broken model is highlighted in red — follow the chain upward to find where the column went missing."
    )
    st.markdown(f"""
    <div style="background:{lineage_note_color};border:1px solid {lineage_note_border};
                border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:13px;
                color:{lineage_note_text}">
        {lineage_note_icon} &nbsp;{lineage_note_msg}
    </div>
    """, unsafe_allow_html=True)

    g_left, g_right = st.columns([5, 2], gap="medium")

    with g_left:
        render_lineage_graph(result.lineage, result.broken_model, result.query_is_valid)

    with g_right:
        st.markdown("**Dependency path**")
        paths = result.lineage.get("paths_to_root", [])
        if paths:
            steps = paths[0]
            for i, step in enumerate(steps):
                is_broken = step == result.broken_model
                color = "#ef4444" if is_broken else "#a78bfa"
                icon  = "💥" if is_broken else ("🔗" if i < len(steps)-1 else "✅")
                st.markdown(
                    f'<div style="padding:8px 12px;margin-bottom:6px;'
                    f'background:#0f172a;border:1px solid #1e293b;border-radius:8px;'
                    f'font-size:13px;color:{color};font-weight:{"700" if is_broken else "500"}">'
                    f'{icon} {step}</div>',
                    unsafe_allow_html=True
                )

        if result.lineage.get("upstream"):
            st.markdown("<br>**Upstream models**", unsafe_allow_html=True)
            for u in result.lineage["upstream"]:
                cols_list = result.upstream_columns.get(u, [])
                with st.expander(f"↑ {u}  ({len(cols_list)} cols)"):
                    st.markdown("".join(chip(c, "purple") for c in cols_list),
                                unsafe_allow_html=True)

    # ── 5: rule engine ────────────────────────────────────────────────────────
    section("5", "Rule Engine — Deterministic Analysis")

    if result.query_is_valid:
        st.markdown("""
        <div style="background:#052e16;border:1px solid #166534;border-radius:10px;
                    padding:14px 18px;font-size:14px;color:#86efac">
            ✅ &nbsp;No rule violations — all columns referenced in the SQL
            are present in the upstream models. Nothing to fix here.
        </div>
        """, unsafe_allow_html=True)
    elif not result.rule_hits:
        st.info("No deterministic rule hits. Needs Claude reasoning.")
    else:
        for i, hit in enumerate(result.rule_hits, 1):
            s     = hit.confidence
            level = "high" if s >= .85 else ("med" if s >= .65 else "low")
            color = conf_color(s)
            label = conf_label(s)
            st.markdown(f"""
            <div class="rule-card {level}">
                <div class="rule-rank" style="color:{color}">
                    #{i} &nbsp;·&nbsp; {label} CONFIDENCE
                </div>
                <div class="rule-title">{hit.title}</div>
                {conf_bar(s)}
                <div class="rule-evidence" style="margin-top:10px">
                    📎 &nbsp;{hit.evidence}
                </div>
                <div class="rule-fix">
                    🔧 &nbsp;{hit.fix_hint}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── 5b: rule-based corrected SQL (always shown, no Claude needed) ─────────
    if not result.query_is_valid and result.corrected_sql_from_rules:
        section("5b", "Auto-corrected SQL — from Rule Engine")
        st.markdown("""
        <div style="background:#0c1a0c;border:1px solid #166534;border-radius:10px;
                    padding:10px 16px;margin-bottom:12px;font-size:13px;color:#86efac">
            ⚡ Generated automatically from rule engine — no Claude needed.
            Verify before running.
        </div>""", unsafe_allow_html=True)
        d1, d2 = st.columns(2, gap="medium")
        with d1:
            st.markdown('<div class="diff-label broken">✗  Original (broken)</div>', unsafe_allow_html=True)
            st.code(sql_input.strip(), language="sql")
        with d2:
            st.markdown('<div class="diff-label fixed">✓  Corrected (rule engine)</div>', unsafe_allow_html=True)
            st.code(result.corrected_sql_from_rules.strip(), language="sql")
            st.download_button("⬇ Download corrected SQL", result.corrected_sql_from_rules,
                               file_name="fixed_model.sql", mime="text/plain",
                               use_container_width=True)

    # ── 6: Claude ─────────────────────────────────────────────────────────────
    section("6", "Claude Reasoning")

    if result.query_is_valid:
        st.markdown("""
        <div style="background:#052e16;border:1px solid #166534;border-radius:10px;
                    padding:14px 18px;font-size:14px;color:#86efac">
            ✅ &nbsp;Claude was not called — the query is already correct.
            No corrected SQL needed.
        </div>
        """, unsafe_allow_html=True)
    elif not use_llm:
        st.info("Claude reasoning is disabled. Turn it on in the sidebar.")
    elif result.error and not result.llm_result:
        st.warning(f"Claude was not called: {result.error}")
    elif result.llm_result:
        llm   = result.llm_result
        score = llm.confidence_score
        warn  = score < 0.75
        banner_class = "rc-banner warn" if warn else "rc-banner"
        score_label  = conf_label(score)

        st.markdown(f"""
        <div class="{banner_class}">
            <div class="rc-label">{'⚠ ' if warn else '✦ '}Root Cause · {score_label}</div>
            <div class="rc-text">{llm.root_cause}</div>
            <span class="rc-score">{score:.0%} confidence</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div style="font-size:15px;color:#cbd5e1;line-height:1.7;margin-bottom:20px">{llm.explanation}</div>',
                    unsafe_allow_html=True)

        # SQL diff
        st.markdown("---")
        d1, d2 = st.columns(2, gap="medium")
        with d1:
            st.markdown('<div class="diff-label broken">✗  Original (broken)</div>', unsafe_allow_html=True)
            st.code(sql_input.strip(), language="sql")
        with d2:
            st.markdown('<div class="diff-label fixed">✓  Corrected</div>', unsafe_allow_html=True)
            st.code(llm.corrected_sql.strip(), language="sql")
            st.download_button("⬇ Download corrected SQL", llm.corrected_sql,
                               file_name="fixed_model.sql", mime="text/plain",
                               use_container_width=True)

        # Validation checklist
        if llm.validation_steps:
            st.markdown("---")
            st.markdown("**Validation checklist**")
            for step in llm.validation_steps:
                st.markdown(f"""
                <div class="check-item">
                    <span class="check-icon">○</span>
                    <span>{step}</span>
                </div>""", unsafe_allow_html=True)

        # Ranked causes
        if llm.ranked_causes:
            with st.expander("All ranked causes"):
                for rc in llm.ranked_causes:
                    c_val = rc.get("confidence", 0)
                    st.markdown(
                        f'<div style="margin-bottom:8px">'
                        f'<span style="color:{conf_color(c_val)};font-weight:700">{int(c_val*100)}%</span>'
                        f' &nbsp;{rc.get("title", rc.get("cause",""))}</div>',
                        unsafe_allow_html=True
                    )

    # ── 7: raw JSON ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🗂  Raw pipeline output (JSON)"):
        st.json(result.to_dict())
