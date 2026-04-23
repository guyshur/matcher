import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple

st.set_page_config(page_title="Preference Matcher", layout="wide", page_icon="◈")

# ── Typography & Global Styles ──────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet">

<style>
/* ── Root palette ── */
:root {
  --bg:        #0e0e1a;
  --surface:   #16162a;
  --surface-2: #1e1e36;
  --border:    rgba(212, 168, 83, 0.18);
  --border-hi: rgba(212, 168, 83, 0.55);
  --amber:     #d4a853;
  --amber-dim: rgba(212, 168, 83, 0.08);
  --violet:    #8b72e0;
  --text:      #e2dff5;
  --muted:     #6e6a8a;
  --success:   #5dd6a8;
}

/* ── App shell ── */
.stApp {
  background: var(--bg);
  background-image:
    linear-gradient(rgba(212,168,83,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(212,168,83,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stTextArea textarea {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.8rem !important;
  border-radius: 4px !important;
}
[data-testid="stSidebar"] .stTextArea textarea:focus {
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 2px var(--amber-dim) !important;
}

/* ── Typography ── */
h1 { display: none !important; } /* replaced by custom header below */

h2, h3, .stSubheader {
  font-family: 'Fraunces', Georgia, serif !important;
  font-weight: 300 !important;
  letter-spacing: -0.02em !important;
  color: var(--text) !important;
}

p, label, .stMarkdown {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.82rem !important;
  color: var(--muted) !important;
}

/* ── Sidebar headers ── */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-family: 'Fraunces', serif !important;
  font-size: 1rem !important;
  color: var(--amber) !important;
  font-weight: 600 !important;
}

/* ── Number inputs ── */
.stNumberInput input {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 1rem !important;
  font-weight: 400 !important;
  border-radius: 4px !important;
  text-align: center !important;
}
.stNumberInput input:focus {
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 2px var(--amber-dim) !important;
}
.stNumberInput button {
  background: var(--surface-2) !important;
  border-color: var(--border) !important;
  color: var(--amber) !important;
}
.stNumberInput button:hover {
  background: var(--amber-dim) !important;
  border-color: var(--border-hi) !important;
}

/* ── Labels on number inputs ── */
.stNumberInput label {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}

/* ── Run button ── */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--amber) !important;
  color: var(--amber) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  border-radius: 2px !important;
  padding: 0.6rem 2rem !important;
  transition: background 0.2s, box-shadow 0.2s !important;
}
.stButton > button:hover {
  background: var(--amber-dim) !important;
  box-shadow: 0 0 16px var(--amber-dim) !important;
}
.stButton > button:active {
  background: rgba(212,168,83,0.15) !important;
}

/* ── Containers / preference cards ── */
[data-testid="stContainer"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  padding: 1rem !important;
}
[data-testid="stContainer"]:hover {
  border-color: var(--border-hi) !important;
  transition: border-color 0.2s;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Fraunces', serif !important;
  font-size: 2rem !important;
  font-weight: 300 !important;
  color: var(--amber) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  overflow: hidden !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] [role="slider"] {
  background-color: var(--amber) !important;
  border-color: var(--amber) !important;
}

/* ── Divider ── */
hr {
  border-color: var(--border) !important;
  margin: 1.5rem 0 !important;
}

/* ── Info / warning ── */
.stAlert {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.8rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  background: var(--surface) !important;
}
[data-testid="stExpander"] summary {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.05em !important;
  color: var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Custom header ───────────────────────────────────────────────────────────
st.markdown("""
<div style="
  padding: 2.5rem 0 1.5rem 0;
  border-bottom: 1px solid rgba(212,168,83,0.18);
  margin-bottom: 2rem;
">
  <div style="
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #d4a853;
    margin-bottom: 0.5rem;
  ">◈ Optimal Assignment Engine</div>
  <h1 style="
    font-family: 'Fraunces', Georgia, serif;
    font-size: 3rem;
    font-weight: 300;
    letter-spacing: -0.04em;
    color: #e2dff5;
    margin: 0 0 0.5rem 0;
    line-height: 1;
  ">Preference<br><em style='color:#d4a853;'>Matcher</em></h1>
  <p style="
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #6e6a8a;
    margin: 0;
    letter-spacing: 0.02em;
  ">Hungarian algorithm · Optimal one-to-one assignment · Rank 1 = highest preference</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────
if "list1_items" not in st.session_state:
    st.session_state.list1_items = ["Alice", "Bob"]
    st.session_state.list2_items = ["Red", "Blue", "Green"]
    st.session_state.preferences = {}

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="
      font-family:'DM Mono',monospace;
      font-size:0.65rem;
      letter-spacing:0.18em;
      text-transform:uppercase;
      color:#6e6a8a;
      padding: 1.2rem 0 0.6rem 0;
      border-bottom: 1px solid rgba(212,168,83,0.12);
      margin-bottom: 1rem;
    ">◈ Configuration</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("List 1")
        list1_input = st.text_area(
            "Items (one per line):",
            value="\n".join(st.session_state.list1_items),
            height=150,
            key="list1_input"
        )
        st.session_state.list1_items = [
            x.strip() for x in list1_input.split("\n") if x.strip()]

    with col2:
        st.subheader("List 2")
        list2_input = st.text_area(
            "Items (one per line):",
            value="\n".join(st.session_state.list2_items),
            height=150,
            key="list2_input"
        )
        st.session_state.list2_items = [
            x.strip() for x in list2_input.split("\n") if x.strip()]

    st.divider()
    max_preferences = st.slider(
        "Max preference depth:",
        min_value=1,
        max_value=len(st.session_state.list2_items) if st.session_state.list2_items else 1,
        value=min(3, len(st.session_state.list2_items) or 1)
    )
    st.markdown(
        f"<p style='font-size:0.7rem;margin-top:-0.5rem;'>Rank 1–{max_preferences} · 0 = no preference</p>",
        unsafe_allow_html=True
    )

# ── Main content ─────────────────────────────────────────────────────────────
if not st.session_state.list1_items or not st.session_state.list2_items:
    st.warning("Add items to both lists in the sidebar to begin.")
else:
    # ── Statistics row ───────────────────────────────────────────────────────
    total_prefs = sum(len(p) for p in st.session_state.preferences.values())
    max_possible = len(st.session_state.list1_items) * max_preferences
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("List 1", len(st.session_state.list1_items))
    with col2:
        st.metric("List 2", len(st.session_state.list2_items))
    with col3:
        st.metric("Preferences set", f"{total_prefs} / {max_possible}")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Preferences ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
      font-family:'Fraunces',serif;
      font-size:1.4rem;
      font-weight:300;
      letter-spacing:-0.02em;
      color:#e2dff5;
      margin-bottom:0.25rem;
    ">Set Preferences</div>
    """, unsafe_allow_html=True)
    st.markdown(
        f"<p>Assign a rank 1–{max_preferences} to each item in List 2. Leave at 0 to skip.</p>",
        unsafe_allow_html=True
    )

    cols_per_row = 3
    for item1 in st.session_state.list1_items:
        with st.container(border=True):
            st.markdown(
                f"<div style='font-family:Fraunces,serif;font-size:1.05rem;font-weight:600;"
                f"color:#d4a853;letter-spacing:-0.01em;margin-bottom:0.5rem;'>{item1}</div>",
                unsafe_allow_html=True
            )

            if item1 not in st.session_state.preferences:
                st.session_state.preferences[item1] = {}

            cols = st.columns(cols_per_row)
            for j, item2 in enumerate(st.session_state.list2_items):
                with cols[j % cols_per_row]:
                    current_val = st.session_state.preferences[item1].get(item2, "")
                    rank = st.number_input(
                        item2,
                        min_value=0,
                        max_value=max_preferences,
                        value=int(current_val) if current_val else 0,
                        key=f"pref_{item1}_{item2}"
                    )
                    if rank > 0:
                        st.session_state.preferences[item1][item2] = rank
                    elif item2 in st.session_state.preferences[item1]:
                        del st.session_state.preferences[item1][item2]

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Algorithm ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
      font-family:'Fraunces',serif;
      font-size:1.4rem;
      font-weight:300;
      letter-spacing:-0.02em;
      color:#e2dff5;
      margin-bottom:0.75rem;
    ">Optimal Assignment</div>
    """, unsafe_allow_html=True)

    run_matching = st.button("Run Hungarian Algorithm →", key="run_algo_btn")

    def compute_matching(
        preferences: Dict, list1: List[str], list2: List[str], max_prefs: int
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Hungarian algorithm: minimise cost where rank = cost (1 = most preferred)."""
        cost_matrix = np.full((len(list1), len(list2)), float(max_prefs + 1))
        for i, item1 in enumerate(list1):
            for j, item2 in enumerate(list2):
                if item1 in preferences and item2 in preferences[item1]:
                    cost_matrix[i, j] = preferences[item1][item2]

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Satisfaction: max_prefs + 1 − rank per matched pair (0 for unranked)
        total_score = sum(
            max_prefs + 1 - cost_matrix[i, j]
            for i, j in zip(row_indices, col_indices)
            if cost_matrix[i, j] <= max_prefs
        )
        return row_indices, col_indices, total_score, cost_matrix

    if any(st.session_state.preferences.values()) and run_matching:
        row_indices, col_indices, total_score, cost_matrix = compute_matching(
            st.session_state.preferences,
            st.session_state.list1_items,
            st.session_state.list2_items,
            max_preferences
        )

        # ── Result cards ─────────────────────────────────────────────────────
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        for i, j in zip(row_indices, col_indices):
            item1 = st.session_state.list1_items[i]
            item2 = st.session_state.list2_items[j]
            pref = st.session_state.preferences.get(item1, {}).get(item2)
            rank_label = f"rank {pref}" if pref else "unranked"
            rank_color = "#5dd6a8" if pref and pref == 1 else ("#d4a853" if pref else "#6e6a8a")

            st.markdown(f"""
            <div style="
              display:flex;
              align-items:center;
              justify-content:space-between;
              background:#16162a;
              border:1px solid rgba(212,168,83,0.18);
              border-radius:6px;
              padding:0.85rem 1.25rem;
              margin-bottom:0.5rem;
              font-family:'DM Mono',monospace;
            ">
              <div style="display:flex;align-items:center;gap:1.5rem;">
                <span style="color:#e2dff5;font-size:0.9rem;">{item1}</span>
                <span style="color:#6e6a8a;font-size:0.75rem;">──→</span>
                <span style="color:#e2dff5;font-size:0.9rem;">{item2}</span>
              </div>
              <span style="
                font-size:0.68rem;
                letter-spacing:0.1em;
                text-transform:uppercase;
                color:{rank_color};
                border:1px solid {rank_color}44;
                border-radius:2px;
                padding:0.15rem 0.5rem;
              ">{rank_label}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        st.metric("Satisfaction Score", f"{total_score:.0f}")

        # ── Preference matrix ─────────────────────────────────────────────────
        with st.expander("View preference matrix"):
            pref_display = {
                item1: {
                    item2: st.session_state.preferences.get(item1, {}).get(item2, None)
                    for item2 in st.session_state.list2_items
                }
                for item1 in st.session_state.list1_items
            }
            matrix_df = pd.DataFrame(pref_display).T
            st.dataframe(
                matrix_df.map(
                    lambda x: str(int(x)) if x is not None and not (
                        isinstance(x, float) and np.isnan(x)) else "—",
                    na_action="ignore"
                ),
                width="stretch"
            )
    else:
        st.markdown("""
        <div style="
          border:1px dashed rgba(212,168,83,0.2);
          border-radius:6px;
          padding:1.5rem;
          text-align:center;
          font-family:'DM Mono',monospace;
          font-size:0.78rem;
          color:#6e6a8a;
          letter-spacing:0.05em;
          margin-top:0.5rem;
        ">Set preferences above, then run the algorithm.</div>
        """, unsafe_allow_html=True)
