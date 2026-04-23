import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple

st.set_page_config(page_title="Preference Matcher", layout="wide", page_icon="◈")

# ── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">

<style>
:root {
  --bg:       #ffffff;
  --surface:  #f5f5f5;
  --border:   #cccccc;
  --blue:     #1a56db;
  --blue-bg:  #eef2fc;
  --text:     #111111;
  --muted:    #555555;
  --green:    #166534;
  --green-bg: #dcfce7;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'Atkinson Hyperlegible', sans-serif !important;
  color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 2px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
  font-family: 'Atkinson Hyperlegible', sans-serif !important;
  color: var(--text) !important;
}
[data-testid="stSidebar"] .stTextArea textarea {
  background: var(--bg) !important;
  border: 2px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'Atkinson Hyperlegible', sans-serif !important;
  font-size: 0.95rem !important;
}
[data-testid="stSidebar"] .stTextArea textarea:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(26,86,219,0.15) !important;
}

/* All text elements */
h1, h2, h3, h4, p, label, span, div, li,
.stMarkdown, .stText {
  font-family: 'Atkinson Hyperlegible', sans-serif !important;
  color: var(--text) !important;
}

/* Number inputs */
.stNumberInput input {
  background: var(--bg) !important;
  border: 2px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'Atkinson Hyperlegible', sans-serif !important;
  font-size: 1.1rem !important;
  font-weight: 700 !important;
  border-radius: 6px !important;
  text-align: center !important;
}
.stNumberInput input:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(26,86,219,0.15) !important;
}
.stNumberInput label {
  font-family: 'Atkinson Hyperlegible', sans-serif !important;
  font-size: 0.9rem !important;
  font-weight: 700 !important;
  color: var(--text) !important;
}
.stNumberInput button {
  border: 2px solid var(--border) !important;
  color: var(--text) !important;
  background: var(--surface) !important;
}
.stNumberInput button:hover {
  background: var(--blue-bg) !important;
  border-color: var(--blue) !important;
}

/* Button */
.stButton > button {
  background: var(--blue) !important;
  color: #ffffff !important;
  font-family: 'Atkinson Hyperlegible', sans-serif !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 0.6rem 2rem !important;
}
.stButton > button:hover {
  background: #1446b5 !important;
}

/* Containers */
[data-testid="stContainer"] {
  background: var(--surface) !important;
  border: 2px solid var(--border) !important;
  border-radius: 8px !important;
}

/* Metrics */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 2px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] p {
  font-family: 'Atkinson Hyperlegible', sans-serif !important;
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Atkinson Hyperlegible', sans-serif !important;
  font-size: 2rem !important;
  font-weight: 700 !important;
  color: var(--text) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
  border: 2px solid var(--border) !important;
  border-radius: 8px !important;
}

/* Slider */
[data-testid="stSlider"] label {
  font-family: 'Atkinson Hyperlegible', sans-serif !important;
  font-weight: 700 !important;
  color: var(--text) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Alert */
.stAlert {
  font-family: 'Atkinson Hyperlegible', sans-serif !important;
  font-size: 0.95rem !important;
}

/* Expander */
[data-testid="stExpander"] summary span {
  font-family: 'Atkinson Hyperlegible', sans-serif !important;
  font-weight: 700 !important;
  color: var(--text) !important;
  font-size: 0.95rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.5rem 0 1rem 0; border-bottom:2px solid #cccccc; margin-bottom:1.5rem;">
  <h1 style="font-family:'Atkinson Hyperlegible',sans-serif; font-size:2.2rem;
             font-weight:700; color:#111111; margin:0 0 0.3rem 0;">
    Preference Matcher
  </h1>
  <p style="font-family:'Atkinson Hyperlegible',sans-serif; font-size:1rem;
            color:#555555; margin:0;">
    Hungarian algorithm &middot; Rank 1 = highest preference
  </p>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "list1_items" not in st.session_state:
    st.session_state.list1_items = ["Alice", "Bob"]
    st.session_state.list2_items = ["Red", "Blue", "Green"]
    st.session_state.preferences = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Setup")

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
        "Max preferences per item:",
        min_value=1,
        max_value=len(st.session_state.list2_items) if st.session_state.list2_items else 1,
        value=min(3, len(st.session_state.list2_items) or 1)
    )
    st.caption(f"Rank 1–{max_preferences}. Enter 0 to skip.")

# ── Main content ──────────────────────────────────────────────────────────────
if not st.session_state.list1_items or not st.session_state.list2_items:
    st.warning("Add items to both lists in the sidebar to begin.")
else:
    # Stats row
    total_prefs = sum(len(p) for p in st.session_state.preferences.values())
    max_possible = len(st.session_state.list1_items) * max_preferences
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("List 1 items", len(st.session_state.list1_items))
    with c2:
        st.metric("List 2 items", len(st.session_state.list2_items))
    with c3:
        st.metric("Preferences set", f"{total_prefs} / {max_possible}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Preferences
    st.subheader("Set Preferences")
    st.caption(f"Assign a rank 1–{max_preferences} to each item in List 2. Leave at 0 to skip.")

    cols_per_row = 3
    for item1 in st.session_state.list1_items:
        with st.container(border=True):
            st.markdown(
                f"<p style='font-family:Atkinson Hyperlegible,sans-serif;"
                f"font-size:1.05rem;font-weight:700;color:#111111;"
                f"margin:0 0 0.5rem 0;'>{item1}</p>",
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

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Algorithm
    st.subheader("Optimal Assignment")

    n1 = len(st.session_state.list1_items)
    n2 = len(st.session_state.list2_items)
    if n1 > n2:
        st.error(
            f"No complete matching is possible: List 1 has {n1} items but List 2 only has {n2}. "
            f"Every item in List 1 needs a unique match in List 2, so at least {n1 - n2} item(s) "
            f"from List 1 will be unmatched. Add more items to List 2 or remove items from List 1."
        )

    run_matching = st.button("Run Algorithm", key="run_algo_btn")

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

        # Identify unmatched List 1 items (only possible when len(list1) > len(list2))
        matched_list1 = set(row_indices)
        unmatched = [
            st.session_state.list1_items[i]
            for i in range(len(st.session_state.list1_items))
            if i not in matched_list1
        ]
        if unmatched:
            st.warning(
                f"No complete matching exists. The following item(s) from List 1 could not be matched: "
                + ", ".join(f"**{name}**" for name in unmatched)
            )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        for i, j in zip(row_indices, col_indices):
            item1 = st.session_state.list1_items[i]
            item2 = st.session_state.list2_items[j]
            pref = st.session_state.preferences.get(item1, {}).get(item2)
            rank_label = f"Rank {pref}" if pref else "Unranked"
            if pref == 1:
                badge_color, badge_bg = "#166534", "#dcfce7"
            elif pref:
                badge_color, badge_bg = "#1a56db", "#eef2fc"
            else:
                badge_color, badge_bg = "#555555", "#f5f5f5"

            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:space-between;
                        background:#ffffff; border:2px solid #cccccc; border-radius:8px;
                        padding:0.75rem 1.25rem; margin-bottom:0.5rem;">
              <span style="font-family:'Atkinson Hyperlegible',sans-serif;
                           font-size:1rem; font-weight:700; color:#111111;">
                {item1} &rarr; {item2}
              </span>
              <span style="font-family:'Atkinson Hyperlegible',sans-serif;
                           font-size:0.85rem; font-weight:700;
                           color:{badge_color}; background:{badge_bg};
                           border-radius:4px; padding:0.2rem 0.6rem;">
                {rank_label}
              </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.metric("Satisfaction Score", f"{total_score:.0f}")

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
        st.info("Set preferences above, then run the algorithm.")
