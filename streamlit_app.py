import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple

st.set_page_config(page_title="Preference Matcher", layout="wide")
st.title("🎯 Preference Matcher")
st.markdown(
    "Create two lists and let items from List 1 rank items from List 2, then find optimal matching.")

# Initialize session state
if "list1_items" not in st.session_state:
    st.session_state.list1_items = ["Alice", "Bob"]
    st.session_state.list2_items = ["Red", "Blue", "Green"]
    st.session_state.preferences = {}

# Sidebar controls
with st.sidebar:
    st.header("📋 Setup")

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
        max_value=len(
            st.session_state.list2_items) if st.session_state.list2_items else 1,
        value=min(3, len(st.session_state.list2_items) or 1)
    )


# Main content
if not st.session_state.list1_items or not st.session_state.list2_items:
    st.warning("Please add items to both lists in the sidebar.")
else:
    st.subheader("⭐ Set Preferences")
    st.markdown(
        f"Each item ranks up to **{max_preferences}** items from List 2 (1=most preferred, leave blank to skip)")

    # Create preference input
    preference_data = {}
    cols_per_row = 3

    for i, item1 in enumerate(st.session_state.list1_items):
        with st.container(border=True):
            st.markdown(f"**{item1}**")

            # Initialize preferences for this item if not exists
            if item1 not in st.session_state.preferences:
                st.session_state.preferences[item1] = {}

            cols = st.columns(cols_per_row)
            for j, item2 in enumerate(st.session_state.list2_items):
                with cols[j % cols_per_row]:
                    key = f"pref_{item1}_{item2}"
                    current_val = st.session_state.preferences[item1].get(
                        item2, "")
                    rank = st.number_input(
                        item2,
                        min_value=0,
                        max_value=max_preferences,
                        value=int(current_val) if current_val else 0,
                        key=key,
                        label_visibility="collapsed"
                    )
                    if rank > 0:
                        st.session_state.preferences[item1][item2] = rank
                    elif item2 in st.session_state.preferences[item1]:
                        del st.session_state.preferences[item1][item2]

    st.divider()

    # Run matching algorithm
    st.subheader("🔄 Optimal Matching")

    def compute_matching(preferences: Dict, list1: List[str], list2: List[str]) -> Tuple[np.ndarray, float]:
        """
        Use Hungarian algorithm to find maximum weight matching.

        Args:
            preferences: Dict mapping list1 items to their ranked preferences
            list1: List of items from list 1
            list2: List of items from list 2

        Returns:
            Indices of matching and total satisfaction score
        """
        # Create cost matrix (negative because we want to maximize)
        cost_matrix = np.zeros((len(list1), len(list2)))

        for i, item1 in enumerate(list1):
            for j, item2 in enumerate(list2):
                if item1 in preferences and item2 in preferences[item1]:
                    # Higher rank = higher preference, so use negative for minimization
                    cost_matrix[i, j] = -preferences[item1][item2]
                else:
                    cost_matrix[i, j] = 0  # No preference given

        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        total_score = -cost_matrix[row_indices, col_indices].sum()

        return row_indices, col_indices, total_score, cost_matrix

    if any(st.session_state.preferences.values()):
        row_indices, col_indices, total_score, cost_matrix = compute_matching(
            st.session_state.preferences,
            st.session_state.list1_items,
            st.session_state.list2_items
        )

        # Display results
        results = []
        for i, j in zip(row_indices, col_indices):
            item1 = st.session_state.list1_items[i]
            item2 = st.session_state.list2_items[j]
            preference = st.session_state.preferences.get(
                item1, {}).get(item2, "-")
            results.append({
                "From List 1": item1,
                "Matched to List 2": item2,
                "Preference Rank": preference
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.metric("Total Satisfaction Score", f"{total_score:.0f}")

        # Show preference matrix
        with st.expander("📊 View Preference Matrix"):
            matrix_df = pd.DataFrame(
                cost_matrix,
                index=st.session_state.list1_items,
                columns=st.session_state.list2_items
            )
            # Replace 0s with empty string for clarity
            matrix_df = matrix_df.replace(0, "")
            st.dataframe(matrix_df.applymap(
                lambda x: f"-{x:.0f}" if x != "" else "", na_action='ignore'), use_container_width=True)
    else:
        st.info("Enter preferences above to run the matching algorithm.")

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("List 1 Size", len(st.session_state.list1_items))
    with col2:
        st.metric("List 2 Size", len(st.session_state.list2_items))
    with col3:
        total_prefs = sum(len(prefs)
                          for prefs in st.session_state.preferences.values())
        max_possible = len(st.session_state.list1_items) * max_preferences
        st.metric("Preferences Set", f"{total_prefs}/{max_possible}")
