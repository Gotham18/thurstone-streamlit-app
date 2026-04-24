import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import re
from scipy.stats import norm

# =========================================================
# Fixed survey pattern config
# =========================================================
APP_TITLE = "Hybrid Thurstone Case V Tool"
APPEAL_PREFIX = "Q1_Overall_Appeal_"
RANK_PREFIX = "Q2_RANK"
OUTPUT_FILENAME = "hybrid_thurstone_case_v_results.xlsx"

# =========================================================
# Helpers
# =========================================================
def normalize_key(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()

def extract_numeric_suffix(col_name, prefix):
    pattern = rf"^{re.escape(prefix)}(\d+)$"
    match = re.match(pattern, str(col_name).strip())
    return int(match.group(1)) if match else None

def get_appeal_columns(columns):
    appeal_cols = {}
    for col in columns:
        idx = extract_numeric_suffix(col, APPEAL_PREFIX)
        if idx is not None:
            appeal_cols[idx] = col
    return dict(sorted(appeal_cols.items()))

def get_rank_columns(columns):
    rank_cols = {}
    for col in columns:
        idx = extract_numeric_suffix(col, RANK_PREFIX)
        if idx is not None:
            rank_cols[idx] = col
    return dict(sorted(rank_cols.items()))

def extract_unique_appeal_values(df, appeal_cols):
    values = set()
    for _, col in appeal_cols.items():
        vals = df[col].dropna().astype(str).str.strip()
        values.update(vals.tolist())
    return sorted(values, key=lambda x: x.lower())

def build_label_map(item_numbers, label_df=None):
    if label_df is None:
        return {i: f"Item {i}" for i in item_numbers}

    required = {"Item_Number", "Item_Label"}
    if not required.issubset(set(label_df.columns)):
        raise ValueError("Label mapping file must contain columns: Item_Number, Item_Label")

    tmp = label_df.copy()
    tmp["Item_Number"] = pd.to_numeric(tmp["Item_Number"], errors="coerce")

    if tmp["Item_Number"].isna().any():
        raise ValueError("Label mapping file has invalid Item_Number values.")

    tmp["Item_Number"] = tmp["Item_Number"].astype(int)
    tmp["Item_Label"] = tmp["Item_Label"].astype(str).str.strip()

    if tmp["Item_Number"].duplicated().any():
        raise ValueError("Label mapping file contains duplicate Item_Number values.")

    label_map = {row["Item_Number"]: row["Item_Label"] for _, row in tmp.iterrows()}

    for i in item_numbers:
        if i not in label_map:
            label_map[i] = f"Item {i}"

    return label_map

# =========================================================
# Templates
# =========================================================
def build_sample_main_template():
    return pd.DataFrame({
        "participant_id": ["p1", "p2", "p3", "p4"],
        "Q1_Overall_Appeal_1": [3, 4, 3, 3],
        "Q1_Overall_Appeal_2": [3, 3, 2, 1],
        "Q1_Overall_Appeal_3": [3, 2, 2, 4],
        "Q1_Overall_Appeal_4": [1, 4, 2, 1],
        "Q1_Overall_Appeal_5": [2, 4, 3, 3],
        "Q1_Overall_Appeal_6": [3, 2, 2, 2],
        "Q1_Overall_Appeal_7": [3, 3, 3, 1],
        "Q1_Overall_Appeal_8": [3, 4, 3, 4],
        "Q1_Overall_Appeal_9": [2, 3, 3, 2],
        "Q2_RANK1": [1, 1, 1, 2],
        "Q2_RANK2": [4, 7, 8, 9],
        "Q2_RANK3": [5, 3, 3, 7]
    })

def build_sample_label_template(num_items=9):
    return pd.DataFrame({
        "Item_Number": list(range(1, num_items + 1)),
        "Item_Label": [f"Concept {i}" for i in range(1, num_items + 1)]
    })

# =========================================================
# Appeal mapping from UI order
# =========================================================
def build_order_based_liking_map(unique_values, mapping_df):
    tmp = mapping_df.copy()
    tmp["Appeal_Value"] = tmp["Appeal_Value"].astype(str).str.strip()
    tmp["Order"] = pd.to_numeric(tmp["Order"], errors="coerce")

    if tmp["Order"].isna().any():
        raise ValueError("Each appeal value must have an order assigned.")

    if not np.all(tmp["Order"].astype(float).apply(float.is_integer)):
        raise ValueError("Order values must be whole numbers.")

    tmp["Order"] = tmp["Order"].astype(int)

    if tmp["Order"].duplicated().any():
        raise ValueError("Each appeal value must have a unique order.")

    expected_orders = list(range(1, len(unique_values) + 1))
    observed_orders = sorted(tmp["Order"].tolist())

    if observed_orders != expected_orders:
        raise ValueError(f"Order values must be exactly {expected_orders}, got {observed_orders}.")

    mapped_values = set(tmp["Appeal_Value"].tolist())
    expected_values = set(unique_values)

    if mapped_values != expected_values:
        missing = expected_values - mapped_values
        extra = mapped_values - expected_values
        msg = []
        if missing:
            msg.append(f"Missing mappings for: {sorted(missing)}")
        if extra:
            msg.append(f"Unexpected mappings for: {sorted(extra)}")
        raise ValueError("Appeal mapping mismatch. " + " | ".join(msg))

    n = len(unique_values)
    tmp["Score"] = tmp["Order"].apply(lambda x: n - x + 1)

    liking_map = {
        normalize_key(row["Appeal_Value"]): int(row["Score"])
        for _, row in tmp.iterrows()
    }

    display_df = (
        tmp[["Appeal_Value", "Order", "Score"]]
        .sort_values("Order")
        .reset_index(drop=True)
        .copy()
    )

    return liking_map, display_df

# =========================================================
# Validation + transform
# =========================================================
def validate_and_transform_main_data(raw_data, liking_map):
    if raw_data.empty:
        raise ValueError("The input data is empty.")

    df = raw_data.copy()

    appeal_cols = get_appeal_columns(df.columns)
    rank_cols = get_rank_columns(df.columns)

    if len(appeal_cols) < 2:
        raise ValueError(
            f"Could not find at least 2 appeal columns named like {APPEAL_PREFIX}1, {APPEAL_PREFIX}2, etc."
        )

    if len(rank_cols) == 0:
        raise ValueError(
            f"Could not find rank columns named like {RANK_PREFIX}1, {RANK_PREFIX}2, etc."
        )

    item_numbers = sorted(appeal_cols.keys())

    liking_score_df = pd.DataFrame(index=df.index)
    for item_num, col in appeal_cols.items():
        liking_score_df[item_num] = df[col].apply(normalize_key).map(liking_map)

    invalid_rows = []

    for idx, row in df.iterrows():
        row_num = idx + 2

        for _, col in appeal_cols.items():
            raw_val = row[col]
            norm_val = normalize_key(raw_val)

            if pd.isna(raw_val) or str(raw_val).strip() == "":
                invalid_rows.append((row_num, f"{col}: missing appeal value."))
            elif norm_val not in liking_map:
                invalid_rows.append((row_num, f"{col}: value '{raw_val}' is not mapped."))

        observed_items = []
        for rank_pos, rank_col in rank_cols.items():
            val = pd.to_numeric(row[rank_col], errors="coerce")

            if pd.isna(val):
                continue
            if val <= 0:
                invalid_rows.append((row_num, f"{rank_col}: ranked item number must be positive."))
                continue
            if not float(val).is_integer():
                invalid_rows.append((row_num, f"{rank_col}: ranked item number must be a whole number."))
                continue

            val = int(val)

            if val not in item_numbers:
                invalid_rows.append(
                    (row_num, f"{rank_col}: ranked item number {val} does not match any appeal column.")
                )
                continue

            observed_items.append(val)

        if len(observed_items) != len(set(observed_items)):
            invalid_rows.append((row_num, "The same item was ranked more than once."))

        non_missing_rank_positions = []
        for rank_pos, rank_col in rank_cols.items():
            val = pd.to_numeric(row[rank_col], errors="coerce")
            if pd.notna(val):
                non_missing_rank_positions.append(rank_pos)

        if len(non_missing_rank_positions) > 0:
            expected_positions = list(range(1, len(non_missing_rank_positions) + 1))
            if non_missing_rank_positions != expected_positions:
                invalid_rows.append(
                    (row_num, f"Ranks must be filled consecutively from {RANK_PREFIX}1 onward without gaps.")
                )

    if invalid_rows:
        msg = "\n".join([f"Row {r}: {reason}" for r, reason in invalid_rows[:25]])
        extra = ""
        if len(invalid_rows) > 25:
            extra = f"\n...and {len(invalid_rows) - 25} more issue(s)."
        raise ValueError("Validation failed.\n\n" + msg + extra)

    rank_df = pd.DataFrame(np.nan, index=df.index, columns=item_numbers)

    for idx, row in df.iterrows():
        for rank_pos, rank_col in rank_cols.items():
            val = pd.to_numeric(row[rank_col], errors="coerce")
            if pd.notna(val):
                rank_df.at[idx, int(val)] = rank_pos

    return {
        "appeal_scores": liking_score_df.copy(),
        "ranks": rank_df.copy(),
        "item_numbers": item_numbers,
        "appeal_cols": appeal_cols,
        "rank_cols": rank_cols
    }

# =========================================================
# Hybrid pairwise logic
# =========================================================
def compare_two_items(rank_i, like_i, rank_j, like_j):
    i_ranked = pd.notna(rank_i)
    j_ranked = pd.notna(rank_j)

    if i_ranked and j_ranked:
        return 1.0 if rank_i < rank_j else 0.0
    if i_ranked and not j_ranked:
        return 1.0
    if not i_ranked and j_ranked:
        return 0.0

    if like_i > like_j:
        return 1.0
    if like_i < like_j:
        return 0.0

    return np.nan

def compute_pairwise_counts_hybrid(appeal_scores, rank_df, item_numbers):
    wins = pd.DataFrame(
        np.zeros((len(item_numbers), len(item_numbers)), dtype=float),
        index=item_numbers,
        columns=item_numbers
    )
    comparisons = pd.DataFrame(
        np.zeros((len(item_numbers), len(item_numbers)), dtype=float),
        index=item_numbers,
        columns=item_numbers
    )

    for idx in appeal_scores.index:
        for i in item_numbers:
            for j in item_numbers:
                if i == j:
                    continue

                like_i = appeal_scores.at[idx, i]
                like_j = appeal_scores.at[idx, j]
                rank_i = rank_df.at[idx, i]
                rank_j = rank_df.at[idx, j]

                result = compare_two_items(rank_i, like_i, rank_j, like_j)

                if pd.isna(result):
                    continue

                comparisons.loc[i, j] = comparisons.loc[i, j] + 1.0
                wins.loc[i, j] = wins.loc[i, j] + result

    return wins.copy(), comparisons.copy()

# =========================================================
# Thurstone Case V
# =========================================================
def compute_preference_matrix(wins, comparisons):
    with np.errstate(divide="ignore", invalid="ignore"):
        p_matrix = wins.divide(comparisons)

    p_matrix = p_matrix.copy()

    for i in range(len(p_matrix)):
        p_matrix.iat[i, i] = np.nan

    return p_matrix

def compute_thurstone_case_v_scores(p_matrix, clip_eps=1e-4):
    z_matrix = p_matrix.copy()

    for row_idx in range(z_matrix.shape[0]):
        for col_idx in range(z_matrix.shape[1]):
            val = z_matrix.iat[row_idx, col_idx]
            if pd.notna(val):
                z_matrix.iat[row_idx, col_idx] = norm.ppf(np.clip(val, clip_eps, 1 - clip_eps))

    scale_values = z_matrix.mean(axis=1, skipna=True)
    scale_values = scale_values - scale_values.mean()

    results = pd.DataFrame({
        "Item_Number": scale_values.index,
        "Thurstone_Score": scale_values.values
    }).sort_values("Thurstone_Score", ascending=False).reset_index(drop=True)

    return z_matrix.copy(), results.copy()

# =========================================================
# Labeling helpers
# =========================================================
def apply_labels_to_square_matrix(df, label_map):
    labeled = df.copy()
    labeled.index = [label_map.get(i, f"Item {i}") for i in labeled.index]
    labeled.columns = [label_map.get(i, f"Item {i}") for i in labeled.columns]
    return labeled

def apply_labels_to_results(results, label_map):
    out = results.copy()
    out["Item_Label"] = out["Item_Number"].map(label_map)
    return out[["Item_Number", "Item_Label", "Thurstone_Score"]].copy()

def build_rank_assignment_table(rank_df, label_map):
    out = rank_df.copy()
    out.columns = [label_map.get(i, f"Item {i}") for i in out.columns]
    out.index.name = "Respondent_Row"
    return out

def build_appeal_score_table(appeal_scores, label_map):
    out = appeal_scores.copy()
    out.columns = [label_map.get(i, f"Item {i}") for i in out.columns]
    out.index.name = "Respondent_Row"
    return out

# =========================================================
# Plot
# =========================================================
def plot_thurstone_vertical(results_labeled):
    df = results_labeled.sort_values("Thurstone_Score", ascending=False).reset_index(drop=True).copy()

    y = df["Thurstone_Score"].to_numpy(copy=True)
    labels = df["Item_Label"].astype(str).to_list()

    fig, ax = plt.subplots(figsize=(5, 8))
    ax.scatter([0] * len(y), y, s=80)

    for yi, label in zip(y, labels):
        ax.text(0.03, yi, label, va="center", ha="left", fontsize=10)

    ax.axhline(0, linestyle="--")
    ax.set_xticks([])
    ax.set_ylabel("Thurstone Scale Value")
    ax.set_title("Thurstone Case V Scale")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    if len(y) > 0:
        ax.text(0, float(np.max(y)) + 0.08, "More Preferred", ha="center", fontsize=10)
        ax.text(0, float(np.min(y)) - 0.08, "Less Preferred", ha="center", fontsize=10)

    plt.tight_layout()
    return fig

# =========================================================
# Notes + Excel export
# =========================================================
def build_method_notes(mapping_df, num_items, num_rank_cols):
    mapping_text = ", ".join(
        [f"{row.Appeal_Value}={row.Score}" for _, row in mapping_df.sort_values("Order").iterrows()]
    )

    return pd.DataFrame({
        "Method Note": [
            "Hybrid Thurstone logic precedence:",
            "1. Ranked vs ranked: lower rank wins.",
            "2. Ranked vs unranked: ranked item wins.",
            "3. Unranked vs unranked: higher appeal wins.",
            "4. Equal appeal among unranked items: no comparison.",
            f"Number of items detected: {num_items}",
            f"Number of rank slots detected: {num_rank_cols}",
            f"Appeal mapping used: {mapping_text}"
        ]
    })

def build_excel_output(
    raw_data,
    label_mapping_df,
    appeal_mapping_df,
    appeal_scores_labeled,
    rank_assignments_labeled,
    wins_labeled,
    comparisons_labeled,
    p_matrix_labeled,
    z_matrix_labeled,
    results_labeled,
    method_notes,
    fig
):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        raw_data.copy().to_excel(writer, sheet_name="Raw Data", index=False)
        appeal_mapping_df.copy().to_excel(writer, sheet_name="Appeal Mapping", index=False)

        if label_mapping_df is not None:
            label_mapping_df.copy().to_excel(writer, sheet_name="Label Mapping", index=False)

        appeal_scores_labeled.copy().to_excel(writer, sheet_name="Appeal Scores")
        rank_assignments_labeled.copy().to_excel(writer, sheet_name="Assigned Ranks")
        wins_labeled.copy().to_excel(writer, sheet_name="Pairwise Wins")
        comparisons_labeled.copy().to_excel(writer, sheet_name="Comparisons")
        p_matrix_labeled.copy().to_excel(writer, sheet_name="Preference Matrix")
        z_matrix_labeled.copy().to_excel(writer, sheet_name="Z Matrix")
        results_labeled.copy().to_excel(writer, sheet_name="Scale Scores", index=False)
        method_notes.copy().to_excel(writer, sheet_name="Method Notes", index=False)

        sheet = writer.sheets["Scale Scores"]
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format="png", dpi=150, bbox_inches="tight")
        imgdata.seek(0)
        sheet.insert_image("F2", "thurstone_plot.png", {"image_data": imgdata})

    output.seek(0)
    return output

# =========================================================
# Streamlit app
# =========================================================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.write(
        """
        Upload a CSV with:
        - numbered appeal columns like `Q1_Overall_Appeal_1`, `Q1_Overall_Appeal_2`, ...
        - rank columns like `Q2_RANK1`, `Q2_RANK2`, ...

        Rank columns must contain item numbers.
        """
    )

    st.subheader("Templates")

    sample_main_csv = build_sample_main_template().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Main Data Template",
        sample_main_csv,
        "thurstone_numbered_input_template.csv",
        "text/csv"
    )

    sample_label_csv = build_sample_label_template().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Optional Label Mapping Template",
        sample_label_csv,
        "thurstone_label_mapping_template.csv",
        "text/csv"
    )

    st.subheader("Upload files")
    main_file = st.file_uploader("Upload main survey CSV", type=["csv"])
    label_file = st.file_uploader("Upload optional label mapping CSV", type=["csv"])

    if main_file is None:
        st.info("Upload a main survey CSV to begin.")
        return

    try:
        raw_data = pd.read_csv(main_file)
        label_mapping_df = pd.read_csv(label_file) if label_file is not None else None

        st.write("### Data preview")
        st.dataframe(raw_data.head(20), use_container_width=True)

        appeal_cols = get_appeal_columns(raw_data.columns)
        rank_cols = get_rank_columns(raw_data.columns)

        if len(appeal_cols) < 2:
            st.error(f"Need at least 2 appeal columns named like {APPEAL_PREFIX}1, {APPEAL_PREFIX}2, etc.")
            return

        if len(rank_cols) == 0:
            st.error(f"Need rank columns named like {RANK_PREFIX}1, {RANK_PREFIX}2, etc.")
            return

        unique_appeal_values = extract_unique_appeal_values(raw_data, appeal_cols)

        st.subheader("Define appeal order")
        st.write("Assign ordinal order: `1 = most positive`, larger number = less positive.")

        n_vals = len(unique_appeal_values)
        order_rows = []

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Appeal Value**")
        with col2:
            st.markdown("**Order**")

        for idx, val in enumerate(unique_appeal_values):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.text(str(val))
            with c2:
                order = st.selectbox(
                    f"Order for {val}",
                    options=list(range(1, n_vals + 1)),
                    index=idx,
                    key=f"order_{idx}"
                )
            order_rows.append({"Appeal_Value": val, "Order": order})

        appeal_order_df = pd.DataFrame(order_rows)
        liking_map, appeal_mapping_df = build_order_based_liking_map(unique_appeal_values, appeal_order_df)

        transformed = validate_and_transform_main_data(raw_data.copy(), liking_map)
        appeal_scores = transformed["appeal_scores"]
        rank_df = transformed["ranks"]
        item_numbers = transformed["item_numbers"]

        label_map = build_label_map(item_numbers, label_mapping_df)

        wins, comparisons = compute_pairwise_counts_hybrid(appeal_scores, rank_df, item_numbers)
        p_matrix = compute_preference_matrix(wins, comparisons)
        z_matrix, results = compute_thurstone_case_v_scores(p_matrix)

        appeal_scores_labeled = build_appeal_score_table(appeal_scores, label_map)
        rank_assignments_labeled = build_rank_assignment_table(rank_df, label_map)
        wins_labeled = apply_labels_to_square_matrix(wins, label_map)
        comparisons_labeled = apply_labels_to_square_matrix(comparisons, label_map)
        p_matrix_labeled = apply_labels_to_square_matrix(p_matrix, label_map)
        z_matrix_labeled = apply_labels_to_square_matrix(z_matrix, label_map)
        results_labeled = apply_labels_to_results(results, label_map)

        method_notes = build_method_notes(
            appeal_mapping_df,
            num_items=len(item_numbers),
            num_rank_cols=len(transformed["rank_cols"])
        )

        st.write("### Appeal order and scores used")
        st.dataframe(appeal_mapping_df, use_container_width=True)

        left, right = st.columns([1, 1.2])

        with left:
            st.write("### Thurstone scale scores")
            st.dataframe(results_labeled, use_container_width=True)

        with right:
            fig = plot_thurstone_vertical(results_labeled)
            st.pyplot(fig)

        with st.expander("See diagnostic tables"):
            st.write("#### Comparison counts")
            st.dataframe(comparisons_labeled, use_container_width=True)

            st.write("#### Pairwise preference matrix")
            st.dataframe(p_matrix_labeled, use_container_width=True)

            st.write("#### Normal deviate (Z) matrix")
            st.dataframe(z_matrix_labeled, use_container_width=True)

        excel_output = build_excel_output(
            raw_data=raw_data,
            label_mapping_df=label_mapping_df,
            appeal_mapping_df=appeal_mapping_df,
            appeal_scores_labeled=appeal_scores_labeled,
            rank_assignments_labeled=rank_assignments_labeled,
            wins_labeled=wins_labeled,
            comparisons_labeled=comparisons_labeled,
            p_matrix_labeled=p_matrix_labeled,
            z_matrix_labeled=z_matrix_labeled,
            results_labeled=results_labeled,
            method_notes=method_notes,
            fig=fig
        )

        st.download_button(
            "Download Thurstone Scale Excel",
            data=excel_output,
            file_name=OUTPUT_FILENAME,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(str(e))

if __name__ == "__main__":
    main()
