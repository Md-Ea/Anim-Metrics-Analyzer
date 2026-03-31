import io
import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = ["playerID", "sequence", "gameTick"]

POSITION_LIST = [
    "GK", "SW", "CB", "RB", "LB", "RWB", "LWB",
    "DM", "CDM", "CM", "RM", "LM", "AM", "CAM",
    "RW", "LW", "CF", "SS", "ST",
]

POSITION_GROUPS = {
    "Defenders": ["GK", "SW", "CB", "RB", "LB", "RWB", "LWB"],
    "Midfielders": ["DM", "CDM", "CM", "RM", "LM", "AM", "CAM"],
    "Forwards": ["RW", "LW", "CF", "SS", "ST"],
}


def load_and_clean(df_full: pd.DataFrame, threshold: int, positions: list = None, player_id: int = None):
    """Clean data, detect runs, filter by threshold. Returns (runs, has_ptm, logs)."""
    logs = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df_full.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Found: {list(df_full.columns)}")

    has_ptm = "ptmDatabase" in df_full.columns
    has_pos = "playerPos" in df_full.columns
    use_cols = REQUIRED_COLUMNS + (["ptmDatabase"] if has_ptm else []) + (["playerPos"] if has_pos else [])
    df = df_full[use_cols].copy()

    # Filter by player ID or position
    if player_id is not None:
        before = len(df)
        df = df[df["playerID"].astype(str) == str(player_id)]
        if len(df) < before:
            logs.append(f"Filtered to playerID {player_id}, removed {before - len(df)} rows.")
        if df.empty:
            raise RuntimeError(f"No rows found for playerID {player_id}.")
    elif positions is not None:
        if has_pos:
            before = len(df)
            df = df[df["playerPos"].astype(str).isin(positions)]
            if len(df) < before:
                logs.append(f"Filtered to positions, removed {before - len(df)} rows.")
        else:
            logs.append("Warning: 'playerPos' column not found; position filter skipped.")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df["playerID"] = df["playerID"].astype(str)
    df["sequence"] = df["sequence"].astype(str)
    df["gameTick"] = pd.to_numeric(df["gameTick"], errors="coerce")
    df = df.dropna(subset=["gameTick"])
    df["gameTick"] = df["gameTick"].astype("int64")

    if df.empty:
        raise RuntimeError("No valid rows after cleaning.")

    df = df.sort_values(["playerID", "gameTick"], kind="mergesort").reset_index(drop=True)

    seq_change = df["sequence"].ne(df["sequence"].shift(1))
    player_change = df["playerID"].ne(df["playerID"].shift(1))
    df["run_id"] = (seq_change | player_change).cumsum()

    group_cols = ["run_id", "playerID", "sequence"]
    agg_dict = dict(startTick=("gameTick", "min"),
                    endTick=("gameTick", "max"),
                    play_times_rows=("gameTick", "size"))
    if has_ptm:
        df["ptmDatabase"] = df["ptmDatabase"].astype(str)
        agg_dict["ptmDatabase"] = ("ptmDatabase", "first")

    runs = df.groupby(group_cols, as_index=False).agg(**agg_dict)
    runs["duration_ticks"] = runs["endTick"] - runs["startTick"]

    before_filter = len(runs)
    runs = runs[runs["duration_ticks"] >= threshold]
    filtered_out = before_filter - len(runs)
    if filtered_out:
        logs.append(f"Filtered out {filtered_out} runs with duration < {threshold}.")

    if runs.empty:
        raise RuntimeError(f"No runs remaining after threshold {threshold}.")

    return runs, has_ptm, logs


def build_general(runs: pd.DataFrame) -> pd.DataFrame:
    result = (runs.groupby("sequence", as_index=False)
              .agg(run_count=("run_id", "count"),
                   play_times_rows=("play_times_rows", "sum"),
                   duration_ticks_min=("duration_ticks", "min"),
                   duration_ticks_max=("duration_ticks", "max"),
                   duration_ticks_avg=("duration_ticks", "mean")))
    result["duration_ticks_avg"] = result["duration_ticks_avg"].round(3)
    return result


def build_database(runs: pd.DataFrame, db_filter: str = "") -> pd.DataFrame:
    if "ptmDatabase" not in runs.columns:
        raise RuntimeError("'ptmDatabase' column not found.")
    if db_filter:
        runs = runs[runs["ptmDatabase"] == db_filter]
        if runs.empty:
            raise RuntimeError(f"No runs found for database '{db_filter}'.")

    seq_counts = runs.groupby(["ptmDatabase", "sequence"], as_index=False).agg(
        sequence_count=("run_id", "count"),
        duration_ticks_avg=("duration_ticks", "mean"))
    seq_counts["duration_ticks_avg"] = seq_counts["duration_ticks_avg"].round(3)

    db_counts = seq_counts.groupby("ptmDatabase", as_index=False).agg(database_count=("sequence", "count"))
    db_seq = seq_counts.merge(db_counts, on="ptmDatabase")
    db_seq = db_seq.sort_values(["database_count", "ptmDatabase", "sequence"],
                                ascending=[False, True, True]).reset_index(drop=True)
    return db_seq


def build_sequence(runs: pd.DataFrame, seq_name: str) -> pd.DataFrame:
    if "ptmDatabase" not in runs.columns:
        raise RuntimeError("'ptmDatabase' column not found.")
    runs_seq = runs[runs["sequence"] == seq_name]
    if runs_seq.empty:
        raise RuntimeError(f"No runs found for sequence '{seq_name}'.")

    result = (runs_seq.groupby(["sequence", "ptmDatabase"], as_index=False)
              .agg(sequence_count=("run_id", "count"),
                   duration_ticks_min=("duration_ticks", "min"),
                   duration_ticks_max=("duration_ticks", "max"),
                   duration_ticks_avg=("duration_ticks", "mean"),
                   tick_duration=("duration_ticks", "sum")))
    result["duration_ticks_avg"] = result["duration_ticks_avg"].round(3)
    result = result.sort_values("tick_duration", ascending=False).reset_index(drop=True)
    return result


def main():
    st.set_page_config(page_title="Anim Metrics Analyzer", layout="wide")
    st.title("Anim Metrics Analyzer")

    # --- Sidebar: Upload & Filters ---
    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            df_raw = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")
        else:
            st.info("Upload a CSV to begin.")
            st.stop()

        st.header("Filters")

        # Player filter
        filter_mode = st.radio("Include", ["Everyone", "All Players", "Player Position", "Player Role", "Player ID"])

        positions = None
        player_id = None

        if filter_mode == "All Players":
            positions = POSITION_LIST
        elif filter_mode == "Player Position":
            pos = st.selectbox("Position", POSITION_LIST)
            positions = [pos]
        elif filter_mode == "Player Role":
            role = st.selectbox("Role", list(POSITION_GROUPS.keys()))
            positions = POSITION_GROUPS[role]
        elif filter_mode == "Player ID":
            player_id = st.number_input("Player ID", min_value=0, max_value=99, value=0, step=1)

        threshold = st.number_input("Min Duration (ticks)", min_value=0, value=2, step=1)

        # Database filter
        db_filter = ""
        if "ptmDatabase" in df_raw.columns:
            databases = sorted(df_raw["ptmDatabase"].dropna().astype(str).unique())
            db_filter = st.selectbox("Filter by Database", ["(all)"] + databases)
            if db_filter == "(all)":
                db_filter = ""

        # Sequence filter
        seq_filter = ""
        if "sequence" in df_raw.columns:
            sequences = sorted(df_raw["sequence"].dropna().astype(str).unique())
            seq_filter = st.selectbox("Filter by Sequence", ["(all)"] + sequences)
            if seq_filter == "(all)":
                seq_filter = ""

        apply = st.button("Apply", type="primary", use_container_width=True)

    # --- Main area: Results ---
    if not apply:
        st.info("Set your filters in the sidebar and click **Apply**.")
        st.stop()

    try:
        runs, has_ptm, logs = load_and_clean(df_raw, threshold, positions, player_id)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    # Tabs for different views
    tab_names = ["General Results"]
    if has_ptm:
        tab_names.append("Database View")
        tab_names.append("Sequence View")

    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.subheader("General Results — All Sequences")
        general_df = build_general(runs)
        st.dataframe(general_df, use_container_width=True, height=600)
        st.caption(f"{len(general_df)} sequences")

    if has_ptm:
        with tabs[1]:
            st.subheader(f"Database View{f' — {db_filter}' if db_filter else ''}")
            try:
                db_df = build_database(runs, db_filter)
                st.dataframe(db_df, use_container_width=True, height=600)
                st.caption(f"{len(db_df)} rows")
            except RuntimeError as e:
                st.error(str(e))

        with tabs[2]:
            st.subheader(f"Sequence View{f' — {seq_filter}' if seq_filter else ''}")
            if seq_filter:
                try:
                    seq_df = build_sequence(runs, seq_filter)
                    st.dataframe(seq_df, use_container_width=True, height=600)
                    st.caption(f"{len(seq_df)} databases")
                except RuntimeError as e:
                    st.error(str(e))
            else:
                st.info("Select a specific sequence in the sidebar to see its database breakdown.")

    # Processing log at the bottom
    if logs:
        with st.expander("Processing log"):
            for msg in logs:
                st.write(msg)


if __name__ == "__main__":
    main()
