import io
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import html as html_mod

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


def _colored_html_table(df_a: pd.DataFrame, df_b: pd.DataFrame, key_cols: list, value_cols: list, name_a: str, name_b: str) -> str:
    """Build an HTML table where each value cell shows blue (A) / red (B) in one cell."""
    merged = pd.merge(df_a, df_b, on=key_cols, how="outer", suffixes=("_A", "_B")).fillna(0)

    rows_html = []
    # Header - mark key cols for sorting
    header = "".join(f"<th data-keycol='true'>{c}</th>" for c in key_cols)
    header += "".join(f"<th class='val-col' data-keycol='false'>{c}</th>" for c in value_cols)
    rows_html.append(f"<tr>{header}</tr>")

    # Store data attributes for sorting
    for _, row in merged.iterrows():
        cells = ""
        for kc in key_cols:
            val = html_mod.escape(str(row[kc]))
            cells += f"<td data-text='{val}'>{val}</td>"
        for vc in value_cols:
            va = row.get(f"{vc}_A", 0)
            vb = row.get(f"{vc}_B", 0)
            # Format numbers
            va_s = f"{va:.3f}" if isinstance(va, float) and va != int(va) else str(int(va)) if va == int(va) else str(va)
            vb_s = f"{vb:.3f}" if isinstance(vb, float) and vb != int(vb) else str(int(vb)) if vb == int(vb) else str(vb)
            va_num = float(va) if va else 0
            vb_num = float(vb) if vb else 0
            cells += (f"<td class='val-col' "
                      f"data-blue='{va_num}' data-red='{vb_num}'>"
                      f"<span class='blue'>{va_s}</span>"
                      f" / "
                      f"<span class='red'>{vb_s}</span></td>")
        rows_html.append(f"<tr>{cells}</tr>")

    row_count = len(rows_html) - 1
    height = min(max(row_count * 35 + 80, 200), 700)

    full_html = f"""<!DOCTYPE html>
<html><head><style>
  body {{ font-family: "Source Sans Pro", sans-serif; margin: 0; padding: 0; background: transparent; color: #31333F; font-size: 14px; }}
  .table-wrap {{ border: 1px solid #e6e9ef; border-radius: 8px; overflow: hidden; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ padding: 8px 16px; text-align: left; font-size: 12px; font-weight: 600; color: #555; text-transform: uppercase; letter-spacing: .4px;
        background: #f0f2f6; border-bottom: 1px solid #e6e9ef; cursor: pointer; user-select: none; white-space: nowrap; }}
  th.val-col {{ text-align: right; }}
  th:hover {{ background: #e1e4eb; }}
  td {{ padding: 8px 16px; border-bottom: 1px solid #e6e9ef; }}
  td.val-col {{ text-align: right; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f7f8fc; }}
  .blue {{ color: #1f77b4; font-weight: 600; }}
  .red {{ color: #d62728; font-weight: 600; }}
</style></head><body>
<div class="table-wrap">
<table id="sortable">
  <thead>{''.join(rows_html[:1])}</thead>
  <tbody>{''.join(rows_html[1:])}</tbody>
</table>
</div>
<script>
(function() {{
  var table = document.getElementById('sortable');
  var headers = table.querySelectorAll('th');
  headers.forEach(function(th, idx) {{
    th.addEventListener('click', function() {{
      var isKey = th.getAttribute('data-keycol') === 'true';
      var rows = Array.from(table.querySelectorAll('tbody tr'));
      var state = parseInt(th.dataset.sortState || '-1');
      headers.forEach(function(h) {{ h.dataset.sortState = '-1'; h.style.color = ''; }});
      if (isKey) {{
        state = (state + 1) % 2;
        th.dataset.sortState = state;
        rows.sort(function(a, b) {{
          var at = a.children[idx].getAttribute('data-text') || '';
          var bt = b.children[idx].getAttribute('data-text') || '';
          return state === 0 ? at.localeCompare(bt) : bt.localeCompare(at);
        }});
      }} else {{
        state = (state + 1) % 4;
        th.dataset.sortState = state;
        rows.sort(function(a, b) {{
          var ac = a.children[idx], bc = b.children[idx];
          var aB = parseFloat(ac.getAttribute('data-blue')) || 0;
          var bB = parseFloat(bc.getAttribute('data-blue')) || 0;
          var aR = parseFloat(ac.getAttribute('data-red')) || 0;
          var bR = parseFloat(bc.getAttribute('data-red')) || 0;
          if (state === 0) return bB - aB;
          if (state === 1) return aB - bB;
          if (state === 2) return bR - aR;
          return aR - bR;
        }});
      }}
      if (!isKey) {{ th.style.color = (state <= 1) ? '#1f77b4' : '#d62728'; }}
      var tbody = table.querySelector('tbody');
      rows.forEach(function(r) {{ tbody.appendChild(r); }});
    }});
  }});
}})();
</script>
</body></html>"""

    return full_html, height


def compare_general(runs_a, runs_b, name_a, name_b):
    gen_a = build_general(runs_a)
    gen_b = build_general(runs_b)
    key_cols = ["sequence"]
    value_cols = ["run_count", "play_times_rows", "duration_ticks_min", "duration_ticks_max", "duration_ticks_avg"]
    return _colored_html_table(gen_a, gen_b, key_cols, value_cols, name_a, name_b)


def compare_database(runs_a, runs_b, name_a, name_b, db_filter=""):
    db_a = build_database(runs_a, db_filter)
    db_b = build_database(runs_b, db_filter)
    key_cols = ["ptmDatabase", "sequence"]
    value_cols = ["sequence_count", "duration_ticks_avg", "database_count"]
    return _colored_html_table(db_a, db_b, key_cols, value_cols, name_a, name_b)


def compare_sequence(runs_a, runs_b, name_a, name_b, seq_name):
    seq_a = build_sequence(runs_a, seq_name)
    seq_b = build_sequence(runs_b, seq_name)
    key_cols = ["sequence", "ptmDatabase"]
    value_cols = ["sequence_count", "duration_ticks_min", "duration_ticks_max", "duration_ticks_avg", "tick_duration"]
    return _colored_html_table(seq_a, seq_b, key_cols, value_cols, name_a, name_b)


def main():
    st.set_page_config(page_title="Anim Metrics Analyzer", layout="wide")

    # Wider sidebar + smaller sidebar fonts
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { min-width: 420px; font-size: 13px; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 16px; }
    [data-testid="stSidebar"] label { font-size: 13px; }
    [data-testid="stSidebar"] .stCaption { font-size: 11px; }
    </style>
    """, unsafe_allow_html=True)

    # --- Sidebar: Upload & Filters ---
    with st.sidebar:
        st.header("Input")
        uploaded_files = st.file_uploader("Upload CSV(s)", type=["csv"], accept_multiple_files=True)

        if not uploaded_files:
            st.info("Upload one or more CSV files to begin.")
            st.stop()

        # Read all uploaded files
        csv_data = {}
        for f in uploaded_files:
            csv_data[f.name] = pd.read_csv(f)

        file_names = list(csv_data.keys())

        # File selection checkboxes
        selected = []
        if len(file_names) >= 2:
            st.caption("Select 1 or 2 files to compare")
        for fn in file_names:
            if st.checkbox(fn, value=True, key=f"sel_{fn}"):
                selected.append(fn)
        if len(selected) > 2:
            st.warning("Select at most 2 files.")
            st.stop()

        if not selected:
            st.warning("Select at least one file.")
            st.stop()

        st.header("Filters")

        # Use first selected file for filter options
        df_raw = csv_data[selected[0]]

        filter_mode = st.radio("Include", ["Everyone", "All Players", "Player Position", "Player Role", "Player ID"], index=1)

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

        db_filter = ""
        if "ptmDatabase" in df_raw.columns:
            databases = sorted(df_raw["ptmDatabase"].dropna().astype(str).unique())
            db_filter = st.selectbox("Filter by Database", ["(all)"] + databases)
            if db_filter == "(all)":
                db_filter = ""

        seq_filter = ""
        if "sequence" in df_raw.columns:
            sequences = sorted(df_raw["sequence"].dropna().astype(str).unique())
            seq_filter = st.selectbox("Filter by Sequence", ["(all)"] + sequences)
            if seq_filter == "(all)":
                seq_filter = ""

        apply = st.button("Apply", type="primary", use_container_width=True)

    # --- Main area ---
    if not apply:
        st.info("Set your filters in the sidebar and click **Apply**.")
        st.stop()

    # Process selected files
    all_runs = {}
    all_logs = {}
    all_ptm = {}
    for name in selected:
        try:
            runs, has_ptm, logs = load_and_clean(csv_data[name], threshold, positions, player_id)
            all_runs[name] = runs
            all_ptm[name] = has_ptm
            all_logs[name] = logs
        except RuntimeError as e:
            st.error(f"**{name}**: {e}")

    if not all_runs:
        st.stop()

    # --- Single file view ---
    if len(selected) == 1:
        name = selected[0]
        runs = all_runs[name]
        has_ptm = all_ptm[name]

        tab_names = ["General Results"]
        if has_ptm:
            tab_names += ["Database View", "Sequence View"]
        tabs = st.tabs(tab_names)

        with tabs[0]:
            st.subheader(f"General Results — {name}")
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

    # --- Comparison view ---
    elif len(selected) == 2:
        name_a, name_b = selected
        runs_a, runs_b = all_runs[name_a], all_runs[name_b]
        has_ptm_both = all_ptm[name_a] and all_ptm[name_b]

        # Short labels for column suffixes
        label_a = name_a.replace(".csv", "")
        label_b = name_b.replace(".csv", "")

        tab_names = ["General Comparison", f"{name_a}", f"{name_b}"]
        if has_ptm_both:
            tab_names.append("Database Comparison")
            tab_names.append("Sequence Comparison")
        tabs = st.tabs(tab_names)

        with tabs[0]:
            st.subheader(f"Comparison: {name_a} vs {name_b}")
            st.markdown(f"Legend: <span style='color:#4A90D9;font-weight:bold;'>{name_a}</span> / <span style='color:#D94A4A;font-weight:bold;'>{name_b}</span>", unsafe_allow_html=True)
            html_table, h = compare_general(runs_a, runs_b, label_a, label_b)
            components.html(html_table, height=h, scrolling=True)

        with tabs[1]:
            st.subheader(f"General Results — {name_a}")
            st.dataframe(build_general(runs_a), use_container_width=True, height=600)

        with tabs[2]:
            st.subheader(f"General Results — {name_b}")
            st.dataframe(build_general(runs_b), use_container_width=True, height=600)

        if has_ptm_both:
            with tabs[3]:
                st.subheader("Database Comparison")
                st.markdown(f"Legend: <span style='color:#4A90D9;font-weight:bold;'>{name_a}</span> / <span style='color:#D94A4A;font-weight:bold;'>{name_b}</span>", unsafe_allow_html=True)
                try:
                    html_table, h = compare_database(runs_a, runs_b, label_a, label_b, db_filter)
                    components.html(html_table, height=h, scrolling=True)
                except RuntimeError as e:
                    st.error(str(e))

            with tabs[4]:
                st.subheader(f"Sequence Comparison{f' — {seq_filter}' if seq_filter else ''}")
                if seq_filter:
                    st.markdown(f"Legend: <span style='color:#4A90D9;font-weight:bold;'>{name_a}</span> / <span style='color:#D94A4A;font-weight:bold;'>{name_b}</span>", unsafe_allow_html=True)
                    try:
                        html_table, h = compare_sequence(runs_a, runs_b, label_a, label_b, seq_filter)
                        components.html(html_table, height=h, scrolling=True)
                    except RuntimeError as e:
                        st.error(str(e))
                else:
                    st.info("Select a specific sequence in the sidebar to see its database breakdown comparison.")

    # Processing logs at the bottom
    all_log_items = [(n, l) for n, l in all_logs.items() if l]
    if all_log_items:
        with st.expander("Processing log"):
            for name, logs in all_log_items:
                st.write(f"**{name}**")
                for msg in logs:
                    st.write(f"  - {msg}")


if __name__ == "__main__":
    main()
