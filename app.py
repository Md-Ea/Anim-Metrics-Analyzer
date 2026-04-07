
import os
import sys
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QCheckBox,
    QRadioButton, QSpinBox, QTextEdit, QButtonGroup, QCompleter, QComboBox,
    QSplitter, QTabWidget, QTableView, QHeaderView, QGroupBox,
)
from PySide6.QtCore import Qt, QSettings, QStringListModel, QAbstractTableModel

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

def _load_and_clean(csv_path: str, threshold: int, positions: list = None, player_id: int = None, log_fn=print):
    """
    Shared loading, cleaning, run detection, and threshold filtering.
    positions: list of position strings to filter by, or None for no position filtering.
    player_id: specific player ID to filter by, or None.
    """
    log_fn(f"Loading CSV: {csv_path}")
    try:
        df_full = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df_full.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}. Found columns: {list(df_full.columns)}")

    has_ptm = "ptmDatabase" in df_full.columns
    has_pos = "playerPos" in df_full.columns
    use_cols = REQUIRED_COLUMNS + (["ptmDatabase"] if has_ptm else []) + (["playerPos"] if has_pos else [])
    df = df_full[use_cols].copy()

    # Filter by player ID or position
    if player_id is not None:
        before = len(df)
        df = df[df["playerID"].astype(str) == str(player_id)]
        filtered = before - len(df)
        if filtered:
            log_fn(f"Filtered to playerID {player_id}, removed {filtered} rows.")
        if df.empty:
            raise RuntimeError(f"No rows found for playerID {player_id}.")
    elif positions is not None:
        if has_pos:
            before = len(df)
            df = df[df["playerPos"].astype(str).isin(positions)]
            filtered = before - len(df)
            if filtered:
                log_fn(f"Filtered to positions {positions}, removed {filtered} rows.")
        else:
            log_fn("Warning: 'playerPos' column not found; position filter skipped.")

    # Drop rows with nulls in required columns
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    dropped = before - len(df)
    if dropped:
        log_fn(f"Dropped {dropped} rows with nulls in required columns.")

    # Coerce types
    # playerID/sequence as string, gameTick as integer-like
    df["playerID"] = df["playerID"].astype(str)
    df["sequence"] = df["sequence"].astype(str)

    # gameTick may be float if there are NaNs previously; coerce safely
    df["gameTick"] = pd.to_numeric(df["gameTick"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["gameTick"])
    df["gameTick"] = df["gameTick"].astype("int64")
    if len(df) != before:
        log_fn(f"Dropped {before-len(df)} rows with non-numeric gameTick.")

    if df.empty:
        raise RuntimeError("No valid rows after cleaning. Check your CSV content and required columns.")

    log_fn("Sorting by playerID, gameTick...")
    df = df.sort_values(["playerID", "gameTick"], kind="mergesort").reset_index(drop=True)

    # Identify contiguous segments ("runs") of the same sequence for each player.
    seq_change = df["sequence"].ne(df["sequence"].shift(1))
    player_change = df["playerID"].ne(df["playerID"].shift(1))
    new_run = seq_change | player_change
    df["run_id"] = new_run.cumsum()

    # Aggregate runs: start/end tick per (run_id, playerID, sequence)
    log_fn("Computing run durations...")
    group_cols = ["run_id", "playerID", "sequence"]
    agg_dict = dict(startTick=("gameTick", "min"),
                    endTick=("gameTick", "max"),
                    play_times_rows=("gameTick", "size"))

    # If ptmDatabase exists, carry forward the first value per run
    if has_ptm:
        df["ptmDatabase"] = df["ptmDatabase"].astype(str)
        agg_dict["ptmDatabase"] = ("ptmDatabase", "first")

    runs = df.groupby(group_cols, as_index=False).agg(**agg_dict)
    runs["duration_ticks"] = runs["endTick"] - runs["startTick"]

    # Filter out runs below the duration threshold
    before_filter = len(runs)
    runs = runs[runs["duration_ticks"] >= threshold]
    filtered_out = before_filter - len(runs)
    if filtered_out:
        log_fn(f"Filtered out {filtered_out} runs with duration_ticks < {threshold}.")

    if runs.empty:
        raise RuntimeError(f"No runs remaining after filtering with threshold {threshold}.")

    in_dir = os.path.dirname(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    return df, runs, has_ptm, base, in_dir


def analyze_all_sequences(csv_path: str, threshold: int = 2, positions: list = None, player_id: int = None, log_fn=print) -> str:
    """Produces the _general.csv summary per sequence."""
    df, runs, _has_ptm, base, in_dir = _load_and_clean(csv_path, threshold, positions, player_id, log_fn)

    log_fn("Summarizing per sequence...")
    seq_summary = (runs.groupby("sequence", as_index=False)
                     .agg(
                         run_count=("run_id", "count"),
                         duration_ticks_min=("duration_ticks", "min"),
                         duration_ticks_max=("duration_ticks", "max"),
                         duration_ticks_avg=("duration_ticks", "mean"),
                     ))
    seq_summary["duration_ticks_avg"] = seq_summary["duration_ticks_avg"].round(3)
    seq_summary = seq_summary.rename(columns={
        "sequence": "Sequence",
        "run_count": "Count",
        "duration_ticks_min": "Duration Min",
        "duration_ticks_max": "Duration Max",
        "duration_ticks_avg": "Duration Average",
    })

    out_path = os.path.join(in_dir, f"{base}_general.csv")
    log_fn(f"Saving general CSV: {out_path}")
    seq_summary.to_csv(out_path, index=False)
    log_fn(f"Done. Unique sequences: {len(seq_summary)}")
    return out_path


def analyze_database(csv_path: str, threshold: int = 2, positions: list = None, player_id: int = None, db_filter: str = "", log_fn=print) -> str:
    """Produces the _database.csv: ptmDatabase, sequence, database_count."""
    df, runs, has_ptm, base, in_dir = _load_and_clean(csv_path, threshold, positions, player_id, log_fn)

    if not has_ptm:
        raise RuntimeError("Column 'ptmDatabase' not found in the CSV. Cannot produce database export.")

    log_fn("Building database export...")

    # Filter to a specific database if requested
    if db_filter:
        runs = runs[runs["ptmDatabase"] == db_filter]
        if runs.empty:
            raise RuntimeError(f"No runs found for ptmDatabase '{db_filter}'.")
        log_fn(f"Filtered to database: {db_filter}")

    # Count how many times each sequence is played per database and avg duration
    seq_counts = runs.groupby(["ptmDatabase", "sequence"], as_index=False).agg(
        sequence_count=("run_id", "count"),
        duration_ticks_avg=("duration_ticks", "mean"))
    seq_counts["duration_ticks_avg"] = seq_counts["duration_ticks_avg"].round(3)

    # Count total runs per database
    db_counts = runs.groupby("ptmDatabase", as_index=False).agg(database_count=("run_id", "count"))

    # Merge database count back
    db_seq = seq_counts.merge(db_counts, on="ptmDatabase")
    db_seq = db_seq.sort_values(["database_count", "ptmDatabase", "sequence"],
                                ascending=[False, True, True]).reset_index(drop=True)
    db_seq = db_seq.rename(columns={
        "ptmDatabase": "Database",
        "sequence": "Sequence",
        "sequence_count": "Sequence Count",
        "duration_ticks_avg": "Duration Average",
        "database_count": "Database Count",
    })

    out_path = os.path.join(in_dir, f"{base}_database.csv")
    log_fn(f"Saving database CSV: {out_path}")
    db_seq.to_csv(out_path, index=False)
    log_fn(f"Done. Unique databases: {len(db_counts)}, rows: {len(db_seq)}")
    return out_path


def analyze_sequence(csv_path: str, seq_name: str, threshold: int = 2, positions: list = None, player_id: int = None, log_fn=print) -> str:
    """Produces a _sequence.csv for a specific sequence: databases it appears in + duration."""
    df, runs, has_ptm, base, in_dir = _load_and_clean(csv_path, threshold, positions, player_id, log_fn)

    if not has_ptm:
        raise RuntimeError("Column 'ptmDatabase' not found in the CSV. Cannot produce sequence export.")

    runs_seq = runs[runs["sequence"] == seq_name]
    if runs_seq.empty:
        raise RuntimeError(f"No runs found for sequence '{seq_name}' (after threshold filtering).")

    log_fn(f"Building sequence export for '{seq_name}'...")
    result = (runs_seq.groupby(["sequence", "ptmDatabase"], as_index=False)
              .agg(sequence_count=("run_id", "count"),
                   duration_ticks_min=("duration_ticks", "min"),
                   duration_ticks_max=("duration_ticks", "max"),
                   duration_ticks_avg=("duration_ticks", "mean"),
                   tick_duration=("duration_ticks", "sum")))
    result["duration_ticks_avg"] = result["duration_ticks_avg"].round(3)
    result = result.sort_values("tick_duration", ascending=False).reset_index(drop=True)
    result = result.rename(columns={
        "sequence": "Sequence",
        "ptmDatabase": "Database",
        "sequence_count": "Sequence Count",
        "duration_ticks_min": "Duration Min",
        "duration_ticks_max": "Duration Max",
        "duration_ticks_avg": "Duration Average",
        "tick_duration": "Total Duration",
    })

    out_path = os.path.join(in_dir, f"{base}_sequence.csv")
    log_fn(f"Saving sequence CSV: {out_path}")
    result.to_csv(out_path, index=False)
    log_fn(f"Done. Databases containing '{seq_name}': {len(result)}")
    return out_path


def analyze_timeline(csv_path: str, player_id: int, threshold: int = 2, positions: list = None, log_fn=print) -> str:
    """Produces a _timeline.csv: chronological sequence list for a specific player."""
    df, runs, has_ptm, base, in_dir = _load_and_clean(csv_path, threshold, positions, player_id, log_fn)

    log_fn(f"Building timeline for player {player_id}...")
    timeline = runs.sort_values("startTick").reset_index(drop=True)
    timeline["order"] = range(1, len(timeline) + 1)
    timeline["game_time"] = timeline["startTick"]
    cols = ["order", "sequence"]
    if has_ptm:
        cols.append("ptmDatabase")
    cols += ["startTick", "endTick", "duration_ticks", "game_time"]
    result = timeline[cols].copy()
    result = result.rename(columns={
        "order": "Order",
        "sequence": "Sequence",
        "ptmDatabase": "Database",
        "startTick": "Start Tick",
        "endTick": "End Tick",
        "duration_ticks": "Duration (Ticks)",
        "game_time": "Game Time (Ticks)",
    })

    out_path = os.path.join(in_dir, f"{base}_timeline.csv")
    log_fn(f"Saving timeline CSV: {out_path}")
    result.to_csv(out_path, index=False)
    log_fn(f"Done. {len(result)} sequences in chronological order.")
    return out_path


# --- Build functions for in-app tables (matching web.py) ---

def build_general(runs):
    result = (runs.groupby("sequence", as_index=False)
              .agg(run_count=("run_id", "count"),
                   duration_ticks_min=("duration_ticks", "min"),
                   duration_ticks_max=("duration_ticks", "max"),
                   duration_ticks_avg=("duration_ticks", "mean")))
    result["duration_ticks_avg"] = result["duration_ticks_avg"].round(3)
    return result.rename(columns={
        "sequence": "Sequence", "run_count": "Count",
        "duration_ticks_min": "Duration Min", "duration_ticks_max": "Duration Max",
        "duration_ticks_avg": "Duration Average",
    })


def build_database(runs, db_filter=""):
    if "ptmDatabase" not in runs.columns:
        raise RuntimeError("'ptmDatabase' column not found.")
    if db_filter:
        runs = runs[runs["ptmDatabase"] == db_filter]
        if runs.empty:
            raise RuntimeError(f"No runs found for database '{db_filter}'.")
    seq_counts = runs.groupby(["ptmDatabase", "sequence"], as_index=False).agg(
        sequence_count=("run_id", "count"), duration_ticks_avg=("duration_ticks", "mean"))
    seq_counts["duration_ticks_avg"] = seq_counts["duration_ticks_avg"].round(3)
    db_counts = runs.groupby("ptmDatabase", as_index=False).agg(database_count=("run_id", "count"))
    db_seq = seq_counts.merge(db_counts, on="ptmDatabase")
    db_seq = db_seq.sort_values(["database_count", "ptmDatabase", "sequence"],
                                ascending=[False, True, True]).reset_index(drop=True)
    return db_seq.rename(columns={
        "ptmDatabase": "Database", "sequence": "Sequence",
        "sequence_count": "Sequence Count", "duration_ticks_avg": "Duration Average",
        "database_count": "Database Count",
    })


def build_sequence(runs, seq_name):
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
    return result.rename(columns={
        "sequence": "Sequence", "ptmDatabase": "Database",
        "sequence_count": "Sequence Count", "duration_ticks_min": "Duration Min",
        "duration_ticks_max": "Duration Max", "duration_ticks_avg": "Duration Average",
        "tick_duration": "Total Duration",
    })


def build_player_timeline(runs):
    timeline = runs.sort_values("startTick").reset_index(drop=True)
    timeline["game_time"] = timeline["startTick"]
    cols = ["sequence"]
    if "ptmDatabase" in timeline.columns:
        cols.append("ptmDatabase")
    cols += ["startTick", "endTick", "duration_ticks", "game_time"]
    result = timeline[cols].copy()
    return result.rename(columns={
        "sequence": "Sequence", "ptmDatabase": "Database",
        "startTick": "Start Tick", "endTick": "End Tick",
        "duration_ticks": "Duration (Ticks)", "game_time": "Game Time (Ticks)",
    })


class PandasModel(QAbstractTableModel):
    """Qt table model wrapping a pandas DataFrame."""
    def __init__(self, df=None):
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            val = self._df.iloc[index.row(), index.column()]
            if isinstance(val, float):
                return f"{val:.3f}" if val != int(val) else str(int(val))
            return str(val)
        if role == Qt.TextAlignmentRole:
            val = self._df.iloc[index.row(), index.column()]
            if isinstance(val, (int, float)):
                return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            return str(section + 1)
        return None

    def setDataFrame(self, df):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def sort(self, column, order=Qt.AscendingOrder):
        if self._df.empty:
            return
        col_name = self._df.columns[column]
        ascending = order == Qt.AscendingOrder
        self.beginResetModel()
        self._df = self._df.sort_values(col_name, ascending=ascending).reset_index(drop=True)
        self.endResetModel()


def _make_table_view():
    tv = QTableView()
    tv.setAlternatingRowColors(True)
    tv.setSortingEnabled(True)
    tv.setSelectionBehavior(QTableView.SelectRows)
    tv.horizontalHeader().setStretchLastSection(True)
    tv.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    tv.verticalHeader().setDefaultSectionSize(24)
    return tv


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anim Metrics Analyzer")
        self.resize(1200, 700)
        self.settings = QSettings("EAAnimAnalyzer", "App")
        self._all_databases = []
        self._all_sequences = []
        self._runs = None
        self._has_ptm = False

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # === LEFT SIDEBAR ===
        sidebar = QWidget()
        sidebar.setMinimumWidth(300)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(6, 6, 6, 6)

        # Input CSV
        sb.addWidget(QLabel("<b>Input</b>"))
        row_csv = QHBoxLayout()
        self.csv_entry = QLineEdit()
        self.csv_entry.setText(self.settings.value("last_csv_path", ""))
        row_csv.addWidget(self.csv_entry, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_csv)
        row_csv.addWidget(browse_btn)
        sb.addLayout(row_csv)

        # Filters
        sb.addWidget(QLabel("<b>Filters</b>"))

        # Radio buttons in a flow layout
        filter_box = QHBoxLayout()
        self.filter_all_radio = QRadioButton("Everyone")
        self.filter_players_radio = QRadioButton("All Players")
        self.filter_players_radio.setChecked(True)
        self.filter_position_radio = QRadioButton("Position")
        self.filter_group_radio = QRadioButton("Role")
        self.filter_playerid_radio = QRadioButton("Player ID")
        self.pos_group = QButtonGroup(self)
        for r in [self.filter_all_radio, self.filter_players_radio, self.filter_position_radio,
                  self.filter_group_radio, self.filter_playerid_radio]:
            self.pos_group.addButton(r)
            filter_box.addWidget(r)
        sb.addLayout(filter_box)

        # Position / Role / Player ID controls
        filter_detail = QHBoxLayout()
        self.position_combo = QComboBox()
        self.position_combo.addItems(POSITION_LIST)
        self.position_combo.setEnabled(False)
        self.filter_position_radio.toggled.connect(self.position_combo.setEnabled)
        filter_detail.addWidget(QLabel("Pos:"))
        filter_detail.addWidget(self.position_combo)
        self.group_combo = QComboBox()
        self.group_combo.addItems(list(POSITION_GROUPS.keys()))
        self.group_combo.setEnabled(False)
        self.filter_group_radio.toggled.connect(self.group_combo.setEnabled)
        filter_detail.addWidget(QLabel("Role:"))
        filter_detail.addWidget(self.group_combo)
        self.playerid_spin = QSpinBox()
        self.playerid_spin.setRange(0, 99)
        self.playerid_spin.setValue(0)
        self.playerid_spin.setFixedWidth(50)
        self.playerid_spin.setEnabled(False)
        self.filter_playerid_radio.toggled.connect(self.playerid_spin.setEnabled)
        filter_detail.addWidget(QLabel("ID:"))
        filter_detail.addWidget(self.playerid_spin)
        filter_detail.addStretch()
        sb.addLayout(filter_detail)

        # Min Duration
        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Min Duration:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 999999)
        self.threshold_spin.setValue(2)
        dur_row.addWidget(self.threshold_spin)
        dur_row.addStretch()
        sb.addLayout(dur_row)

        # Database filter
        db_row = QHBoxLayout()
        db_row.addWidget(QLabel("Database:"))
        self.db_filter_entry = QLineEdit()
        self.db_filter_entry.setPlaceholderText("(all)")
        self.db_filter_entry.setText(self.settings.value("last_db_filter", ""))
        self.db_completer_model = QStringListModel()
        self.db_completer = QCompleter(self.db_completer_model, self)
        self.db_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.db_completer.setFilterMode(Qt.MatchContains)
        self.db_completer.setMaxVisibleItems(15)
        self.db_filter_entry.setCompleter(self.db_completer)
        db_row.addWidget(self.db_filter_entry, 1)
        sb.addLayout(db_row)

        # Sequence filter
        seq_row = QHBoxLayout()
        seq_row.addWidget(QLabel("Sequence:"))
        self.seq_filter_entry = QLineEdit()
        self.seq_filter_entry.setPlaceholderText("(all)")
        self.seq_filter_entry.setText(self.settings.value("last_seq_filter", ""))
        self.seq_completer_model = QStringListModel()
        self.seq_completer = QCompleter(self.seq_completer_model, self)
        self.seq_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.seq_completer.setFilterMode(Qt.MatchContains)
        self.seq_completer.setMaxVisibleItems(15)
        self.seq_filter_entry.setCompleter(self.seq_completer)
        seq_row.addWidget(self.seq_filter_entry, 1)
        sb.addLayout(seq_row)

        # Apply button
        apply_btn = QPushButton("Apply")
        apply_btn.setMinimumHeight(36)
        apply_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 13px; "
            "font-weight: bold; padding: 6px 20px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:pressed { background-color: #3d8b40; }"
        )
        apply_btn.clicked.connect(self.apply_filters)
        sb.addWidget(apply_btn)

        # Log
        sb.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        sb.addWidget(self.log_text)
        sb.addStretch()

        splitter.addWidget(sidebar)

        # === RIGHT MAIN AREA ===
        self.tab_widget = QTabWidget()
        splitter.addWidget(self.tab_widget)

        # Create tabs: each tab is a wrapper widget with table + status label
        self.general_model = PandasModel()
        self.general_table = _make_table_view()
        self.general_table.setModel(self.general_model)
        self.general_status = QLabel("")
        self.tab_widget.addTab(self._make_tab(self.general_table, self.general_status), "General Results")

        self.db_model = PandasModel()
        self.db_table = _make_table_view()
        self.db_table.setModel(self.db_model)
        self.db_status = QLabel("")
        self.tab_widget.addTab(self._make_tab(self.db_table, self.db_status), "Database View")

        self.seq_model = PandasModel()
        self.seq_table = _make_table_view()
        self.seq_table.setModel(self.seq_model)
        self.seq_status = QLabel("")
        self.tab_widget.addTab(self._make_tab(self.seq_table, self.seq_status), "Sequence View")

        self.timeline_model = PandasModel()
        self.timeline_table = _make_table_view()
        self.timeline_table.setModel(self.timeline_model)
        self.timeline_status = QLabel("")
        self.tab_widget.addTab(self._make_tab(self.timeline_table, self.timeline_status), "Timeline View")

        # Set initial splitter sizes: sidebar ~400px, main area gets the rest
        splitter.setSizes([400, 800])

        # Disable all data tabs initially
        self._set_tab_enabled(0, False, "Click Apply to load data")
        self._set_tab_enabled(1, False, "Requires ptmDatabase column")
        self._set_tab_enabled(2, False, "Enter a sequence name and click Apply")
        self._set_tab_enabled(3, False, "Select Player ID filter and click Apply")

        self._log("Select a CSV and click Apply to view results.")
        self._load_csv_metadata()
        self.csv_entry.textChanged.connect(self._on_csv_changed)

    def browse_csv(self):
        last_dir = self.settings.value("last_browse_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV file", last_dir, "CSV files (*.csv);;All files (*.*)")
        if path:
            self.csv_entry.setText(path)
            self.settings.setValue("last_csv_path", path)
            self.settings.setValue("last_browse_dir", os.path.dirname(path))
            self._log(f"Selected: {path}")

    @staticmethod
    def _make_tab(table, status_label):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(4, 4, 4, 4)
        vl.addWidget(table, 1)
        vl.addWidget(status_label)
        return w

    def _set_tab_enabled(self, idx, enabled, tooltip=""):
        self.tab_widget.setTabEnabled(idx, enabled)
        self.tab_widget.setTabToolTip(idx, tooltip if not enabled else "")

    def _on_csv_changed(self, text):
        self._load_csv_metadata()

    def _load_csv_metadata(self):
        csv_path = self.csv_entry.text().strip()
        if not csv_path or not os.path.exists(csv_path):
            return
        try:
            df = pd.read_csv(csv_path)
            if "ptmDatabase" in df.columns:
                self._all_databases = sorted(df["ptmDatabase"].dropna().astype(str).unique())
                self.db_completer_model.setStringList(self._all_databases)
            if "sequence" in df.columns:
                self._all_sequences = sorted(df["sequence"].dropna().astype(str).unique())
                self.seq_completer_model.setStringList(self._all_sequences)
        except Exception:
            pass

    def _log(self, msg: str):
        self.log_text.append(msg)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _get_filter_params(self):
        """Return (positions, player_id, threshold, db_filter, seq_filter)."""
        if self.filter_all_radio.isChecked():
            positions = None
        elif self.filter_players_radio.isChecked():
            positions = POSITION_LIST
        elif self.filter_position_radio.isChecked():
            positions = [self.position_combo.currentText()]
        elif self.filter_group_radio.isChecked():
            positions = POSITION_GROUPS[self.group_combo.currentText()]
        else:
            positions = None
        player_id = self.playerid_spin.value() if self.filter_playerid_radio.isChecked() else None
        threshold = self.threshold_spin.value()
        db_filter = self.db_filter_entry.text().strip()
        seq_filter = self.seq_filter_entry.text().strip()
        self.settings.setValue("last_db_filter", db_filter)
        self.settings.setValue("last_seq_filter", seq_filter)
        return positions, player_id, threshold, db_filter, seq_filter

    def apply_filters(self):
        csv_path = self.csv_entry.text().strip()
        if not csv_path or not os.path.exists(csv_path):
            QMessageBox.critical(self, "Missing CSV", "Please select a valid input CSV file.")
            return

        positions, player_id, threshold, db_filter, seq_filter = self._get_filter_params()

        try:
            _df, runs, has_ptm, _base, _in_dir = _load_and_clean(
                csv_path, threshold, positions, player_id, self._log)
            self._runs = runs
            self._has_ptm = has_ptm
        except RuntimeError as e:
            self._log(f"ERROR: {e}")
            QMessageBox.critical(self, "Load failed", str(e))
            return

        # General Results
        try:
            gen_df = build_general(runs)
            self.general_model.setDataFrame(gen_df)
            self.general_status.setText(f"{len(gen_df)} sequences")
            self._set_tab_enabled(0, True)
            self.tab_widget.setCurrentIndex(0)
        except Exception as e:
            self.general_status.setText(f"Error: {e}")
            self._set_tab_enabled(0, False, "Error loading data")

        # Database View
        if has_ptm:
            try:
                db_df = build_database(runs, db_filter)
                self.db_model.setDataFrame(db_df)
                self.db_status.setText(f"{len(db_df)} rows")
                self._set_tab_enabled(1, True)
            except RuntimeError as e:
                self.db_model.setDataFrame(pd.DataFrame())
                self.db_status.setText(f"Error: {e}")
                self._set_tab_enabled(1, False, str(e))
        else:
            self.db_model.setDataFrame(pd.DataFrame())
            self.db_status.setText("No ptmDatabase column found.")
            self._set_tab_enabled(1, False, "Requires ptmDatabase column in CSV")

        # Sequence View
        if has_ptm and seq_filter:
            try:
                seq_df = build_sequence(runs, seq_filter)
                self.seq_model.setDataFrame(seq_df)
                self.seq_status.setText(f"{len(seq_df)} databases")
                self._set_tab_enabled(2, True)
            except RuntimeError as e:
                self.seq_model.setDataFrame(pd.DataFrame())
                self.seq_status.setText(f"Error: {e}")
                self._set_tab_enabled(2, False, str(e))
        else:
            self.seq_model.setDataFrame(pd.DataFrame())
            self._set_tab_enabled(2, False, "Enter a sequence name in the sidebar and click Apply")

        # Timeline View
        if player_id is not None:
            try:
                tl_df = build_player_timeline(runs)
                self.timeline_model.setDataFrame(tl_df)
                self.timeline_status.setText(f"{len(tl_df)} sequences in chronological order")
                self._set_tab_enabled(3, True)
            except Exception as e:
                self.timeline_model.setDataFrame(pd.DataFrame())
                self.timeline_status.setText(f"Error: {e}")
                self._set_tab_enabled(3, False, str(e))
        else:
            self.timeline_model.setDataFrame(pd.DataFrame())
            self._set_tab_enabled(3, False, "Select Player ID filter and click Apply")

        self._log("Tables updated.")


def main():
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
