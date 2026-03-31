
import os
import sys
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QCheckBox,
    QRadioButton, QSpinBox, QTextEdit, QButtonGroup, QCompleter, QComboBox,
)
from PySide6.QtCore import Qt, QSettings, QStringListModel

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
                         play_times_rows=("play_times_rows", "sum"),
                         duration_ticks_min=("duration_ticks", "min"),
                         duration_ticks_max=("duration_ticks", "max"),
                         duration_ticks_avg=("duration_ticks", "mean"),
                     ))
    seq_summary["duration_ticks_avg"] = seq_summary["duration_ticks_avg"].round(3)

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

    # Count how many unique sequences each database has
    db_counts = seq_counts.groupby("ptmDatabase", as_index=False).agg(database_count=("sequence", "count"))

    # Merge database count back
    db_seq = seq_counts.merge(db_counts, on="ptmDatabase")
    db_seq = db_seq.sort_values(["database_count", "ptmDatabase", "sequence"],
                                ascending=[False, True, True]).reset_index(drop=True)

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

    out_path = os.path.join(in_dir, f"{base}_sequence.csv")
    log_fn(f"Saving sequence CSV: {out_path}")
    result.to_csv(out_path, index=False)
    log_fn(f"Done. Databases containing '{seq_name}': {len(result)}")
    return out_path


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EA Anim Sequence Analyzer (All Sequences)")
        self.resize(900, 560)
        self.settings = QSettings("EAAnimAnalyzer", "App")
        self._all_databases = []
        self._all_sequences = []

        layout = QVBoxLayout(self)

        # Row 1 — Input CSV + Browse
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Input CSV:"))
        self.csv_entry = QLineEdit()
        self.csv_entry.setText(self.settings.value("last_csv_path", ""))
        row1.addWidget(self.csv_entry, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_csv)
        row1.addWidget(browse_btn)
        layout.addLayout(row1)

        # Row 2 — Export options
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Export:"))
        self.export_analyzed_cb = QCheckBox("General Results")
        self.export_analyzed_cb.setChecked(True)
        row2.addWidget(self.export_analyzed_cb)
        row2.addStretch()
        layout.addLayout(row2)

        # Row 2a — Database export
        row2a = QHBoxLayout()
        row2a.addWidget(QLabel("Database CSV:"))
        self.db_filter_entry = QLineEdit()
        self.db_filter_entry.setPlaceholderText("Enter database name to export (empty = skip)")
        self.db_filter_entry.setText(self.settings.value("last_db_filter", ""))
        self.db_completer_model = QStringListModel()
        self.db_completer = QCompleter(self.db_completer_model, self)
        self.db_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.db_completer.setFilterMode(Qt.MatchContains)
        self.db_completer.setMaxVisibleItems(15)
        self.db_filter_entry.setCompleter(self.db_completer)
        row2a.addWidget(self.db_filter_entry, 1)
        layout.addLayout(row2a)

        # Row 2b — Sequence export
        row2b = QHBoxLayout()
        row2b.addWidget(QLabel("Sequence CSV:"))
        self.seq_filter_entry = QLineEdit()
        self.seq_filter_entry.setPlaceholderText("Enter sequence name to export (empty = skip)")
        self.seq_filter_entry.setText(self.settings.value("last_seq_filter", ""))
        self.seq_completer_model = QStringListModel()
        self.seq_completer = QCompleter(self.seq_completer_model, self)
        self.seq_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.seq_completer.setFilterMode(Qt.MatchContains)
        self.seq_completer.setMaxVisibleItems(15)
        self.seq_filter_entry.setCompleter(self.seq_completer)
        row2b.addWidget(self.seq_filter_entry, 1)
        layout.addLayout(row2b)

        # Row 3 — Filter radios
        row3 = QHBoxLayout()
        self.filter_all_radio = QRadioButton("Everyone")
        self.filter_players_radio = QRadioButton("All Players")
        self.filter_players_radio.setChecked(True)
        self.filter_position_radio = QRadioButton("Player Position")
        self.filter_group_radio = QRadioButton("Player Role")
        self.filter_playerid_radio = QRadioButton("Player ID")
        self.pos_group = QButtonGroup(self)
        self.pos_group.addButton(self.filter_all_radio)
        self.pos_group.addButton(self.filter_players_radio)
        self.pos_group.addButton(self.filter_position_radio)
        self.pos_group.addButton(self.filter_group_radio)
        self.pos_group.addButton(self.filter_playerid_radio)
        row3.addWidget(self.filter_all_radio)
        row3.addWidget(self.filter_players_radio)
        row3.addWidget(self.filter_position_radio)
        self.position_combo = QComboBox()
        self.position_combo.addItems(POSITION_LIST)
        self.position_combo.setEnabled(False)
        self.filter_position_radio.toggled.connect(self.position_combo.setEnabled)
        row3.addWidget(self.position_combo)
        row3.addWidget(self.filter_group_radio)
        self.group_combo = QComboBox()
        self.group_combo.addItems(list(POSITION_GROUPS.keys()))
        self.group_combo.setEnabled(False)
        self.filter_group_radio.toggled.connect(self.group_combo.setEnabled)
        row3.addWidget(self.group_combo)
        row3.addWidget(self.filter_playerid_radio)
        self.playerid_spin = QSpinBox()
        self.playerid_spin.setRange(0, 99)
        self.playerid_spin.setValue(0)
        self.playerid_spin.setFixedWidth(50)
        self.playerid_spin.setEnabled(False)
        self.filter_playerid_radio.toggled.connect(self.playerid_spin.setEnabled)
        row3.addWidget(self.playerid_spin)
        row3.addStretch()
        layout.addLayout(row3)

        # Row 3b — Min Duration
        row3b = QHBoxLayout()
        row3b.addWidget(QLabel("Min Duration:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 999999)
        self.threshold_spin.setValue(2)
        row3b.addWidget(self.threshold_spin)
        row3b.addStretch()
        layout.addLayout(row3b)

        # Row 4 — Analyze & Save button
        row4 = QHBoxLayout()
        row4.addStretch()
        analyze_btn = QPushButton("Analyze && Save")
        analyze_btn.setMinimumHeight(40)
        analyze_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 14px; "
            "font-weight: bold; padding: 8px 24px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:pressed { background-color: #3d8b40; }"
        )
        analyze_btn.clicked.connect(self.run_analysis)
        row4.addWidget(analyze_btn)
        layout.addLayout(row4)

        # Log
        layout.addWidget(QLabel("Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text, 1)

        self._log("Select a CSV, then click 'Analyze & Save'.\n"
                  "Output is saved next to the input file with suffix '_analyzed.csv'.")

        # Auto-populate completers if CSV is already set
        self._load_csv_metadata()
        self.csv_entry.textChanged.connect(self._on_csv_changed)

    def browse_csv(self):
        last_dir = self.settings.value("last_browse_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV file", last_dir,
            "CSV files (*.csv);;All files (*.*)"
        )
        if path:
            self.csv_entry.setText(path)
            self.settings.setValue("last_csv_path", path)
            self.settings.setValue("last_browse_dir", os.path.dirname(path))
            self._log(f"Selected: {path}")

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
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def run_analysis(self):
        csv_path = self.csv_entry.text().strip()
        if not csv_path:
            QMessageBox.critical(self, "Missing CSV", "Please select an input CSV file.")
            return
        if not os.path.exists(csv_path):
            QMessageBox.critical(self, "File not found", f"CSV path does not exist:\n{csv_path}")
            return

        threshold = self.threshold_spin.value()

        do_analyzed = self.export_analyzed_cb.isChecked()
        db_name = self.db_filter_entry.text().strip()
        seq_name = self.seq_filter_entry.text().strip()
        if not do_analyzed and not db_name and not seq_name:
            QMessageBox.critical(self, "No export selected", "Please select at least one export option.")
            return

        filter_pos = self.filter_players_radio.isChecked()
        player_id = self.playerid_spin.value() if self.filter_playerid_radio.isChecked() else None

        # Determine positions filter
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

        # Save last used values
        self.settings.setValue("last_db_filter", db_name)
        self.settings.setValue("last_seq_filter", seq_name)

        results = []
        try:
            if do_analyzed:
                out = analyze_all_sequences(csv_path, threshold=threshold, positions=positions, player_id=player_id, log_fn=self._log)
                results.append(out)
            if db_name:
                out = analyze_database(csv_path, threshold=threshold, positions=positions, player_id=player_id, db_filter=db_name, log_fn=self._log)
                results.append(out)
            if seq_name:
                out = analyze_sequence(csv_path, seq_name=seq_name, threshold=threshold, positions=positions, player_id=player_id, log_fn=self._log)
                results.append(out)
            QMessageBox.information(self, "Success", "Saved:\n" + "\n".join(results))
        except Exception as e:
            self._log(f"ERROR: {e}")
            QMessageBox.critical(self, "Analysis failed", str(e))


def main():
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
