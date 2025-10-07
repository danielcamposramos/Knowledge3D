from __future__ import annotations

"""
Minimal Tkinter tablet UI that consumes the mmap log emitted by the fused head.

The UI is intentionally simple – it keeps all logic self-contained so the file
can be executed directly (`python viewer/tablet/tablet_ui.py`) during manual
tests or demos.  The design mirrors the Step7.2 blueprint: logs are colour
coded by confidence, and the interface provides quick filters for action types.
"""

import json
import mmap
import threading
import time
from pathlib import Path
from typing import Dict, Iterable

import tkinter as tk
from tkinter import scrolledtext, ttk

MMAP_PATH = Path("tablet_log.mmap")
MMAP_SIZE = 4 * 1024 * 1024
UPDATE_INTERVAL_MS = 100


class TabletLogReader:
    """Zero-copy mmap reader shared between UI and tests."""

    def __init__(self, mmap_path: Path = MMAP_PATH, buffer_size: int = MMAP_SIZE) -> None:
        self.path = mmap_path
        self.buffer_size = buffer_size
        if not self.path.exists():
            raise FileNotFoundError(
                f"Tablet mmap file not found at {self.path}. Start the fused head pipeline first."
            )
        self._file = self.path.open("r+b")
        self._mmap = mmap.mmap(self._file.fileno(), self.buffer_size)
        self._offset = 0

    def read_available(self) -> Iterable[Dict[str, object]]:
        self._mmap.seek(self._offset)
        chunk = self._mmap.read(self.buffer_size)
        if not chunk:
            return []
        lines = chunk.split(b"\n")
        messages = []
        for line in lines:
            if not line.strip():
                continue
            try:
                messages.append(json.loads(line.decode("utf-8")))
            except json.JSONDecodeError:
                continue
        self._offset = self._mmap.tell()
        if self._offset >= self.buffer_size:
            self._offset = 0
        return messages


class TabletUI:
    def __init__(self, master: tk.Tk, reader: TabletLogReader) -> None:
        self.master = master
        self.reader = reader
        self.master.title("Knowledge3D Tablet")

        self.log_widget = scrolledtext.ScrolledText(master, wrap=tk.WORD, state=tk.DISABLED, height=20)
        self.log_widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        controls = tk.Frame(master)
        controls.pack(fill=tk.X, padx=8, pady=(0, 8))

        tk.Label(controls, text="Filter:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar(value="All")
        self.filter_combo = ttk.Combobox(
            controls, textvariable=self.filter_var, state="readonly", values=["All", "dialogue", "memory_write", "tablet_update"]
        )
        self.filter_combo.pack(side=tk.LEFT, padx=4)

        self.filter_combo.bind("<<ComboboxSelected>>", lambda _: None)

        clear_btn = ttk.Button(controls, text="Clear", command=self._clear_log)
        clear_btn.pack(side=tk.RIGHT)

        self._lock = threading.Lock()
        self._running = True
        self._worker = threading.Thread(target=self._update_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    def _clear_log(self) -> None:
        self.log_widget.config(state=tk.NORMAL)
        self.log_widget.delete("1.0", tk.END)
        self.log_widget.config(state=tk.DISABLED)

    def _append(self, entry: Dict[str, object]) -> None:
        filter_value = self.filter_var.get()
        entry_type = entry.get("type", "unknown")
        if filter_value != "All" and entry_type != filter_value:
            return

        confidence = float(entry.get("confidence", 0.0))
        prefix = f"[{entry.get('timestamp', 0)}] [{entry_type}] "
        colour = "#4caf50" if confidence >= 0.7 else "#ff9800"

        text = json.dumps(entry, ensure_ascii=False)
        self.log_widget.config(state=tk.NORMAL)
        self.log_widget.insert(tk.END, prefix, ("prefix",))
        self.log_widget.insert(tk.END, text + "\n", (colour,))
        self.log_widget.config(state=tk.DISABLED)
        self.log_widget.see(tk.END)

    def _update_loop(self) -> None:
        while self._running:
            with self._lock:
                for entry in self.reader.read_available():
                    self.master.after(0, self._append, entry)
            time.sleep(UPDATE_INTERVAL_MS / 1000.0)

    def stop(self) -> None:
        self._running = False
        self._worker.join(timeout=1.0)


def main() -> None:
    reader = TabletLogReader()
    root = tk.Tk()
    ui = TabletUI(root, reader)
    try:
        root.mainloop()
    finally:
        ui.stop()


if __name__ == "__main__":
    main()

