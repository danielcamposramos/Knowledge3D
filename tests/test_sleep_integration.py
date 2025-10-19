import shutil
import time
from pathlib import Path

import pytest
import numpy as np

from knowledge3d.cranium.bridges import pdf_ingestion_bridge


class DummyScheduler:
    def __init__(self, *args, **kwargs):
        self.idle_threshold = kwargs.get("idle_threshold", 300.0)
        self.log_path = kwargs.get("log_path")
        self.running = False
        self.mark_calls = 0
        self.last_activity = time.time()

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def mark_activity(self):
        self.mark_calls += 1
        self.last_activity = time.time()


@pytest.fixture(autouse=True)
def restore_font_db(monkeypatch, tmp_path):
    # Ensure glyph consolidator backup exists to avoid large changes during tests.
    backup = Path("/K3D/Knowledge3D.local/font_db_pre_consolidation.pkl")
    original = Path("/K3D/Knowledge3D.local/font_db.pkl")
    if backup.exists():
        shutil.copy2(backup, original)
    yield


def test_sleep_integration(monkeypatch):
    from knowledge3d.cranium.sleep import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "SleepScheduler", DummyScheduler)
    bridge = pdf_ingestion_bridge.PDFIngestionBridge()

    assert isinstance(bridge.sleep_scheduler, DummyScheduler)
    assert bridge.sleep_scheduler.running is True

    def fake_parse(self, pdf_bytes, pdf_buffer_gpu, buffer_size, page_num):
        return {
            "objects": np.zeros((0, 8), dtype=np.float32),
            "object_count": 0,
            "is_scanned": False,
        }

    bridge._parse_pdf_structure = fake_parse.__get__(bridge, type(bridge))

    initial_calls = bridge.sleep_scheduler.mark_calls
    pdf_path = (
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/"
        "How to think/Algorithmic.Thinking.BASE.pdf"
    )
    bridge.ingest_pdf_page(pdf_path, page_num=0)

    assert bridge.sleep_scheduler.mark_calls == initial_calls + 1
