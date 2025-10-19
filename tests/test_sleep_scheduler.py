import time

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.sleep.scheduler import SleepScheduler


def test_idle_detection():
    """Scheduler should detect idle duration threshold crossings."""
    engine = RPNEmbeddingEngine()
    scheduler = SleepScheduler(rpn_engine=engine, idle_threshold=2.0)

    scheduler.mark_activity()
    time.sleep(1.1)
    assert (time.time() - scheduler.last_activity) < scheduler.idle_threshold

    time.sleep(1.1)
    assert (time.time() - scheduler.last_activity) > scheduler.idle_threshold


def test_scheduler_start_stop():
    engine = RPNEmbeddingEngine()
    scheduler = SleepScheduler(rpn_engine=engine, idle_threshold=300.0)

    scheduler.start()
    assert scheduler.running is True
    assert scheduler.thread is not None

    scheduler.stop()
    assert scheduler.running is False
