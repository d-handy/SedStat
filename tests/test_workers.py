"""Tests for sedstat.gui.workers.BatchWorker."""

from __future__ import annotations

import tempfile
from pathlib import Path

from sedstat.gui.workers import BatchWorker

from sedstat.core.classification import ClassificationScheme

# ---------------------------------------------------------------------------
# Helpers — minimal synthetic .$av content (mirrors test_io_beckman_coulter.py)
# ---------------------------------------------------------------------------


def _make_av_content(
    *,
    n_bins: int = 5,
    include_sizestats: bool = True,
    include_sisave: bool = True,
    include_obs: bool = True,
) -> str:
    """Return a minimal .$av file string.

    Args:
        n_bins: Number of size classes (bin boundaries and heights are
            truncated to this many entries).
        include_sizestats: Whether to emit a [SizeStats] section.
        include_sisave: Whether to emit a [SIsave0] section.
        include_obs: Whether to emit [SizeN] Obs= blocks.

    Returns:
        The synthesized .$av file content.
    """
    boundaries = "\n".join(f"{v:.4f}" for v in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0][: n_bins + 1])
    heights = "\n".join(f"{v:.4f}" for v in [5.0, 20.0, 50.0, 20.0, 5.0][:n_bins])

    sizestats = ""
    if include_sizestats:
        sizestats = (
            "[SizeStats]\n"
            "Mean= 18.5\n"
            "Mode= 16.0\n"
            "Median= 17.2\n"
            "SD= 2.4\n"
            "FWMean= 6.3\n"
            "FWMedian= 6.1\n"
            "FWSD= 1.8\n"
            "FWSkew= 0.05\n"
            "FWKurt= 1.1\n"
        )

    sisave = ""
    if include_sisave:
        sisave = "[SIsave0]\nGroupID=TestGroup\nDensity=2.65\nOperator=JD\n"

    obs_blocks = ""
    if include_obs:
        obs_blocks = "[Size0]\nObs= 12.5\n[Size1]\nObs= 11.8\n[Size2]\nObs= 13.1\n"

    return (
        "[common]\n"
        "Program=Beckman Coulter LS\n"
        "\n" + sisave + "\n"
        "[#Bindiam]\n" + boundaries + "\n\n"
        "[#Binheight]\n" + heights + "\n\n" + sizestats + obs_blocks
    )


def _write_temp_av(content: str) -> Path:
    with tempfile.NamedTemporaryFile(
        suffix=".$av", delete=False, mode="w", encoding="latin-1"
    ) as tmp:
        tmp.write(content)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Successful processing
# ---------------------------------------------------------------------------


class TestBatchWorkerSuccess:
    """Successful runs emit result_ready per file and progress signals."""

    def test_result_ready_emitted_for_each_file(self, qtbot):
        paths = [_write_temp_av(_make_av_content()) for _ in range(2)]
        worker = BatchWorker(paths)

        results = []
        errors = []
        progress_calls = []
        worker.result_ready.connect(lambda p, r, rec: results.append((p, r, rec)))
        worker.error_occurred.connect(lambda p, e: errors.append((p, e)))
        worker.progress.connect(lambda i, t: progress_calls.append((i, t)))

        worker.run()

        assert len(results) == 2
        assert errors == []
        assert progress_calls == [(1, 2), (2, 2)]

    def test_result_ready_payload_types(self, qtbot):
        from sedstat.core.results import GrainSizeResult
        from sedstat.io.beckman_coulter import LSRecord

        p = _write_temp_av(_make_av_content())
        worker = BatchWorker([p])

        captured = {}

        def _on_result(path_str, result, rec):
            captured["path"] = path_str
            captured["result"] = result
            captured["rec"] = rec

        worker.result_ready.connect(_on_result)
        worker.run()

        assert captured["path"] == str(p)
        assert isinstance(captured["result"], GrainSizeResult)
        assert isinstance(captured["rec"], LSRecord)

    def test_scheme_is_passed_through(self, qtbot):
        p = _write_temp_av(_make_av_content())
        worker = BatchWorker([p], scheme=ClassificationScheme.USDA)
        assert worker._scheme == ClassificationScheme.USDA
        # Should still process successfully with a non-default scheme.
        results = []
        worker.result_ready.connect(lambda *args: results.append(args))
        worker.run()
        assert len(results) == 1

    def test_using_start_emits_via_qthread(self, qtbot):
        """Exercise the real QThread.start() path with waitSignal."""
        p = _write_temp_av(_make_av_content())
        worker = BatchWorker([p])
        with qtbot.waitSignal(worker.result_ready, timeout=5000):
            worker.start()
        worker.wait(5000)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestBatchWorkerErrors:
    """Failures on individual files emit error_occurred without aborting the batch."""

    def test_nonexistent_path_emits_error_and_continues(self, qtbot):
        good = _write_temp_av(_make_av_content())
        bad = Path("/nonexistent/path/does_not_exist.$av")
        worker = BatchWorker([bad, good])

        results = []
        errors = []
        progress_calls = []
        worker.result_ready.connect(lambda p, r, rec: results.append(p))
        worker.error_occurred.connect(lambda p, e: errors.append((p, e)))
        worker.progress.connect(lambda i, t: progress_calls.append((i, t)))

        worker.run()

        assert len(errors) == 1
        assert errors[0][0] == str(bad)
        assert len(results) == 1
        assert results[0] == str(good)
        # Both files should have reported progress despite the error.
        assert progress_calls == [(1, 2), (2, 2)]

    def test_all_bad_paths_emit_only_errors(self, qtbot):
        bad_paths = [
            Path("/nonexistent/one.$av"),
            Path("/nonexistent/two.$av"),
        ]
        worker = BatchWorker(bad_paths)

        results = []
        errors = []
        worker.result_ready.connect(lambda p, r, rec: results.append(p))
        worker.error_occurred.connect(lambda p, e: errors.append(p))

        worker.run()

        assert results == []
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestBatchWorkerCancellation:
    """cancel() stops processing before or during run() and emits cancelled."""

    def test_cancel_before_run_emits_cancelled_and_processes_nothing(self, qtbot):
        paths = [_write_temp_av(_make_av_content()) for _ in range(3)]
        worker = BatchWorker(paths)
        worker.cancel()

        results = []
        cancelled_calls = []
        worker.result_ready.connect(lambda p, r, rec: results.append(p))
        worker.cancelled.connect(lambda: cancelled_calls.append(True))

        worker.run()

        assert results == []
        assert cancelled_calls == [True]

    def test_cancel_mid_loop_stops_processing_remaining_files(self, qtbot):
        paths = [_write_temp_av(_make_av_content()) for _ in range(3)]
        worker = BatchWorker(paths)

        results = []
        cancelled_calls = []

        def _on_result(p, r, rec):
            results.append(p)
            if len(results) == 1:
                worker.cancel()

        worker.result_ready.connect(_on_result)
        worker.cancelled.connect(lambda: cancelled_calls.append(True))

        worker.run()

        # Only the first file is processed; cancellation stops before file 2.
        assert len(results) == 1
        assert cancelled_calls == [True]

    def test_cancel_flag_set_by_cancel_method(self, qtbot):
        worker = BatchWorker([])
        assert worker._cancelled is False
        worker.cancel()
        assert worker._cancelled is True
