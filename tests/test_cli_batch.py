"""The `fleetlm batch` and `fleetlm up` helpers.

These cover the parts a user hits before any network call: how a JSONL file is
turned into requests, what happens to a line that is malformed, and whether the
port check actually detects a port in use.
"""

import socket
from argparse import Namespace

from node_agent.cli import _load_requests, _port_is_free, _progress


def _args(max_tokens=256, temperature=0.7):
    return Namespace(max_tokens=max_tokens, temperature=temperature)


def _write(tmp_path, *lines):
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# ── reading the input file ──────────────────────────────────────────────

def test_a_bare_prompt_becomes_a_user_message(tmp_path):
    path = _write(tmp_path, '{"prompt": "hello"}')

    requests, problems = _load_requests(path, _args())

    assert problems == []
    assert requests[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_an_explicit_messages_array_is_passed_through(tmp_path):
    path = _write(tmp_path, '{"messages": [{"role": "system", "content": "be terse"}]}')

    requests, _ = _load_requests(path, _args())

    assert requests[0]["messages"][0]["role"] == "system"


def test_per_line_settings_override_the_defaults(tmp_path):
    path = _write(tmp_path, '{"prompt": "hi", "max_tokens": 9, "temperature": 0.1}')

    requests, _ = _load_requests(path, _args(max_tokens=256, temperature=0.7))

    assert requests[0]["max_tokens"] == 9
    assert requests[0]["temperature"] == 0.1


def test_blank_lines_are_ignored(tmp_path):
    path = _write(tmp_path, '{"prompt": "a"}', "", "   ", '{"prompt": "b"}')

    requests, problems = _load_requests(path, _args())

    assert len(requests) == 2
    assert problems == []


# ── malformed input ─────────────────────────────────────────────────────

def test_a_broken_line_is_reported_by_number_and_does_not_abort_the_run(tmp_path):
    """One bad line in a 10,000-line file must not cost the whole batch."""
    path = _write(tmp_path, '{"prompt": "good"}', "{not json", '{"prompt": "also good"}')

    requests, problems = _load_requests(path, _args())

    assert len(requests) == 2
    assert len(problems) == 1
    assert "line 2" in problems[0]


def test_a_line_with_no_usable_field_is_reported(tmp_path):
    path = _write(tmp_path, '{"nothing": "useful"}')

    requests, problems = _load_requests(path, _args())

    assert requests == []
    assert "line 1" in problems[0] and "messages" in problems[0]


# ── progress rendering ──────────────────────────────────────────────────

def test_progress_shows_counts_and_a_pending_eta_before_anything_finishes():
    line = _progress({"completed": 0, "failed": 0, "in_flight": 4}, 10, __import__("time").monotonic())

    assert "0/10" in line
    assert "ETA --" in line  # no rate yet, so no fabricated estimate


def test_progress_counts_failures_as_done():
    """Otherwise a batch with failures never reaches 100% and the ETA never ends."""
    line = _progress({"completed": 6, "failed": 2, "in_flight": 0}, 8, __import__("time").monotonic() - 4)

    assert "8/8" in line


# ── port checking ───────────────────────────────────────────────────────

def test_a_free_port_reads_as_free():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert _port_is_free("127.0.0.1", port) is True


def test_a_port_in_use_reads_as_taken():
    """Regression: SO_REUSEADDR made this return True while a server was bound."""
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]

        assert _port_is_free("127.0.0.1", port) is False
