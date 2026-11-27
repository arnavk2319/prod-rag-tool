import json

import pytest
import structlog

from rag_platform.logging_setup import configure_logging

SECRETS = ["what is jane doe's salary", "the salary is 123456", "CONFIDENTIAL DOC BODY"]


def test_logs_have_trace_id_and_no_content(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id="abc12345")
    structlog.get_logger().info(
        "answered",
        question=SECRETS[0],
        answer=SECRETS[1],
        document=SECRETS[2],
        nested={"text": SECRETS[2]},
        chunks=3,
    )
    out = capsys.readouterr().out
    rec = json.loads(out.strip().splitlines()[-1])
    assert rec["trace_id"] == "abc12345"
    assert rec["chunks"] == 3
    for s in SECRETS:
        assert s not in out
    structlog.contextvars.clear_contextvars()


def test_redaction_is_case_insensitive_and_nested_lists(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    structlog.contextvars.clear_contextvars()
    canary = "CANARY-9f3b-do-not-log"
    structlog.get_logger().info(
        "evt",
        Question=canary,
        items=[{"content": canary}, {"prompt": canary, "ok": "visible"}],
        meta={"deep": {"answer": canary}},
    )
    out = capsys.readouterr().out
    assert canary not in out
    rec = json.loads(out.strip().splitlines()[-1])
    assert rec["items"][1]["ok"] == "visible"
    assert rec["event"] == "evt"
    assert rec["level"] == "info"
    assert "timestamp" in rec


def test_log_level_filters(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING")
    structlog.get_logger().info("quiet")
    structlog.get_logger().warning("loud")
    out = capsys.readouterr().out
    assert "quiet" not in out
    assert "loud" in out
    configure_logging("INFO")
