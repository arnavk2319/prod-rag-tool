"""Shared wiring for the T-011 tests: service over the real Postgres, plus superuser reads."""

from ingest_helpers import Stack

from rag_platform.adapters.fakes import FakeAuditLog, FakeClock
from rag_platform.adapters.policy_fakes import FakeRegistryMirror
from rag_platform.adapters.sql_index_versions import SqlIndexVersionStore
from rag_platform.services.index_versions import IndexVersionService
from rag_platform.services.policy_gate import PolicyGate


def make_service(s: Stack, on_swap=None) -> IndexVersionService:  # type: ignore[no-untyped-def]
    gate = PolicyGate(s.policies, FakeAuditLog(), FakeClock(), FakeRegistryMirror())
    return IndexVersionService(SqlIndexVersionStore(s.dsn), s.policies, gate, on_swap=on_swap)
