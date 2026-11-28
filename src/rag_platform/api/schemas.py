"""Pydantic models for the v1 API contract (one model per entity).

Response models carry no defaults, so optional values are explicit `null` and every
field is `required` in the OpenAPI document. Request models forbid unknown fields.
"""

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorCode(StrEnum):
    unauthenticated = "unauthenticated"
    forbidden = "forbidden"
    not_found = "not_found"
    validation_failed = "validation_failed"
    conflict = "conflict"
    rate_limited = "rate_limited"
    payload_too_large = "payload_too_large"
    dependency_unavailable = "dependency_unavailable"
    internal_error = "internal_error"


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    request_id: str
    details: dict[str, Any] | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class AnswerStatus(StrEnum):
    answered = "answered"
    abstained = "abstained"
    evidence_only = "evidence_only"


class CacheStatus(StrEnum):
    hit = "hit"
    miss = "miss"
    bypass = "bypass"


class DocumentStatus(StrEnum):
    received = "received"
    parsing = "parsing"
    chunking = "chunking"
    embedding = "embedding"
    indexing = "indexing"
    live = "live"
    quarantined = "quarantined"
    failed = "failed"


class JobState(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class IndexVersionStatus(StrEnum):
    building = "building"
    staged = "staged"
    committed = "committed"
    rolled_back = "rolled_back"
    failed = "failed"


class ServiceStatus(StrEnum):
    ok = "ok"
    degraded = "degraded"
    down = "down"


class ServiceLink(BaseModel):
    name: str
    url: str


class ConfigResponse(BaseModel):
    env: str
    version: str
    provider_mode: str
    enabled_chain: list[str]
    honesty_text: str
    service_links: list[ServiceLink]


class LoginRequest(_Request):
    username: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=1024)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class Me(BaseModel):
    user_id: str
    tenant_id: str
    roles: list[str]
    labels: list[str]


class Span(BaseModel):
    start: int
    end: int


class Document(BaseModel):
    id: str
    title: str
    status: DocumentStatus
    acl_labels: list[str]
    effective_date: date | None
    quarantine_reason: str | None
    created_at: datetime


class DocumentCreate(_Request):
    title: str = Field(min_length=1, max_length=512)
    filename: str = Field(min_length=1, max_length=255)
    content_type: str
    content_base64: str
    acl_labels: list[str] = Field(default_factory=list)
    effective_date: date | None = None


class DocumentAccepted(BaseModel):
    document_id: str
    job_id: str
    status: DocumentStatus


class DocumentPage(BaseModel):
    items: list[Document]
    next_cursor: str | None


class Chunk(BaseModel):
    document_id: str
    chunk_id: str
    parent_id: str | None = None
    document_title: str
    text: str
    heading_path: list[str]
    page: int | None
    span: Span


class Job(BaseModel):
    id: str
    kind: str
    state: JobState
    document_id: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime


class ActionResult(BaseModel):
    id: str
    status: str


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    parent_id: str | None = None
    span: Span


class Versions(BaseModel):
    index_version: str
    embedding_model: str
    prompt_version: str
    policy_hash: str | None = None


class Message(BaseModel):
    id: str
    chat_id: str
    content: str
    answer_status: AnswerStatus
    citations: list[Citation]
    versions: Versions
    served_model: str | None
    cache: CacheStatus
    trace_id: str
    latency_ms: int
    created_at: datetime


class MessageCreate(_Request):
    content: str = Field(min_length=1, max_length=8000)


class Chat(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatDetail(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[Message]


class ChatCreate(_Request):
    title: str | None = Field(default=None, max_length=200)


class ChatUpdate(_Request):
    title: str = Field(min_length=1, max_length=200)


class ChatPage(BaseModel):
    items: list[Chat]
    next_cursor: str | None


class Attempt(BaseModel):
    model: str
    outcome: str
    latency_ms: int
    tokens_in: int
    tokens_out: int


class StageLatency(BaseModel):
    stage: str
    latency_ms: int


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    score: float


class TraceSummary(BaseModel):
    trace_id: str
    message_id: str | None
    at: datetime
    outcome: str
    served_model: str | None
    cache: CacheStatus
    latency_ms: int


class TraceDetail(BaseModel):
    trace_id: str
    message_id: str | None
    at: datetime
    outcome: str
    served_model: str | None
    cache: CacheStatus
    latency_ms: int
    attempts: list[Attempt]
    fallback_reasons: list[str]
    stage_latency: list[StageLatency]
    retrieved: list[RetrievedChunk]


class TracePage(BaseModel):
    items: list[TraceSummary]
    next_cursor: str | None


class IndexVersion(BaseModel):
    version: str
    status: IndexVersionStatus
    gate_passed: bool | None
    gate_metrics: dict[str, float]
    parent: str | None
    created_at: datetime


class IndexVersionPage(BaseModel):
    items: list[IndexVersion]
    next_cursor: str | None


class QuarantineItem(BaseModel):
    id: str
    document_id: str
    reason: str
    detail: str
    created_at: datetime


class QuarantinePage(BaseModel):
    items: list[QuarantineItem]
    next_cursor: str | None


class RegistryPointer(BaseModel):
    name: str
    ref: str


class RegistryPointers(BaseModel):
    pointers: list[RegistryPointer]


class RegistryPointersUpdate(_Request):
    pointers: list[RegistryPointer]


class Flag(BaseModel):
    key: str
    value: bool | int | str


class FlagUpdate(_Request):
    value: bool | int | str


class AuditEntry(BaseModel):
    seq: int
    at: datetime
    actor: str
    action: str
    target: str
    hash: str


class AuditPage(BaseModel):
    items: list[AuditEntry]
    next_cursor: str | None


class AuditVerifyResult(BaseModel):
    ok: bool
    checked: int
    first_bad_seq: int | None


class ServiceHealth(BaseModel):
    name: str
    status: ServiceStatus
    link: str | None
    detail: str | None


class SystemHealth(BaseModel):
    services: list[ServiceHealth]


class HealthStatus(BaseModel):
    status: str
