"""Collapse child hits into parent passages (pure; the SQL retrieval itself is a later ticket).

Rule: keep the best-scoring child per parent, order parents by (-best score, parent_id), keep at
most `top_n`, then trim to `token_budget` tokens. The result is deterministic: ties break on
parent id, then child id. A parent must match its child on tenant, index version, document id
and ACL labels, else the child is dropped.
Duplicate hits of one child id are merged (best score kept).
Budget flow: parents are taken in rank order; a parent larger than the remaining budget is cut to
it (`truncated=True`, children outside the visible prefix dropped) and uses all of what is left,
so when the first parent alone exceeds the budget it is truncated and later parents get nothing.
A child whose parent is not in `parents` (forged or stale id, other
tenant or index version) is dropped, never served.
"""

from collections.abc import Mapping, Sequence

from rag_platform.domain.ports import ChildMatch, ParentChunk, RetrievedPassage, ScoredChunk
from rag_platform.services.ingest.chunker import truncate_tokens


def collapse_to_parents(
    hits: Sequence[ScoredChunk],
    parents: Mapping[str, ParentChunk],
    *,
    top_n: int,
    token_budget: int,
) -> list[RetrievedPassage]:
    by_parent: dict[str, list[ScoredChunk]] = {}
    for h in hits:
        parent = parents.get(h.chunk.parent_id)
        if parent is None or parent.tenant_id != h.chunk.tenant_id:
            continue  # orphan or cross-tenant link: dropped
        if (
            parent.index_version != h.chunk.index_version
            or parent.doc_id != h.chunk.doc_id
            or parent.acl_labels != h.chunk.acl_labels
        ):
            continue  # defence in depth: other document or different ACL labels
        by_parent.setdefault(parent.parent_id, []).append(h)
    ranked = sorted(
        by_parent.items(),
        key=lambda kv: (-max(h.score for h in kv[1]), kv[0]),
    )[: max(top_n, 0)]
    out: list[RetrievedPassage] = []
    remaining = token_budget
    for pid, kids in ranked:
        if remaining <= 0:
            break
        parent = parents[pid]
        text, truncated = parent.text, False
        if parent.token_count > remaining:
            text, truncated = truncate_tokens(parent.text, remaining), True
        visible_end = parent.span[0] + len(text)
        best: dict[str, ChildMatch] = {}
        for h in kids:  # duplicate hits of one child merge, keeping the best score
            cur = best.get(h.chunk.chunk_id)
            if cur is None or h.score > cur.score:
                best[h.chunk.chunk_id] = ChildMatch(h.chunk.chunk_id, h.score, h.chunk.span)
        matches = sorted(best.values(), key=lambda m: (-m.score, m.chunk_id))
        if truncated:
            matches = [m for m in matches if m.span[0] < visible_end]
            if not matches:
                continue  # none of the matched children is visible after trimming
        used = remaining if truncated else parent.token_count
        out.append(
            RetrievedPassage(
                parent_id=pid,
                doc_id=parent.doc_id,
                text=text,
                token_count=min(parent.token_count, remaining),
                score=max(h.score for h in kids),
                children=tuple(matches),
                truncated=truncated,
            )
        )
        remaining -= used
    return out
