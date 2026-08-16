"""
Access control for retrieval.

WHY THIS EXISTS
---------------
In a real organization, not every user may see every document. Finance,
legal, and general staff have different entitlements. RAG that ignores this
can leak documents *between employees* -- often a bigger real-world risk
than the LLM provider ever was.

This module enforces that at RETRIEVAL time: each user has a role, each role
maps to the set of doc_types it may see, and every query is filtered to that
set BEFORE anything reaches the LLM. A user who can't see regulations simply
never has regulation chunks retrieved, generated from, or cited.

DESIGN
------
- doc_type already lives on every chunk's metadata (set in ingest.py), so we
  filter on it directly -- no re-tagging of the corpus required.
- The filter is expressed as a ChromaDB `where` clause, so it's applied by
  the vector store itself, not as a fragile post-filter on results.
- Roles are defined in ONE place (ROLE_VISIBILITY) so entitlements are
  auditable and easy to change. In production this table would come from
  your identity provider / entitlement service; here it's explicit so the
  demo can show the mechanism.

DEMO vs PRODUCTION
------------------
For the demo, a role is passed in per request (or per session). In
production you'd derive the role from the authenticated user (SSO / Entra
ID / IAM), never trust a client-supplied role. The ENFORCEMENT code is
identical either way -- only the SOURCE of the role changes. That's the
seam this module is built around.
"""
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# The doc_types that exist in the corpus (mirrors CORPUS in ingest.py).
# Kept here as the canonical set so role definitions can be validated
# against it -- a typo'd doc_type in a role would otherwise silently grant
# access to nothing (or, worse, be missed).
# ---------------------------------------------------------------------------
ALL_DOC_TYPES = {
    "reference",
    "regulation",
    "market_data",
    "listings",
    "listings_data",
}

# Sentinel meaning "no restriction" -- an admin/analyst who sees everything.
ALL = "__ALL__"


# ---------------------------------------------------------------------------
# Role -> visible doc_types. THE single source of truth for entitlements.
#
# Edit here to change who sees what. Each value is either the ALL sentinel
# (sees everything) or a set of doc_types the role is permitted to retrieve.
#
# The example roles below are illustrative of how you'd carve a real org:
#   - analyst:    full access (internal analyst / admin)
#   - agent:      customer-facing sales -- listings only, no legal/market data
#   - legal:      regulation + reference, not commercial listings/market data
#   - public:     the safe public-facing subset
# Adjust to match the real entitlement model once governance defines it.
# ---------------------------------------------------------------------------
ROLE_VISIBILITY = {
    "analyst": ALL,
    "agent":   {"listings", "listings_data", "reference"},
    "legal":   {"regulation", "reference"},
    "public":  {"reference", "listings"},
}

DEFAULT_ROLE = "public"   # least-privilege default when no role is supplied


class AccessError(ValueError):
    """Raised when a role is unknown -- fail closed, never fail open."""


@dataclass(frozen=True)
class AccessPolicy:
    """Resolved entitlements for one user/role."""
    role: str
    visible_doc_types: frozenset   # empty frozenset is possible (sees nothing)
    unrestricted: bool             # True == sees every doc_type

    def can_see(self, doc_type: str) -> bool:
        return self.unrestricted or doc_type in self.visible_doc_types

    def where_clause(self) -> Optional[dict]:
        """
        Build the ChromaDB `where` filter for this policy.

        - unrestricted  -> None (no filter; retrieve across all doc_types)
        - one doc_type  -> {"doc_type": {"$eq": "listings"}}
        - several       -> {"doc_type": {"$in": ["listings", "reference"]}}
        - none          -> a filter that matches nothing, so a role with zero
                           entitlements retrieves zero chunks (fail closed),
                           rather than accidentally matching everything.
        """
        if self.unrestricted:
            return None
        vis = sorted(self.visible_doc_types)
        if len(vis) == 1:
            return {"doc_type": {"$eq": vis[0]}}
        if len(vis) > 1:
            return {"doc_type": {"$in": vis}}
        # zero visible doc_types: match an impossible value so nothing returns
        return {"doc_type": {"$eq": "__none__"}}


def resolve_policy(role: Optional[str]) -> AccessPolicy:
    """
    Turn a role name into an AccessPolicy. Unknown roles raise (fail closed);
    a missing role falls back to the least-privilege DEFAULT_ROLE rather than
    to full access.
    """
    if role is None:
        role = DEFAULT_ROLE
    role = role.strip().lower()
    if role not in ROLE_VISIBILITY:
        raise AccessError(
            f"Unknown role {role!r}. Known roles: {sorted(ROLE_VISIBILITY)}. "
            f"Refusing to grant access to an unrecognized role."
        )
    entitlement = ROLE_VISIBILITY[role]
    if entitlement == ALL:
        return AccessPolicy(role=role, visible_doc_types=frozenset(ALL_DOC_TYPES),
                            unrestricted=True)
    # validate the role's doc_types against the known set -- a typo here
    # should surface loudly, not silently narrow access
    unknown = set(entitlement) - ALL_DOC_TYPES
    if unknown:
        raise AccessError(
            f"Role {role!r} references unknown doc_type(s) {sorted(unknown)}. "
            f"Fix ROLE_VISIBILITY. Known doc_types: {sorted(ALL_DOC_TYPES)}."
        )
    return AccessPolicy(role=role, visible_doc_types=frozenset(entitlement),
                        unrestricted=False)


def filter_chunks_by_policy(chunks, policy: AccessPolicy):
    """
    Defense-in-depth: even after a where-filtered vector query, drop any chunk
    the policy doesn't permit. This catches the BM25/sparse path too, which
    does NOT go through Chroma's where clause -- keyword search would
    otherwise surface a chunk the user shouldn't see. Belt and suspenders:
    the where clause handles the dense path efficiently; this guarantees the
    invariant holds for whatever comes back, from any retrieval path.
    """
    if policy.unrestricted:
        return chunks
    return [c for c in chunks if policy.can_see(c.get("doc_type", ""))]