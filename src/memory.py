"""
Conversation memory for RAG, with query rewriting (contextualization).

The subtle problem this solves: a follow-up like "and who pays it?" is
useless for retrieval on its own -- embedding it retrieves garbage because
the question doesn't contain its own subject. So BEFORE retrieving, we use
the conversation history to rewrite the follow-up into a standalone question
("Who pays the DLD transfer fee on an off-plan resale?"), THEN retrieve on
that. The history is also passed to generation so answers flow naturally.

Design:
  - ConversationMemory holds the running list of (question, answer) turns.
  - contextualize_query() rewrites a follow-up into a standalone query using
    the LLM + recent history. If it's already standalone (first turn, or a
    fresh topic), it returns the question unchanged.
  - history is capped (MAX_TURNS) so the prompt doesn't grow without bound.

Single-conversation (one shared memory) -- fine for a demo. Multi-session
would key memories by a session id.
"""
import os, re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REWRITE_MODEL = "claude-haiku-4-5"
MAX_TURNS = 6   # how many recent turns to keep for context


def _client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set -- see generate_llm.py for setup.")
    import anthropic
    return anthropic.Anthropic(api_key=key)


class ConversationMemory:
    """Holds recent turns of a single conversation."""
    def __init__(self, max_turns=MAX_TURNS):
        self.turns = []          # list of {"question":..., "answer":...}
        self.max_turns = max_turns

    def add(self, question, answer):
        self.turns.append({"question": question, "answer": answer})
        # keep only the most recent max_turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def history_text(self):
        """Recent turns formatted for a prompt."""
        if not self.turns:
            return ""
        lines = []
        for t in self.turns:
            lines.append(f"User: {t['question']}")
            lines.append(f"Assistant: {t['answer']}")
        return "\n".join(lines)

    def clear(self):
        self.turns = []


class SessionMemoryStore:
    """
    Holds one ConversationMemory per session id, so concurrent users don't
    share (or corrupt) each other's conversation. Keyed by an opaque session
    id supplied by the client.

    In-memory + capped by count with simple LRU-ish eviction. Production
    would back this with Redis (fast, shared across app servers, built-in
    TTL expiry) instead of a process-local dict -- but the interface
    (get/reset by session id) would be identical.
    """
    def __init__(self, max_sessions=500):
        self._sessions = {}          # session_id -> ConversationMemory
        self._order = []             # session ids, oldest-touched first
        self.max_sessions = max_sessions

    def get(self, session_id):
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory()
            self._order.append(session_id)
            self._evict_if_needed()
        else:
            # mark as recently used
            self._order.remove(session_id)
            self._order.append(session_id)
        return self._sessions[session_id]

    def reset(self, session_id):
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def _evict_if_needed(self):
        # drop the least-recently-used sessions if we're over the cap
        while len(self._order) > self.max_sessions:
            oldest = self._order.pop(0)
            self._sessions.pop(oldest, None)

    def active_sessions(self):
        return len(self._sessions)


REWRITE_SYSTEM = """You rewrite a user's latest question into a standalone question that can be understood without the conversation history.

Rules:
- If the latest question refers to something earlier (pronouns like "it", "that", "they", or an implied subject), resolve the reference using the history and produce a full, self-contained question.
- If the latest question is already self-contained, or starts a new topic, return it unchanged.
- Output ONLY the rewritten question, nothing else. No preamble, no quotes."""


def contextualize_query(question, memory, model=REWRITE_MODEL):
    """
    Rewrite a follow-up question into a standalone one using recent history.
    Returns (rewritten_question, was_rewritten).
    On the first turn (no history) returns the question unchanged, no LLM call.
    """
    history = memory.history_text()
    if not history:
        return question, False   # nothing to contextualize against

    client = _client()
    user_msg = f"CONVERSATION HISTORY:\n{history}\n\nLATEST QUESTION: {question}\n\nRewrite the latest question to be standalone:"
    resp = client.messages.create(
        model=model, max_tokens=200, system=REWRITE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    rewritten = "".join(b.text for b in resp.content if b.type == "text").strip()
    # guard: if the model returned something empty or absurdly long, fall back
    if not rewritten or len(rewritten) > 500:
        return question, False
    was_rewritten = rewritten.strip().lower() != question.strip().lower()
    return rewritten, was_rewritten


if __name__ == "__main__":
    # demo of the contextualization
    mem = ConversationMemory()
    mem.add("What is the DLD transfer fee on an off-plan resale?",
            "The DLD transfer fee is 4% of the sale price.")
    try:
        rewritten, changed = contextualize_query("And who pays it?", mem)
        print("Original:  And who pays it?")
        print("Rewritten:", rewritten)
        print("Changed:", changed)
    except RuntimeError as e:
        print("Could not run:", e)