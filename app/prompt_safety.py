"""Prompt injection defenses and query sanitization utilities.
Ensures user input is safely wrapped in delimited blocks and cannot inject directives.
"""
import re

MAX_QUERY_LENGTH = 500


def sanitize_user_prompt(text: str) -> str:
    """Sanitize raw user text:
    1. Strip ASCII control characters (< 32 except newline/tab).
    2. Escape closing delimiter tag `</user_query>` to prevent jailbreak escapes.
    3. Truncate to MAX_QUERY_LENGTH characters.
    """
    if not text:
        return ""

    # Strip control characters (0x00 to 0x1f except tab \t and newline \n, \r)
    sanitized = "".join(ch for ch in text if ch in ("\t", "\n", "\r") or ord(ch) >= 32)

    # Escape delimiter tag </user_query>
    sanitized = re.sub(r"</user_query>", "&lt;/user_query&gt;", sanitized, flags=re.IGNORECASE)

    # Length cap
    if len(sanitized) > MAX_QUERY_LENGTH:
        sanitized = sanitized[:MAX_QUERY_LENGTH]

    return sanitized.strip()


def wrap_user_query(text: str) -> str:
    """Format sanitized user query inside unambiguous XML delimiter tags."""
    clean = sanitize_user_prompt(text)
    return f"<user_query>\n{clean}\n</user_query>"
