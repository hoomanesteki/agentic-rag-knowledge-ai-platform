"""Messaging channels: surfaces that talk to the assistant over a chat app, not the web widget.

Each channel is a thin MCP client of mcp_server, so it inherits every gate (anonymous, order and
account PII blocked, grounded and cited). The engine stays domain agnostic: persona and brand come
from the active pack, never hardcoded here.
"""
