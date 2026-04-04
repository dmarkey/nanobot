# Fireworks Internal Error Investigation

## Problem
Fireworks AI returns `Error: {"error":"Internal error"}` consistently for session `telegram:8310624420` (~25K tokens, 27 messages).

## What we know
- The error is **not** caused by the upstream merge — same issue on commit `c557bf4` (pre-merge).
- Simple API calls to `accounts/fireworks/models/minimax-m2p5` work fine.
- Large synthetic conversations (51 messages, ~25K tokens) with `tool_choice='required'` also work fine.
- The error reproduces consistently for this specific session, suggesting something in the **actual conversation content** triggers it (not payload size alone).

## Likely culprits
- Special characters or encoding in a message
- A tool call result with unusual structure (e.g., large JSON, nested objects)
- Image or media content in the conversation history
- A message format that Fireworks doesn't handle (e.g., multi-part content blocks)

## Next steps
1. Inspect the actual session messages for `telegram:8310624420` on the server
2. Try sending subsets of the conversation to isolate which message triggers the error
3. Check if any messages contain images, tool results, or non-standard content blocks
4. Binary search: send first half, then second half, narrow down the offending message

## Fix applied
Added `"internal error"` to `_TRANSIENT_ERROR_MARKERS` in `nanobot/providers/base.py` so these errors are retried (3 attempts with backoff). This helps for transient cases but won't fix a persistent content-triggered error.
