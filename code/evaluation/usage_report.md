# Token Usage & Cost Report

Covers the exact multimodal-extraction and explanation-generation calls
that produced the submitted `output.csv` (250 rows, `dataset/requests.csv`).
The deterministic phases (financial forecasting, plan ranking) make no
LLM calls at all — see `code/simulate.py` / `code/plans.py`.

## Summary

| | |
|---|---|
| Total model calls | 482 |
| Total input tokens | 208,826 |
| Total output tokens | 28,883 |
| Total tokens | 237,709 |
| Average tokens / request (n=250) | 950.8 |
| Estimated total cost | ≈ $0.0027 (see "Cost estimate" below — most calls run under an existing flat-rate plan with $0 marginal cost) |
| Estimated cost / request | ≈ $0.000011 |

## Per-model / per-call-type breakdown

| Provider | Model | Purpose | Calls | Input tokens | Output tokens | Total tokens |
|---|---|---|---:|---:|---:|---:|
| Groq | `meta-llama/llama-4-scout-17b-16e-instruct` | Vision: read the amount off the 16 blank-amount events' linked images | 16 | ~22,928† | ~667† | ~23,595† |
| Featherless AI | `Qwen/Qwen2.5-72B-Instruct` | Text: structured signal extraction from all 215 messages | 216‡ | 106,370 | 15,557 | 121,927 |
| Featherless AI | `Qwen/Qwen2.5-72B-Instruct` | Text: `decision_explanation` wording for all 250 requests | 250 | 79,528 | 12,659 | 92,187 |
| **Total** | | | **482** | **208,826** | **28,883** | **237,709** |

† Estimated, not an exact API-reported figure — see "Vision figures are estimated" below.
‡ 215 messages, 216 billed calls: one message's first response was malformed JSON and was retried once (see `code/extraction.py`); both attempts were real, billed API calls, so both are counted.

**Featherless-only subtotal** (message parsing + explanations, same provider/model): 466 calls, 185,898 input tokens, 28,216 output tokens, 214,114 total tokens.

## Cost estimate

**Featherless AI — 466 calls, 214,114 tokens.** Billed under Featherless's
flat-rate "Chat Plan" (`feather_pro_plus`: $25/month, unlimited tokens,
4 concurrent request units) — confirmed directly against
[featherless.ai/pricing](https://featherless.ai/pricing) and against the
account's own API error messages, which named this exact plan. Every
call in this run falls under that existing flat monthly rate, so the
**marginal cost of these specific calls is $0**. The $25/month
subscription itself is a fixed cost the user carries independent of this
run's volume, not a per-call or per-token charge, so it is not
apportioned into the per-request estimate above.

**Groq — 16 calls, ~23,595 tokens.** Groq's own pricing page renders
pricing client-side and did not expose a queryable rate table to
automated fetching at the time of writing. Using a third-party-reported
rate for this model (~$0.11 / 1M input tokens, ~$0.34 / 1M output
tokens; not independently confirmed against Groq's first-party pricing):

- Input: 22,928 × $0.11 / 1,000,000 ≈ $0.0025
- Output: 667 × $0.34 / 1,000,000 ≈ $0.0002
- **Subtotal ≈ $0.0027**

**Total estimated cost for this run: ≈ $0.0027**, i.e. effectively
negligible — driven entirely by the small Groq vision step, since the
much larger Featherless volume costs $0 marginally under the flat plan
already in place.

## Vision figures are estimated, not measured — here's why

The 16 blank-amount images were extracted via Groq's
`meta-llama/llama-4-scout-17b-16e-instruct` on 2026-09-12, and those
results (not the token counts) are exactly what feeds `output.csv` today
via `code/.cache/vision_extractions.json`. Two things happened before an
exact, independently-verified token count could be produced for this
report:

1. The original run's own usage logging had a caching-related gap: a
   later, fully-cached rerun of the same command computed an empty
   usage summary (since cache hits skip the token-counting code path
   entirely) and overwrote the same log file the earlier, real-usage run
   had written to, before that number could be durably saved elsewhere.
2. By 2026-09-13, when this report was prepared, Groq had removed
   `meta-llama/llama-4-scout-17b-16e-instruct` from its available models
   entirely (confirmed by querying Groq's own `/v1/models` endpoint,
   which no longer lists it, and by a live `404 model_not_found` when
   attempting to reproduce the call) — so the original calls could not
   be independently re-measured against the same model at all.

Given that, the figures above are estimated by:

- reconstructing the exact system+user prompt text sent for each of the
  16 calls (deterministic given the event data) and estimating input
  text tokens at ~4 characters/token,
- adding ~1,100 tokens/call for image encoding, anchored to the one real
  measurement available for a comparable prompt+image pair on a
  different provider (Gemini, ~1,398 total input tokens for a similar
  prompt, of which ~330 was the text portion),
- estimating output tokens the same way (~4 chars/token) from the
  actual cached response text -- this part reflects the real returned
  content, not a guess about what was said, only an approximate
  conversion of that real text into a token count.

## What *was* independently re-verified

The `message_parse` and `explanation` figures above are exact: a fresh,
completely uncached re-run against Featherless (unaffected by Groq's
catalog change) on 2026-09-13 produced real, freshly-measured token
counts for all 215 messages and, separately, all 250 explanations.
Comparing that fresh run's extracted message intents against the
production cache found 212/215 (98.6%) identical and 3/215 with a
different classified intent for the same message -- a useful, honest
finding: LLM extraction here is *not* perfectly deterministic even at
temperature 0, though the disagreement rate is small. The production
cache actually used for `output.csv` was not touched or replaced by this
verification.

## Explanation grounding in practice

Of the 250 `decision_explanation` values in `output.csv`, 180 (72%) use
the LLM's rewording and 70 (28%) fall back to the deterministic sentence
built straight from the computed facts -- either because the model
declined to rephrase (it's instructed to echo the grounded summary
verbatim when it can't add value safely) or because its rewording
introduced a number not present in the input and was discarded by the
grounding check in `code/explain.py`. Both outcomes are correct behavior,
not failures: the 28% fallback rate is the grounding safeguard actually
doing its job, not a gap in it.

## Methodology

- Explanation counts include only the 250 real evaluation requests in
  `dataset/requests.csv`, not the 25 solved examples in
  `dataset/sample_requests.csv` used solely for internal validation
  during development.
- "Average tokens / request" divides the grand total (237,709) by 250,
  the number of predicted rows in `output.csv` -- extraction calls are
  shared infrastructure (a message or image can relate to any request
  for that user) rather than 1:1 with a request, so this is a
  system-wide efficiency figure, not a claim that each request
  individually consumed ~951 tokens.
- No LLM calls occur in the deterministic core of the pipeline: event
  resolution, the 90-day balance simulation, plan generation, and the
  6-step ranking are pure Python (see `code/resolve.py`,
  `code/simulate.py`, `code/plans.py`). The deterministic
  `decision_explanation` template (`code/explain.py`'s
  `deterministic_explanation`) also makes no LLM call; the `explanation`
  row above reflects the optional LLM rewording step only, which is
  independently checked for numeric grounding against that deterministic
  sentence before being accepted (see `code/explain.py`).
