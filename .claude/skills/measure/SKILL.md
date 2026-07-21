---
name: measure
description: How to produce a performance or cost number for FleetLM that is safe to publish. Use before adding any throughput, latency, cost, or efficiency claim to the README, a benchmark file, or a post.
---

# Measuring FleetLM

The project's credibility rests on never overstating what it does. An earlier
README claimed "~80% native performance" and "<200 ms/token" with no
measurement behind either; both were deleted. Do not recreate that situation.

## Decide what you are producing

**Smoke test** - one run, no controls. Fine for "does this path work at all"
and for a directional number. Must be labeled as a smoke test and must state
what it does not show.

**Benchmark** - a number anyone may quote. Requires matched inputs, fixed
seeds, at least three replicates, and a reported spread. Only claim
differences that clear the noise.

Never let the first quietly become the second.

## Rules

1. **Measure the production path.** Real orchestrator, real node agent, real
   engine, real API calls. Not a synthetic loop around an internal function.
2. **Matched pairs for comparisons.** Both arms see identical inputs and
   identical sampling seeds, so the difference isolates the change rather
   than question difficulty or sampling luck.
3. **Separate telemetry from benchmarks.** Numbers read off `/metrics` during
   a live run are telemetry; say so.
4. **Attribute time honestly.** `generation_sec` is engine time reported by
   the node. Wall clock includes queueing, network, and your own polling
   interval - if your poll loop sleeps 250 ms, say that, because it is inside
   the number.
5. **State the hardware, model, quantization, and batch shape.** A tok/s
   figure without them means nothing.
6. **Write down what the run does not show.** Single-host runs say nothing
   about multi-machine behavior; a churn test says nothing about throughput.

## Where numbers live

Write results to `benchmarks/YYYY-MM-DD-<topic>.md`, including setup,
procedure, results, and an explicit "what this does not show" section. That
directory is currently untracked - treat it as the working record and quote
from it rather than inventing numbers elsewhere.

Only promote a number into the README once it is a real benchmark, and link
or cite where it came from.

## Cost per token

The number the project's argument ultimately rests on, and the most
error-prone. To be defensible it needs:

- Measured energy, not TDP guesses - wall meter, or `powermetrics` sampled
  across the run on Apple silicon.
- The machine's *idle* draw subtracted, since the premise is that the laptop
  was already on.
- A local electricity price, stated.
- The comparison stated on equal terms: cloud batch pricing for the same
  model class and the same token counts, dated, since prices move.

If any of those are estimated rather than measured, label the result an
estimate in the same sentence as the number.
