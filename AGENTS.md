# AGENTS

## Purpose

You are working in the `volatility-forecast` repository. This repo owns canonical volatility-model code, reusable experiment drivers, evaluation logic, and tests. For the cross-repo workflow, model implementations belong here, not in the blog repo.

This repo supports one long-running volatility forecasting workflow rather than a single PGARCH subproject. The accepted series frontier is currently `volatility-forecasts-6`: constant-`mu`, channel-allocated PGARCH with per-channel screening on the expanded feature tier. The active linear frontier is screened `PGARCH-L (K=5)`, and the active nonlinear frontier is screened `XGBPGARCH [gbtree-loose, K=5]`. Every post after Part 6 should be treated as a code-first search branch for a future frontier model: search and validation happen here in the package first, and the corresponding blog notebook should be written only after a candidate clears the agreed acceptance bar. The current contents of `volatility-forecasts-7` through `volatility-forecasts-10` are stale placeholders or archived drafts unless the user explicitly reopens or rewrites them.

## Scope

This file applies to the repo. Keep compatibility with alphaforge conventions and avoid unnecessary refactors.

## Environment

- OS: macOS
- Python: use conda env `py312`

## Local .env

- Keep environment variables in a `.env` file at the repo root.
- Common keys: `TIINGO_API`, `TIINGO_CACHE_DIR`, `TIINGO_CACHE_MODE`.

## Layout

- `volatility_forecast/model/` contains the core model implementations.
- `volatility_forecast/features/`, `targets/`, `evaluation/`, and `reporting/` hold reusable pipeline pieces.
- `examples/` contains generic runnable demos and public benchmark examples.
- `research/` contains internal frontier-search and branch-evaluation drivers that are not part of the public example surface.
- `tests/` contains regression and unit coverage.
- `docs/design_decisions/training_protocol.md` documents the preferred overfitting-control protocol for neural or high-capacity models.
- `outputs/` is for local experiment artifacts and summaries, not for canonical blog ownership.

## Commands

- Bootstrap canonical Alphaforge runtime: `conda run -n py312 python -m pip install -e ../alphaforge -e .`
- Verify canonical Alphaforge surface: `conda run -n py312 python -c "from alphaforge.data.context import DataContext; assert hasattr(DataContext, 'from_adapters'); assert hasattr(DataContext, 'load')"`
- Run tests: `conda run -n py312 python -m pytest -q`
- If the exact experiment command is unclear, leave `TODO(user): add canonical experiment command` instead of inventing one.

## What to do

- Preserve public interfaces where possible. If you change constructor arguments, fit or predict behavior, output schemas, or shared utility semantics, update tests and affected examples in the same change.
- Keep code modular. Promote reusable logic into package modules rather than burying it in notebooks or one-off scripts.
- Keep `examples/` blog-agnostic. Series- or branch-specific research drivers should live under `research/`, not under the public example namespace.
- Prefer small, composable changes over broad refactors.
- Treat model work as part of the broader volatility-series workflow, not as an isolated family-specific branch.
- Treat Part 6 as the accepted continuation point, and treat every later post slot as a gated search branch for a future frontier model.
- Read `../codex-workflows/volatility-forecasts-experience.yaml` before `../codex-workflows/volatility-forecasts-log.md` so past mistakes, branch guardrails, and reopen conditions are loaded before current branch state.
- Record series rung, baseline family, candidate family, decision, and next branch in `../codex-workflows/volatility-forecasts-log.md` for substantive research iterations.
- Append durable lessons to `../codex-workflows/volatility-forecasts-experience.yaml` when the work reveals:
  - specification mistakes
  - coding mistakes
  - negative-result reroutes
  - branch reopen conditions
  - durable workflow guardrails
- Separate experiment axes when possible:
  - feature changes should hold regularization and structure fixed
  - regularization changes should hold features and structure fixed
  - structural model changes should begin from a frozen feature tier and loss definition
- On the accepted Part 6 frontier, hold these fixed unless the task explicitly studies a deviation:
  - constant `mu`
  - hard channel allocation (`mu` constant, `phi` and `g` from the expanded pool)
  - per-channel screening as part of the model definition
- For any post-6 frontier search work, follow a code-first gate:
  - search for promising directions inside `volatility-forecast`
  - validate the best candidate against the Part 6 comparison frame
  - only after a candidate qualifies should the corresponding blog-side `posts/volatility-forecasts-*/draft.ipynb` be written
- If no post-6 candidate qualifies, record the negative search honestly in the canonical log and do not force a new frontier notebook.
- Keep tests aligned with agreed conventions and update them when behavior intentionally changes.

## Experiment Protocol

- Start from an explicit baseline before introducing a candidate.
- Name the baseline precisely: for example `GARCH(1,1)`, `STES`, `XGBSTES`, `PGARCH-L (3 feat, const mu)`, `PGARCH-L (hard alloc)`, `PGARCH-L (screened K=5)`, `XGB-g-PGARCH`, or `XGBPGARCH [gbtree-loose, screened K=5]`.
- Hold the sample window, target, loss, and split constant while comparing a candidate to its baseline.
- Compare serious candidates against `GARCH(1,1)` and the current family baseline when the series stage requires both.
- For frontier-relevant Part 6 work, the default comparison frame is:
  - `GARCH(1,1)` as the benchmark anchor
  - screened `PGARCH-L (K=5)` as the linear family baseline
  - screened `XGBPGARCH [gbtree-loose, K=5]` as the accepted nonlinear extension
- For any post-6 frontier-search work, a candidate should not be treated as notebook-worthy unless it improves on the relevant Part 6 frontier model under the fixed comparison frame and has a defensible mechanism story.
- Treat the failed `PGARCHMLPModel` gate as evidence against extra static head flexibility as the default next move, not as proof that all later temporal branches are dead.
- Do not reopen `PGARCHHybridModel`, `PGARCHAttentionModel`, or `PGARCHAutoformerModel` merely because they are implemented or more expressive.
- Only reopen Attention-style or Autoformer-style branches if the search produces concrete evidence that missing temporal-path information is the current bottleneck:
  - regime-sliced residual structure
  - gains from richer but still cheap lag/path summaries
  - failed controlled relaxations that still point to path dependence
  - or a cheap additive Hybrid / Attention admission signal on top of frozen screened `PGARCH-L`
- If a temporal branch is reopened, say explicitly:
  - which evidence class triggered it
  - why that evidence justifies reopening the branch
  - whether it only justifies a cheap admission gate or a fuller expansion
  - whether `PGARCHAutoformerModel` is still premature
- Record whether the candidate changed:
  - features
  - regularization
  - structure
- Track three states for each branch:
  - baseline
  - candidate
  - best-so-far

## Overfitting Control

- Do not tune on the test segment.
- Use time-series validation or the documented early-stopping/refit protocol for high-capacity models.
- Report both point metrics and calibration-style diagnostics when they materially affect the conclusion.
- Treat gains that appear only after repeated test-set peeking as invalid.
- Prefer pruning weak directions quickly rather than expanding a high-capacity search with no validation support.
- Do not treat all-feature allocations as the default frontier. On the accepted Part 6 tier, screening is part of the current winning comparison frame rather than an optional cosmetic add-on.

## Conventions

- Fixed-split evaluation aligns with `examples/volatility_forecast_2.py`:
  - Train: `[is_index : os_index)`
  - Test: `[os_index : ]`
- STES prediction is aligned to next-step variance; compare to shifted target when needed.

## Data and PIT

- `asof_utc` should be present by default in data flowing through the pipeline.

## Caching

- Tiingo caching is pluggable via alphaforge cache backends.
- Use CLI flags for cache control when available.

## What not to do

- Do not copy blog prose or blog-specific notebook glue into the core package unless it becomes reusable code.
- Do not ship one-off hacks that only work for a single notebook run.
- Do not silently change train/test alignment, target alignment, or metric definitions.
- Do not treat `outputs/` artifacts as a stable API contract.
- Do not make avoidable interface changes without updating the dependent examples and tests.
- Do not treat the current contents of `volatility-forecasts-7` through `volatility-forecasts-10` as the active frontier. Post-6 slots are gated search branches, not already-accepted results.

## Validation

- Run targeted tests for the modules you changed.
- Run broader regression coverage when shared model interfaces, pipeline helpers, or evaluation utilities changed.
- If you changed an example-backed workflow, run the smallest reproducible command that exercises it.
- Verify that metrics, path outputs, and reported baselines line up with the actual implementation.

## Deliverables

- Code changes in reusable modules or explicit experiment drivers.
- Updated tests for intentional behavior changes or new model behavior.
- A concise record of series rung, baseline, candidate, and best-so-far when the work is part of a search cycle.
- Clear validation notes: run, skipped, or blocked.

## Linear Issue Writing Spec

Linear is the shared work ledger for cross-agent work. Issues should be
specific, searchable, dependency-aware, and tied to an observable outcome.
Treat an issue as a short engineering spec, not as a note or a chat summary.

### When to create or update an issue

- Create a Linear issue when work must survive beyond the current chat, spans
  more than one file or agent, introduces a blocker, or needs durable tracking.
- Update an existing issue instead of creating a duplicate when the scope is
  the same.
- Split a ticket when it contains more than one independent reviewable outcome.
  Use an umbrella parent issue for the broader objective and child issues for
  delivery slices.
- Do not mirror every local task row into Linear. Use Linear for coordination,
  blockers, durable planning, or user-visible work that benefits from tracking.

### Title and naming

- Use outcome-first titles of the form `<Area>: <specific result>`.
- Keep titles short, concrete, and searchable.
- Put the domain noun in the title, not just the implementation verb.
- If the work concerns data contracts, feature/target semantics, evaluation
  semantics, or workflow governance, include `Semantic` in the title.
- If the work is route-local, name the module, workflow, or artifact directly.
- Avoid vague titles such as `Cleanup`, `Refactor`, or `Improve model` unless
  paired with the exact target.
- Good examples:
  - `Alphaforge migration: align py312 runtime with canonical API`
  - `Pipeline: move market loads onto ctx.load`
  - `Semantic target: canonical forward realized variance loading`
  - `Examples: adopt adapter-backed dataset recipes`

### Issue body shape

Use a compact spec structure so any agent can recover the plan quickly:

- Objective: what should exist when the issue is done.
- Why now: why this matters now.
- Scope: the exact modules, workflows, or docs in scope.
- Non-goals: what is explicitly out of scope.
- Dependencies: hard blockers with issue IDs.
- Acceptance criteria: observable conditions that define success.
- Validation: tests, experiment checks, docs, or review gates.
- Follow-on work: separate issues for future slices, if needed.

If the issue is exploratory or research-oriented, add the decision it informs
and the concrete artifact it should produce.

### Dependency rules

- Use parent/child relationships for umbrella work and delivery slices.
- Use `blockedBy` and `blocks` only for hard prerequisites.
- Use `relatedTo` for adjacent work that does not prevent completion.
- Keep dependency chains shallow. Do not create a blocker unless the work
  cannot be completed correctly without it.
- If the blocker does not yet exist, create the blocker issue first or state
  the missing prerequisite explicitly.
- Do not block on anticipated future reuse alone; keep that as a scoped note
  unless the reuse is already real.

### Priority rules

- Priority 1 / Urgent: active outage, broken build, release blocker, or a hard
  blocker for an approved near-term task.
- Priority 2 / High: user-visible work, foundational platform work, or an item
  that unlocks multiple other issues.
- Priority 3 / Normal: planned implementation slices and most feature work.
- Priority 4 / Low: docs, cleanup, exploratory refactors, and deferred follow-ups.
- Priority 0 / None: parking-lot ideas that are not ready to schedule.
- Default to Priority 3 unless there is a concrete reason to raise it.
- Do not use Urgent for roadmap ambition alone.

### Blocker handling

- If a task is blocked, say so explicitly in the issue and when communicating
  with the user.
- State the blocking issue ID(s), the missing prerequisite, and the next unblock
  step.
- Never present a blocked issue as complete.
- If the user asks to complete work but a Linear blocker remains, surface the
  blocker before claiming success.
- Convert vague blockers into concrete missing primitives, data contracts,
  validation gates, or environment prerequisites whenever possible.

### Done criteria

- Mark an issue Done only when the implementation slice is landed, the agreed
  tests or validation pass, and the acceptance criteria are satisfied.
- For design or docs work, Done means the artifact exists, is reviewed, and the
  user-visible contract has been updated.
- If an issue changes behavior, APIs, workflows, validation, or research
  process that users or developers rely on, update the relevant official docs
  as part of the closeout:
  - `docs/design_decisions/` for modeling, target, and protocol behavior
  - `docs/models/` for model-facing implementation and usage behavior
  - `docs/index.md` and `README.md` for repo entrypoint or workflow changes
  - `../codex-workflows/volatility-forecasts-log.md` and
    `../codex-workflows/volatility-forecasts-experience.yaml` when the ticket
    changes frontier-search state or durable workflow guidance
- If the issue is intentionally doc-free, record that explicitly in the issue
  closeout note.
- Close the issue with a short note summarizing the result, the validation, and
  any follow-on issue IDs.
- If useful work remains, split it into follow-on issues instead of leaving the
  original issue ambiguous.
- Do not leave an issue in Done if it still has an unresolved hard blocker.

### Engineering plan quality

- Every implementation issue should include a concrete plan with small phases.
  Each phase should produce a checked-in artifact or a validation gate.
- Prefer stable scaffolds plus surgical deltas over whole-module rewrites.
- If the plan cannot be explained as a few reviewable phases, the issue is too
  large and should be split.

## Ticket Implementation Workflow

All coding agents working from Linear must follow this workflow for every
implementation ticket unless the user explicitly overrides it.

### Required execution order

1. Review upstream context before coding.
2. Announce the current ticket number and its plain-English goal on screen.
3. Implement the current ticket with test-driven development.
4. Update the relevant docs after the implementation and tests pass.
5. Leave a handoff note in Linear describing what changed and any caveats.
6. Mark the ticket `Done`, then update any repo-local plan or workflow mirror.

### Step 1: Review upstream context

Before writing code, the implementer must:

- read the current ticket body in full
- read all hard-blocking upstream tickets and their completion notes
- read recent comments on the parent issue when the parent is an active umbrella
- inspect the referenced docs and current code paths in scope
- identify the exact files, tests, and docs that are likely to change

If an upstream ticket is not done, do not start implementation unless the user
explicitly approves working around the blocker.

### Step 2: Announce the current ticket on screen

Before coding, print the ticket number and a short plain-English explanation
of what the ticket aims to do in the current terminal/chat session.

Minimum expectation:

- include the Linear ticket id, for example `VOL-12`
- explain the ticket goal in one or two plain-English sentences
- do this after reading upstream tickets and before writing code

### Step 3: Implement with TDD

For code-changing tickets:

- start by adding or updating the tests that define the target behavior
- run the tests and confirm they fail for the expected reason
- implement the smallest coherent code change that makes the tests pass
- rerun the targeted tests, then rerun the broader validation required by the
  ticket or module-owner rules

Minimum expectation:

- targeted tests for the changed behavior
- broader regression coverage for the touched subsystem when practical

For documentation-only or planning-only tickets, state explicitly in the ticket
that TDD does not apply.

### Step 4: Update docs after tests pass

When behavior, APIs, runtime flow, governance, validation, or research workflow
changes, update the relevant official docs after the implementation is stable:

- `docs/design_decisions/` for modeling, target, and protocol behavior
- `docs/models/` for model behavior and implementation-facing guidance
- `docs/index.md` and `README.md` for entrypoint and workflow changes
- `research/frontier_search/README.md` when the search-driver workflow changes
- `../codex-workflows/volatility-forecasts-log.md` and
  `../codex-workflows/volatility-forecasts-experience.yaml` for substantive
  frontier-search and durable-workflow updates

Doc updates are part of completing the ticket, not optional follow-up work,
unless the ticket is explicitly scoped as doc-free.

### Step 5: Leave a Linear handoff note

Before marking the ticket done, add a Linear comment with this information:

- what was implemented
- which tests were added or updated
- which test commands were run and whether they passed
- which docs were updated
- any caveats, deferred work, follow-on risks, or compatibility notes that the
  next implementer should know

If the ticket is blocked or only partially complete, leave the same note but do
not mark it done.

### Step 6: Mark done and update the mirrored plan or workflow record

Linear is the source of truth for ticket state. Repo-local plan docs and
workflow logs are mirrors for later coding agents.

Mark the ticket `Done` only when:

- the code is landed
- the agreed validation passed
- the relevant docs are updated or the ticket explicitly records why no docs
  changed
- the Linear handoff note is written

After the ticket is moved to `Done` in Linear:

- update the corresponding row in any repo-local plan doc if one exists
- update `../codex-workflows/volatility-forecasts-log.md` or
  `../codex-workflows/volatility-forecasts-experience.yaml` when the ticket
  changes canonical workflow state or research branch status
- keep any plan table sorted in implementation order
- do not mark a mirror row `Done` before the Linear ticket is actually closed

### Recommended Linear closeout template

Use this structure for the final implementation note:

- Implemented:
- Tests:
- Docs:
- Caveats:
- Follow-ons:
