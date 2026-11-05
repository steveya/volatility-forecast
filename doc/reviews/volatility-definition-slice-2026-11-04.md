# VOL-7: shared volatility definitions and bounded reuse

Review date: 2026-11-04, America/New_York.

## Result and supported use

The existing owner collection at revision 11 contained 2,192 statements and
5,302 current native records. The earlier 1,644,391 figure counted candidate
pairwise alignments among 1,814 generic-placeholder fragments, not accepted
knowledge records. This slice reuses eight exact existing claim revisions and
24 source-mapping links rather than reingesting the sources.

`research/definition_slice.py` defines twelve operational objects: conditional
variance, conditional second moment, conditional standard deviation, squared
return, squared innovation, realized variance, realized standard deviation,
Corsi's weekly average volatility, MSE, RMSE, and two QLIKE conventions. The
source map is `research/definition_slice_sources.json`; its locators retain
source-document byte hashes and existing claim revisions.

The native delta adds twelve constructs, twelve curator definition statements,
twelve anchors, five symbols, one normalization-source document, one symbolic
quantity bundle (three scientific targets and twelve quantity definitions), one
assembly, and twelve open scientific-use obligations. Original statements and
their source contexts are retained unchanged. Source mappings remain candidate
source-role dependencies, not proof or empirical-support edges.

## Meaning and checks

Each native `QuantityDefinition.formula` contains canonical JSON of its role,
scale, expression and retained conditions. Those fields all affect its native
revision. The domain checker interprets this explicit declaration language;
general Synapse does not infer arbitrary prose equivalence. The checker policy
also records its version and exact module digest.

`ForecastScope` names exact target/measurement/loss revisions and declares horizon,
units, return convention, information, origin rule, population, sample, ordered
rows and weights. Compatibility checks these declarations; matching strings do
not authenticate data or timestamps. `compare` computes weighted losses on
supplied, aligned, finite nonnegative observations and forecasts. It runs no
market model and issues no scientific execution or inference receipt.

Raw squared return has a conditional second-moment target; its identification
with conditional variance needs the conditional mean restriction. Realized
variance is distinct from latent quadratic variation and conditional return
variance. Corsi's weekly quantity averages daily roots. Aggregate RMSE retains
the same target as MSE; it is not MSE of standard deviations. `log(h)+y/h`
permits zero observations with positive forecasts; ratio-normalized QLIKE needs
positive observations too. Patton's expected-ranking result retains its source
assumptions and requires a separate target-world mapping.

Unknown conditional moments are native unobserved-truth dependencies. Future
measurements are observable at `t+h` and derive lookahead-dependent relative to
origin `t`. Loss arithmetic is explicitly evaluated ex post. No implementable
forecast estimator is introduced by these target definitions.

`latent_expected_ranking` and other stronger uses remain unresolved. Source
validity, proxy validity, equal expected loss, significance, generalization,
outlier resistance and independent replication are not inferred from a passing
descriptive score comparison. No scalar fit, study, benchmark, proof or
statistical inference is invented.

## Review findings and disposition

Independent review of sources, implementation and tests found three defects in
the initial candidate:

1. Native definition references omitted scale/role/conditions. Condition-only and
   scale-only regressions failed; full canonical declarations now affect revisions.
2. Constructs omitted symbol dependencies, falsely deriving implementability for
   latent targets. A regression reproduced it; explicit roles/clocks now derive
   theoretical-oracle and lookahead statuses appropriately.
3. Equal malformed metadata passed compatibility. Seven regressions reproduced
   it; non-string context fields are now incompatible and missing context stays
   unresolved.

Review also requested checker identity and stronger consumer readback. Both were
added. A real SQLite roundtrip checks exact native references, source revision
visibility, feasibility, and detection of a later source correction. The
independent final review reran all 47 targeted cases and found no remaining
blocker for passive publication and finite-sample descriptive arithmetic.

## Validation and owner publication

The 47 domain cases and 29 existing ingestion cases passed together (76 total)
using Python 3.12 in `py312` and explicit Synapse source runtime at commit
`6aa7f43`. These exercise software contracts and arithmetic examples, not
financial results or universal scientific soundness. A separate clean-source
check and owner consumer results are recorded below when complete.

The owner remains the configured volatility collection, store
`8e240fd4-5148-4406-8a52-d7d638ffb6d6`. A complete pre-admission SQLite backup and
project manifest are retained under `.synapse/backups/definition-slice-2026-11-04/`;
its `quick_check` passed. External source assets remain at their existing paths.
Native preview/workflow reports and consumer artifacts are retained separately
under `.synapse/artifacts/definition-slice-2026-11-04/`. Inspect the workflow
receipt before retrying a publication; optional consumer delivery can be recovered
without republishing the delta.

Fresh owner publication committed revision **12** with automated consumer review
complete. All **5,302 prior native records** and **2,192 prior claim revisions**
were preserved. Four fresh query probes returned their normalized definition and
open use obligation; persisted source inspection confirmed every selected source
revision current, original anchors retained, unproved/untested statuses, and
theoretical-oracle/lookahead labels where applicable. The portable
[consumer summary](evidence/volatility-definition-slice-2026-11-04/consumer-review.json)
records the exact owner identity and checks. No scientific or formal execution ran.

A clean Git source archive with only this slice's new implementation/tests ran
**307 cases: 305 passed, one skipped, one failed**. The failure is the existing
`test_volgru_torch_reduces_to_stes_exactly`: float32 Torch gates are compared to
float64 STES gates at absolute tolerance `1e-10`. Running that original test alone,
without importing the new definition module/tests, reproduced the failure. No
VolGRU code or test tolerance was changed in this slice. Torch is optional in the
repository dependency configuration. Hosted CI remains a separate landing gate;
the local broad run is not reported as green.

## Reproduce

Pure checks require only Python and pytest:

```sh
python -m pytest tests/test_definition_slice.py
```

The native integration case requires the explicitly selected Synapse runtime.
An audit is read-only with respect to the owner and writes a new output directory:

```sh
PYTHONPATH=/absolute/synapse/src python research/definition_slice.py \
  --project /absolute/volatility-forecast --output /absolute/new-audit-directory
```

`--record` deliberately publishes through `make_change` → `preview_change` →
`record_change`, then fresh-binds and reviews the consumer. It does not initialize
or replace a collection. Existing output directories are refused.

## Remaining work

SPS-28 retains general shared-definition composition; SPS-74 owns a trusted
claim/check empirical-use mapping; SPS-75/104 own prespecified real-workflow and
human evaluation. The broader reconstruction plan, model code and existing
accepted frontier remain unchanged. This slice does not close those gates.
