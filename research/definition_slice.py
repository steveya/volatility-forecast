"""VOL-7: declared target/proxy/loss compatibility and bounded score arithmetic.

Pure scoring needs only Python. Native graph publication is an explicit opt-in
using the host-selected Synapse runtime. This module never runs a market model,
proves a proxy mapping, authenticates inputs, or establishes expected rankings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType

PREFIX = 'volatility:definitions:v1:'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _digest(value):
    return sha256(_json(value).encode()).hexdigest()


def _definition(role, scale, formula, conditions):
    return MappingProxyType(dict(role=role, scale=scale, formula=formula, conditions=conditions))


DEFINITIONS = MappingProxyType({
    'conditional_variance': _definition('target', 'squared', 'v_t = E[(r_(t+h)-E[r_(t+h)|F_t])^2|F_t]',
        'Finite conditional second moments; explicit return, horizon and origin information.'),
    'conditional_second_moment': _definition('target', 'squared', 'm2_t = E[r_(t+h)^2|F_t]',
        'Finite conditional second moments. m2_t=v_t+mu_t^2; equality requires mu_t=0.'),
    'conditional_stddev': _definition('target', 'linear', 'sigma_t = sqrt(v_t)',
        'Nonnegative conditional variance. E[sqrt(V)|F_t] is not generally sqrt(E[V|F_t]).'),
    'squared_return': _definition('measurement', 'squared', 'y_(t+h) = r_(t+h)^2',
        'Observed proxy. Its conditional mean is the second moment, not automatically variance.'),
    'squared_innovation': _definition('measurement', 'squared', 'y_(t+h) = (r_(t+h)-mu_t)^2',
        'The mean definition and estimation error matter; a fitted residual is not automatically the true innovation.'),
    'realized_variance': _definition('measurement', 'squared', 'RV_(t,h,grid) = sum_j (Delta log(P)_j)^2',
        'Finite grid, price convention, session, overnight, noise, missingness and roll policies remain required. Not exact QV or conditional variance.'),
    'realized_stddev': _definition('measurement', 'linear', 'RVol_(t,h,grid) = sqrt(sum_j (Delta log(P)_j)^2)',
        'Same measurement policies as realized variance; return units, not squared-return units.'),
    'corsi_weekly_volatility': _definition('measurement', 'linear', 'RVol_week = (1/5) sum_(d=1..5) sqrt(RV_d)',
        'Source Corsi convention: average daily roots, not sqrt of weekly summed variance.'),
    'mse': _definition('loss', 'target_squared', 'MSE = sum_i w_i (y_i-h_i)^2 / sum_i w_i',
        'Same named target/proxy, rows, units and weights; expected-risk interpretation requires integrability.'),
    'rmse': _definition('loss', 'target', 'RMSE = sqrt(sum_i w_i (y_i-h_i)^2 / sum_i w_i)',
        'Monotone aggregate transformation of matched MSE; not MSE on square-root-transformed targets.'),
    'qlike': _definition('loss', 'variance_convention', 'QLIKE = sum_i w_i (log(h_i)+y_i/h_i) / sum_i w_i',
        'h_i>0, y_i>=0. Absolute levels depend on units. Proxy expected-ranking robustness needs the source assumptions and a checked target-world mapping.'),
    'qlike_ratio': _definition('loss', 'variance_convention', 'QLIKE_ratio = sum_i w_i (y_i/h_i-log(y_i/h_i)-1) / sum_i w_i',
        'h_i>0 and y_i>0; differs from log(h)+y/h by an observation-only term on this positive domain.'),
})
CHECKER_VERSION = 'observed-proxy-loss-v1'
IMPLEMENTATION_DIGEST = sha256(Path(__file__).read_bytes()).hexdigest()
POLICY_DIGEST = _digest({'definitions':{key:dict(value) for key,value in DEFINITIONS.items()},
    'checker_version':CHECKER_VERSION,'implementation_sha256':IMPLEMENTATION_DIGEST})


@dataclass(frozen=True)
class DefinitionRef:
    kind: str
    logical_id: str
    revision_digest: str


def _quantity_payload(key):
    # The native formula string is a canonical domain declaration: conditions,
    # role and scale are part of its exact revision, not unversioned comments.
    return dict(definition_id=PREFIX+'quantity:'+key, formula=_json(dict(DEFINITIONS[key])),
                scale_ref=None, normalization_procedure_ref=None)


def definition_ref(key):
    payload = _quantity_payload(key)
    return DefinitionRef('quantity_definition', payload['definition_id'], _digest(payload))


@dataclass(frozen=True)
class ForecastScope:
    target: DefinitionRef
    measurement: DefinitionRef
    loss: DefinitionRef
    horizon: str | None
    unit: str | None
    return_convention: str | None
    information_set: str | None
    origin_rule: str | None
    population: str | None
    sample: str | None
    rows: tuple[str, ...] | None
    weights: tuple[float, ...] | None


@dataclass(frozen=True)
class ComparisonReport:
    status: str
    reasons: tuple[str, ...]
    retained_conditions: tuple[str, ...] = ()
    scores: tuple[float, float] | None = None
    input_digest: str | None = None
    policy_digest: str = POLICY_DIGEST
    checker_version: str = CHECKER_VERSION
    implementation_digest: str = IMPLEMENTATION_DIGEST
    empirical_support: bool = False
    independent_replication: bool = False


def _key(ref, role):
    if type(ref) is not DefinitionRef:
        return None
    return next((key for key in DEFINITIONS if DEFINITIONS[key]['role'] == role and definition_ref(key) == ref), None)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def assess(left, right, *, requested_use='observed_proxy_loss'):
    """Compare declarations only; equal labels cannot establish source validity."""
    if type(left) is not ForecastScope or type(right) is not ForecastScope:
        return ComparisonReport('unresolved', ('Malformed comparison scope',))
    reasons, missing = [], []
    for field, role in [('target', 'target'), ('measurement', 'measurement'), ('loss', 'loss')]:
        a, b = getattr(left, field), getattr(right, field)
        if _key(a, role) is None or _key(b, role) is None:
            missing.append(field+' definition or current revision is unknown')
        if a != b:
            reasons.append(field+' differs; an explicit reviewed mapping is required')
    for field in ('horizon', 'unit', 'return_convention', 'information_set', 'origin_rule', 'population', 'sample', 'rows', 'weights'):
        a, b = getattr(left, field), getattr(right, field)
        if field not in {'rows', 'weights'} and any(x is not None and type(x) is not str for x in (a,b)):
            reasons.append(field+' must be a nonempty string')
        elif a is None or b is None or (isinstance(a, str) and not a.strip()) or (isinstance(b, str) and not b.strip()):
            missing.append(field+' is missing')
        elif a != b:
            reasons.append(field+' differs')
    for side in (left, right):
        if side.rows is not None and (type(side.rows) is not tuple or not side.rows or any(type(x) is not str or not x.strip() for x in side.rows) or len(set(side.rows)) != len(side.rows)):
            reasons.append('rows must be nonempty ordered unique identities')
        if side.weights is not None and (type(side.weights) is not tuple or not side.weights or any(not _finite(w) or w < 0 for w in side.weights) or not math.isfinite(sum(side.weights)) or sum(side.weights) <= 0):
            reasons.append('weights must be finite nonnegative and have positive sum')
        if type(side.rows) is tuple and type(side.weights) is tuple and len(side.rows) != len(side.weights):
            reasons.append('rows and weights have different lengths')
        target, measurement, loss = _key(side.target, 'target'), _key(side.measurement, 'measurement'), _key(side.loss, 'loss')
        if target and measurement and DEFINITIONS[target]['scale'] != DEFINITIONS[measurement]['scale']:
            reasons.append('target and measurement scales differ')
        if target and loss in {'qlike', 'qlike_ratio'} and DEFINITIONS[target]['scale'] != 'squared':
            reasons.append('QLIKE variance convention requires squared-return scale')
    if requested_use != 'observed_proxy_loss':
        missing.append('Requested scientific use requires a separately checked claim/proxy/inference mapping; source assumptions are not target-world facts')
    conditions = tuple(DEFINITIONS[k]['conditions'] for k in DEFINITIONS if k in {_key(left.target, 'target'), _key(left.measurement, 'measurement'), _key(left.loss, 'loss')})
    conditions += ('Matched declared metadata does not authenticate timestamps/data or establish proxy validity, expected rankings, significance, generalization or independent replication.',)
    status = 'incompatible' if reasons else ('unresolved' if missing else 'compatible')
    return ComparisonReport(status, tuple(dict.fromkeys(reasons+missing)), conditions)


def compare(left, right, observations, left_forecasts, right_forecasts, *, requested_use='observed_proxy_loss'):
    """Compute only the supplied finite-sample proxy losses after compatibility."""
    report = assess(left, right, requested_use=requested_use)
    if report.status != 'compatible':
        return report
    try:
        arrays = tuple(tuple(values) for values in (observations, left_forecasts, right_forecasts))
        if any(len(values) != len(left.rows) or any(not _finite(v) or v < 0 for v in values) for values in arrays):
            raise ValueError('Observations/forecasts must be finite nonnegative aligned arrays')
        y, a, b = arrays
        loss = _key(left.loss, 'loss')
        if loss in {'qlike', 'qlike_ratio'} and any(v <= 0 for v in (*a, *b)):
            raise ValueError('QLIKE forecasts must be strictly positive')
        if loss == 'qlike_ratio' and any(v <= 0 for v in y):
            raise ValueError('Ratio QLIKE observations must be strictly positive')
        def score(h):
            if loss in {'mse', 'rmse'}:
                terms = [(yi-hi)**2 for yi, hi in zip(y, h)]
            elif loss == 'qlike':
                terms = [math.log(hi)+yi/hi for yi, hi in zip(y, h)]
            else:
                terms = [yi/hi-math.log(yi/hi)-1 for yi, hi in zip(y, h)]
            value = math.fsum(w*v for w, v in zip(left.weights, terms))/math.fsum(left.weights)
            value = math.sqrt(value) if loss == 'rmse' else value
            if not math.isfinite(value):
                raise ValueError('Loss is outside finite arithmetic domain')
            return value
        scores = (score(a), score(b))
        return replace(report, scores=scores, input_digest=_digest({'left':asdict(left), 'right':asdict(right), 'arrays':arrays}))
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        return replace(report, status='incompatible', reasons=report.reasons+(str(exc),))


def _symbol_keys_for_definition(key: str, entry: MappingProxyType[str, str]) -> list[str]:
    special = {
        'conditional_variance': ['latent_variance'],
        'conditional_second_moment': ['latent_second_moment'],
        'conditional_stddev': ['latent_variance'],
        'squared_innovation': ['future_returns', 'latent_mean'],
    }
    if key in special:
        return special[key]
    return ['future_returns', 'forecast'] if entry['role'] == 'loss' else ['future_returns']


def _normalize_claim_ref(store: object, ref: str) -> str:
    if not isinstance(ref, str) or not ref.startswith('synapse://'):
        return ref
    from synapse.digest import parse_synapse_uri

    parsed = parse_synapse_uri(ref)
    if parsed.kind == 'claim' and parsed.namespace in store.local_namespaces():
        return parsed.record_id
    return ref


def compute_construct_feasibility(store: object, construct_id: str) -> dict[str, object]:
    """Classify a definition construct as oracle, lookahead_dependent, or implementable."""
    construct = store.get_object(construct_id)
    if construct is None:
        raise KeyError(f'Unknown construct {construct_id!r}')
    spec = dict(construct.spec_payload)
    symbol_ids = spec.get('uses_symbols')
    if not isinstance(symbol_ids, list) or not symbol_ids:
        raise KeyError(f'Missing or empty bound symbols for {construct_id!r}')
    unobserved: list[str] = []
    lookahead: list[str] = []
    for sym_id in symbol_ids:
        if not isinstance(sym_id, str) or not sym_id:
            raise KeyError(f'Invalid bound symbol id {sym_id!r} for {construct_id!r}')
        sym = store.get_object(sym_id)
        if sym is None:
            raise KeyError(f'Missing bound symbol {sym_id!r} for {construct_id!r}')
        sym_spec = sym.spec_payload
        if sym_spec.get('role') == 'unobserved_truth':
            unobserved.append(sym_id)
        if sym_spec.get('measurable_at') == 't+h':
            lookahead.append(sym_id)
    if unobserved:
        status = 'theoretical_oracle'
    elif spec.get('decision_time') == 't+h' or lookahead:
        status = 'lookahead_dependent'
    else:
        status = 'implementable_statistic'
    return {
        'construct_id': construct_id,
        'status': status,
        'unobserved_symbols': unobserved,
        'lookahead_symbols': lookahead,
    }


def build_delta(store: object, packet: dict) -> dict:
    """Publish reviewed operational volatility definitions into a Synapse v3.0 store."""
    from synapse.digest import sha256_hex
    from synapse.models import (
        AssemblySection,
        Claim,
        DigestConflictError,
        Edge,
        ObjectRecord,
        SourceDocument,
        SourcePassage,
        TopicAssembly,
    )

    if not isinstance(packet, dict) or packet.get('format_version') != 1:
        raise ValueError('Unsupported or missing definition packet format_version')
    sources_map = packet.get('sources')
    def_sources = packet.get('definition_sources')
    if (
        not isinstance(sources_map, dict)
        or not isinstance(def_sources, dict)
        or set(def_sources) != set(DEFINITIONS)
    ):
        raise ValueError('Unsupported or incomplete definition mapping')

    for key, sources in def_sources.items():
        if (
            not isinstance(sources, list)
            or not sources
            or any(not isinstance(sid, str) or not sid for sid in sources)
            or len(sources) != len(set(sources))
            or any(sid not in sources_map for sid in sources)
        ):
            raise ValueError('Missing or invalid definition source mapping: ' + key)

    source_digests: dict[str, str] = {}
    for sid, expected in sources_map.items():
        if not isinstance(expected, str) or not expected:
            raise ValueError('Invalid expected source revision digest for: ' + sid)
        src_claim = store.get_claim(sid)
        if src_claim is None or src_claim.meaning_digest != expected:
            raise ValueError('Changed or missing source revision: ' + sid)
        source_digests[sid] = src_claim.meaning_digest

    for key in DEFINITIONS:
        sid = PREFIX + 'statement:' + key
        if store.list_warrants_for_claim(sid):
            raise ValueError('Cannot overwrite warranted definition claim: ' + sid)

    spec_digest = _digest({k: dict(v) for k, v in DEFINITIONS.items()})
    module_path = Path(__file__).resolve()
    doc_id = PREFIX + 'normalization-source'
    symbol_specs = (
        ('latent_variance', 'unobserved_truth', 't', 'Unknown conditional variance given F_t'),
        ('latent_second_moment', 'unobserved_truth', 't', 'Unknown conditional second moment given F_t'),
        ('latent_mean', 'unobserved_truth', 't', 'Unknown true conditional mean given F_t'),
        ('future_returns', 'observable', 't+h', 'Returns over the future outcome interval; not known at forecast origin t'),
        ('forecast', 'observable', 't', 'Declared origin-measurable forecast; actual timestamp authentication remains unresolved'),
    )

    claim_digests: dict[str, str] = {}
    with store._tx():
        existing_doc = store.get_source(doc_id)
        v1_migrated_doc = (
            existing_doc is not None
            and existing_doc.content_sha256 != spec_digest
            and existing_doc.metadata.get('v1_scope') == 'global_canonical'
        )
        if not v1_migrated_doc:
            store.put_source(
                SourceDocument(
                    doc_id=doc_id,
                    uri_or_path=str(module_path),
                    content_sha256=spec_digest,
                    title='VOL-7 reviewed local operational definitions',
                    media_type='text/plain',
                    metadata={
                        'policy_digest': POLICY_DIGEST,
                        'spec_digest': spec_digest,
                        'topics': ['shared volatility definitions', 'target proxy loss'],
                    },
                )
            )

        for sym_key, role, clock, description in symbol_specs:
            sym_id = PREFIX + 'symbol:' + sym_key
            sym_obj = ObjectRecord(
                object_id=sym_id,
                kind='symbol',
                canonical_name=sym_key,
                symbol_latex=sym_key,
                domain_type='volatility_quantity',
                spec_payload={'role': role, 'measurable_at': clock},
                title=sym_key,
                exposition=description,
            )
            existing_sym = store.get_object(sym_id)
            if (
                existing_sym is not None
                and existing_sym.kind == 'symbol'
                and existing_sym.spec_payload.get('role') == role
                and existing_sym.spec_payload.get('measurable_at') == clock
                and existing_sym.spec_payload.get('v1_node_type') == 'symbol'
            ):
                continue
            store.put_object(sym_obj)

        for key, entry in DEFINITIONS.items():
            cid = PREFIX + 'construct:' + key
            sid = PREFIX + 'statement:' + key
            pid = sid + ':anchor'
            sym_keys = _symbol_keys_for_definition(key, entry)
            sym_ids = [PREFIX + 'symbol:' + s for s in sym_keys]
            decision_time = 't+h' if entry['role'] == 'loss' else 't'
            q_payload = _quantity_payload(key)
            d_ref = asdict(definition_ref(key))
            verbatim = f"{entry['formula']} — {entry['conditions']}"

            existing_passage = store.get_passage(pid)
            if not (
                v1_migrated_doc
                and existing_passage is not None
                and existing_passage.doc_id == doc_id
                and existing_passage.verbatim_text == f'DEFINITIONS[{key}]'
            ):
                store.ingest_passages(
                    [
                        SourcePassage(
                            passage_id=pid,
                            doc_id=doc_id,
                            locator='DEFINITIONS[' + key + ']',
                            verbatim_text=verbatim,
                            content_sha256=sha256_hex(verbatim),
                            kind='equation',
                            tags=('shared-volatility-definitions', key),
                        )
                    ]
                )

            con_obj = ObjectRecord(
                object_id=cid,
                kind='construct',
                canonical_name=key,
                definition_latex=entry['formula'],
                domain_type=entry['role'],
                spec_payload={
                    'role': entry['role'],
                    'scale': entry['scale'],
                    'decision_time': decision_time,
                    'uses_symbols': sym_ids,
                    'quantity_definition': q_payload,
                    'definition_ref': d_ref,
                },
                title=key.replace('_', ' '),
                exposition=(
                    f"Local versioned operational definition; role={entry['role']}; scale={entry['scale']}; "
                    f"{entry['conditions']} No universal equivalence, implementable estimator, scientific support or formal proof is granted."
                ),
            )
            existing_con = store.get_object(cid)
            v1_migrated_con = (
                existing_con is not None
                and existing_con.kind == 'construct'
                and existing_con.spec_payload.get('v1_node_type') == 'construct'
                and existing_con.definition_latex == entry['formula']
                and existing_con.spec_payload.get('uses_symbols') == sym_ids
                and existing_con.spec_payload.get('decision_time') == decision_time
            )
            if not v1_migrated_con:
                store.put_object(con_obj)

            claim = Claim(
                claim_id=sid,
                claim_kind='definition',
                bindings=(cid, *sym_ids),
                mathematical_latex=entry['formula'],
                title='Shared definition: ' + key.replace('_', ' '),
                exposition=(
                    f"CURATOR OPERATIONAL DEFINITION: {entry['formula']}. {entry['conditions']} "
                    'Source correspondence is scoped normalization, not a new source theorem or empirical acceptance.'
                ),
                tags=('shared-volatility-definitions', 'target-proxy-loss', key),
            )
            existing_claim = store.get_claim(sid)
            if existing_claim is None or existing_claim.meaning_digest == claim.meaning_digest:
                store.promote_passage_to_claim([pid], claim)
                claim_digests[key] = claim.meaning_digest
            elif (
                v1_migrated_con
                and existing_claim.mathematical_latex == entry['formula']
                and cid in existing_claim.bindings
            ):
                if not any(
                    e.target_ref == pid
                    for e in store.list_edges(source_ref=sid, edge_type='sourced_from')
                ):
                    store.put_edge(
                        Edge(
                            edge_id='src:' + sha256_hex(sid + '\x00' + pid)[:24],
                            source_ref=sid,
                            edge_type='sourced_from',
                            target_ref=pid,
                            assertion_mode='conjectured',
                        )
                    )
                claim_digests[key] = existing_claim.meaning_digest
            else:
                raise DigestConflictError(f'Claim {sid!r} already exists with incompatible meaning_digest')

            gap_id = PREFIX + 'use-gap:' + key
            gap_obj = ObjectRecord(
                object_id=gap_id,
                kind='typed_spec',
                canonical_name=gap_id,
                domain_type='use_obligation',
                spec_payload={
                    'normalization_key': key,
                    'target_claim_id': sid,
                    'target_meaning_digest': claim_digests[key],
                    'gap_kind': 'undercut_application_or_transport',
                    'dependencies': [
                        {
                            'dependency_kind': 'source',
                            'route_id': PREFIX + 'source-route:' + key,
                            'source_ref': {'kind': 'statement', 'node_id': src},
                            'target_ref': {'kind': 'statement', 'node_id': sid},
                            'expected_revision_digest': sources_map[src],
                        }
                        for src in def_sources[key]
                    ],
                    'route_id': PREFIX + 'source-route:' + key,
                    'policy_digest': POLICY_DIGEST,
                    'permitted_evidence': [
                        'reviewed_target_world_mapping',
                        'independently_checked_claim_to_procedure_contract',
                        'appropriate_empirical_evaluation',
                    ],
                    'scientific_support': False,
                    'formal_checking': 'not_run',
                },
                title=gap_id,
                exposition=(
                    'Operational definition and source mapping are retained. Before scientific use, instantiate the '
                    'target-world horizon, information, units, measurement and relevant assumptions. ' + entry['conditions']
                ),
            )
            existing_gap = store.get_object(gap_id)
            if existing_gap is not None and existing_gap.meaning_digest != gap_obj.meaning_digest:
                store.put_object(gap_obj, allow_revision=True, reason='Sync Synapse v3 source meaning_digests')
            else:
                store.put_object(gap_obj)

            edge_ids_for_key: list[str] = []
            for src_id in def_sources[key]:
                edge_id = f'{PREFIX}source-link:{key}:{sha256_hex(src_id)[:16]}'
                edge_ids_for_key.append(edge_id)
                edge = Edge(
                    edge_id=edge_id,
                    source_ref=src_id,
                    edge_type='entails',
                    target_ref=sid,
                    assertion_mode='conjectured',
                    metadata={
                        'dependency_kind': 'source',
                        'expected_revision_digest': sources_map[src_id],
                        'route_id': PREFIX + 'source-route:' + key,
                    },
                )
                existing_edge = store.get_edge(edge_id)
                if existing_edge is not None and (
                    existing_edge.source_ref != edge.source_ref
                    or existing_edge.edge_type != edge.edge_type
                    or existing_edge.target_ref != edge.target_ref
                    or existing_edge.assertion_mode != edge.assertion_mode
                    or existing_edge.metadata != edge.metadata
                ):
                    store.put_edge(edge, allow_revision=True, reason='Sync Synapse v3 source revision edge')
                else:
                    store.put_edge(edge)

            placeholders = ', '.join('?' for _ in edge_ids_for_key)
            store.connection.execute(
                f'DELETE FROM edges WHERE edge_id LIKE ? AND edge_id NOT IN ({placeholders})',
                [f'{PREFIX}source-link:{key}:%', *edge_ids_for_key],
            )

        bundle_id = PREFIX + 'quantity-bundle'
        bundle = ObjectRecord(
            object_id=bundle_id,
            kind='typed_spec',
            canonical_name=bundle_id,
            domain_type='quantity_bundle',
            spec_payload={
                'format_version': 1,
                'bundle_id': bundle_id,
                'policy_digest': POLICY_DIGEST,
                'quantity_definitions': [_quantity_payload(k) for k in DEFINITIONS],
                'definition_refs': {k: asdict(definition_ref(k)) for k in DEFINITIONS},
                'claim_digests': claim_digests,
                'source_digests': source_digests,
            },
            title=bundle_id,
            exposition='Symbolic definitions only; no fabricated study, scalar observation, fit or execution.',
        )
        existing_bundle = store.get_object(bundle_id)
        if existing_bundle is not None and existing_bundle.meaning_digest != bundle.meaning_digest:
            store.put_object(bundle, allow_revision=True, reason='Sync Synapse v3 quantity bundle')
        else:
            store.put_object(bundle)

        sections = tuple(
            AssemblySection(
                section_id=PREFIX + 'section:' + role,
                heading=role,
                summary='Inspect declared definitions with source correspondence and remaining conditions.',
                item_refs=tuple(PREFIX + 'statement:' + key for key, entry in DEFINITIONS.items() if entry['role'] == role),
            )
            for role in ('target', 'measurement', 'loss')
        )
        store.put_assembly(
            TopicAssembly(
                assembly_id=PREFIX + 'assembly',
                title='Volatility target, proxy and loss definition map',
                summary='Navigate exact source revisions, shared local definitions and unresolved empirical uses.',
                sections=sections,
                tags=('shared-volatility-definitions', 'target-proxy-loss'),
            )
        )
    return {'definition_count': len(DEFINITIONS), 'claim_digests': claim_digests, 'source_digests': source_digests}


def inspect_definition(store: object, key: str) -> dict:
    """Read source mappings and feasibility from the persisted v3.0 Synapse store."""
    from synapse.federation import evaluate_claim_epistemic_state, evaluate_warrant_epistemic_state

    sid = PREFIX + 'statement:' + key
    cid = PREFIX + 'construct:' + key
    statement = store.get_claim(sid)
    gap = store.get_object(PREFIX + 'use-gap:' + key)
    bundle = store.get_object(PREFIX + 'quantity-bundle')
    if statement is None or gap is None or bundle is None:
        raise KeyError(f'Missing persisted definition records for {key!r}')

    raw_q_defs = bundle.spec_payload.get('quantity_definitions')
    target_def_id = PREFIX + 'quantity:' + key
    matches: list[dict] = []
    if isinstance(raw_q_defs, list):
        matches = [
            item
            for item in raw_q_defs
            if isinstance(item, dict) and item.get('definition_id') == target_def_id
        ]
    elif isinstance(raw_q_defs, dict):
        candidate = raw_q_defs.get(key)
        if isinstance(candidate, dict) and candidate.get('definition_id') == target_def_id:
            matches = [candidate]
    else:
        raise ValueError(f'Invalid quantity_definitions container in {bundle.object_id}')
    if len(matches) != 1 or not isinstance(matches[0].get('formula'), str):
        raise ValueError(f'Missing or malformed quantity definition {key!r} in {bundle.object_id}')
    q_payload = matches[0]
    persisted_def_ref = {
        'kind': 'quantity_definition',
        'logical_id': q_payload['definition_id'],
        'revision_digest': _digest(q_payload),
    }

    deps = gap.spec_payload.get('dependencies')
    if not isinstance(deps, list) or not deps:
        raise ValueError('Missing or malformed source dependencies')
    expected_by_source: dict[str, str] = {}
    for dep in deps:
        if not isinstance(dep, dict) or dep.get('dependency_kind') != 'source':
            raise ValueError('Missing or malformed source dependencies')
        t_ref = dep.get('target_ref')
        s_ref = dep.get('source_ref')
        expected_rev = dep.get('expected_revision_digest')
        if (
            not isinstance(t_ref, dict)
            or _normalize_claim_ref(store, str(t_ref.get('node_id', ''))) != sid
            or not isinstance(s_ref, dict)
            or not isinstance(s_ref.get('node_id'), str)
            or not s_ref['node_id']
            or not isinstance(expected_rev, str)
            or not expected_rev
        ):
            raise ValueError('Missing or malformed source dependencies')
        src_id = _normalize_claim_ref(store, s_ref['node_id'])
        if src_id in expected_by_source:
            raise ValueError(f'Duplicate source dependency {src_id!r}')
        expected_by_source[src_id] = expected_rev

    all_in_entails = [
        e for e in store.list_edges(edge_type='entails') if _normalize_claim_ref(store, e.target_ref) == sid
    ]
    pre_refresh_v1 = (not all_in_entails) and (gap.spec_payload.get('v1_node_type') == 'obligation')
    if not pre_refresh_v1:
        seen_sources: dict[str, object] = {}
        for edge in all_in_entails:
            src_norm = _normalize_claim_ref(store, edge.source_ref)
            if (
                edge.assertion_mode != 'conjectured'
                or edge.metadata.get('dependency_kind') != 'source'
                or src_norm not in expected_by_source
                or edge.metadata.get('expected_revision_digest') != expected_by_source[src_norm]
                or src_norm in seen_sources
            ):
                raise ValueError('Unexpected or mismatched source entails edges')
            seen_sources[src_norm] = edge
        if set(seen_sources) != set(expected_by_source):
            raise ValueError('Unexpected or mismatched source entails edges')

    sources: list[dict[str, object]] = []
    for source_id, expected_rev in expected_by_source.items():
        src_claim = store.get_claim(source_id)
        observed = None if src_claim is None else src_claim.meaning_digest
        src_passages = [
            p.model_dump(mode='json')
            for e in store.list_edges(source_ref=source_id, edge_type='sourced_from')
            if (p := store.get_passage(e.target_ref)) is not None
        ]
        sources.append(
            {
                'statement_id': source_id,
                'expected_revision': expected_rev,
                'observed_revision': observed,
                'current': observed == expected_rev,
                'conclusion': None if src_claim is None else (src_claim.title or src_claim.mathematical_latex),
                'passages': src_passages,
            }
        )

    badge, warrants, warrant_reasons = evaluate_claim_epistemic_state(store, statement)
    validated_warrants = [(w, evaluate_warrant_epistemic_state(store, w)[0]) for w in warrants]
    if badge == 'falsified':
        deductive_status = 'falsified'
        inferential_status = 'falsified'
    else:
        deductive_status = (
            'proved'
            if badge == 'warranted'
            and any(
                w.warrant_kind in ('formal_lean', 'deductive_proof') and eff == 'valid'
                for w, eff in validated_warrants
            )
            else 'unproved'
        )
        inferential_status = (
            'supported'
            if badge == 'warranted'
            and any(
                w.warrant_kind in ('empirical_study', 'computational_check') and eff == 'valid'
                for w, eff in validated_warrants
            )
            else 'untested'
        )
    empirical_support = bool(
        badge == 'warranted'
        and any(w.warrant_kind == 'empirical_study' and eff == 'valid' for w, eff in validated_warrants)
    )
    return {
        'statement_id': sid,
        'definition_ref': persisted_def_ref,
        'definition_matches_current_checker': persisted_def_ref == asdict(definition_ref(key)),
        'scientific_status': {
            'epistemic_badge': badge,
            'deductive_status': deductive_status,
            'inferential_status': inferential_status,
            'warrant_count': len(warrants),
            'reasons': list(warrant_reasons),
        },
        'construct_feasibility': compute_construct_feasibility(store, cid),
        'sources': sources,
        'open_use_obligation': gap.model_dump(mode='json'),
        'empirical_support': empirical_support,
    }


def _resolve_db_path(project: str | None, db: str | None) -> Path:
    if db is not None:
        return Path(db).resolve()
    if project is not None:
        return (Path(project).resolve() / '.synapse' / 'knowledge.sqlite').resolve()
    raise ValueError('Pass either --db or --project')


def _probe_staging_store(
    ro: object,
    packet: dict,
    prior_claim_digests: dict[str, str],
) -> tuple[object, dict[str, dict[str, object]], dict[str, dict], object]:
    from synapse.logic import audit_store
    from synapse.rag import SynapseRAG

    audit_rep = audit_store(ro)
    if audit_rep.dangling_refs:
        raise ValueError(f'Dangling refs after definition publication: {audit_rep.dangling_refs}')

    current_claim_digests = {
        str(r[0]): str(r[1])
        for r in ro.connection.execute('SELECT claim_id, meaning_digest FROM claims').fetchall()
    }
    for sid, dig in prior_claim_digests.items():
        if not sid.startswith(PREFIX):
            if current_claim_digests.get(sid) != dig:
                raise ValueError(f'Unexpected modification to existing claim {sid}')
        elif sid.startswith(PREFIX + 'statement:'):
            key = sid[len(PREFIX + 'statement:') :]
            existing_claim = ro.get_claim(sid)
            if (
                key in DEFINITIONS
                and existing_claim is not None
                and existing_claim.mathematical_latex == DEFINITIONS[key]['formula']
                and current_claim_digests.get(sid) != dig
            ):
                raise ValueError(f'Existing definition claim {sid} meaning_digest shifted ({dig} != {current_claim_digests.get(sid)})')

    rag = SynapseRAG(ro)
    probes: dict[str, dict[str, object]] = {}
    probe_details: dict[str, dict] = {}
    for key in ('conditional_variance', 'squared_return', 'realized_stddev', 'qlike'):
        sid = PREFIX + 'statement:' + key
        resp = rag.search(f"Shared definition {key.replace('_', ' ')}", limit=12, include_passages=True)
        hit_ids = {h.claim_id for h in resp.claim_hits}
        detail = inspect_definition(ro, key)
        if sid not in hit_ids:
            raise ValueError(f'RAG search missed definition claim {sid}')
        if not detail['definition_matches_current_checker']:
            raise ValueError(f'Definition checker mismatch for {key}')
        if not detail['sources'] or not all(s['current'] for s in detail['sources']):
            raise ValueError(f'Stale or missing source mappings for {key}')
        sci = detail['scientific_status']
        if (
            sci['epistemic_badge'] != 'conjectured'
            or sci['deductive_status'] != 'unproved'
            or sci['inferential_status'] != 'untested'
            or detail['empirical_support'] is not False
        ):
            raise ValueError(f'Unexpected scientific status for {key}: {sci}')
        if key == 'conditional_variance' and detail['construct_feasibility']['status'] != 'theoretical_oracle':
            raise ValueError(f'Unexpected feasibility for {key}')
        if key in {'squared_return', 'realized_stddev'} and detail['construct_feasibility']['status'] != 'lookahead_dependent':
            raise ValueError(f'Unexpected feasibility for {key}')
        probe_details[key] = detail
        probes[key] = {
            'definition_retrieved': sid in hit_ids,
            'rag_claim_hit': sid in hit_ids,
            'source_revisions_current': True,
            'status_unpromoted': True,
            'feasibility': detail['construct_feasibility']['status'],
        }
    return audit_rep, probes, probe_details, ro.store_info()


def _unlink_sqlite_sidecars(db_path: Path) -> None:
    for suffix in ('-wal', '-shm', '-journal'):
        sidecar = Path(f'{db_path}{suffix}')
        if sidecar.exists():
            sidecar.unlink()


def main() -> None:
    import argparse
    import os
    import shutil
    import tempfile

    from synapse.store import SynapseStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=None, help='Project root containing .synapse/knowledge.sqlite')
    parser.add_argument('--db', default=None, help='Direct path to Synapse v3.0 SQLite database')
    parser.add_argument('--packet', default=str(Path(__file__).with_name('definition_slice_sources.json')))
    parser.add_argument('--output', required=True)
    parser.add_argument('--record', action='store_true', help='Persist definitions after backup; otherwise audit on a backup copy')
    args = parser.parse_args()

    db_path = _resolve_db_path(args.project, args.db)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    packet = json.loads(Path(args.packet).read_text(encoding='utf-8'))

    backup_db = output / 'pre-definition-backup.sqlite'
    with SynapseStore(db_path, read_only=True) as ro_init:
        info_before = ro_init.store_info()
        if info_before['user_version'] < 300:
            raise ValueError(f'Expected Synapse v3 store (user_version=300), got {info_before["user_version"]}')
        ro_init.backup_to(backup_db)

    with SynapseStore(backup_db, read_only=True) as ro_bak:
        prior_claim_digests = {
            str(r[0]): str(r[1])
            for r in ro_bak.connection.execute('SELECT claim_id, meaning_digest FROM claims').fetchall()
        }
    (output / 'before-claim-revisions.json').write_text(_json(prior_claim_digests) + '\n', encoding='utf-8')
    (output / 'definition-inventory.json').write_text(
        json.dumps(
            {
                key: {**dict(spec), 'native_ref': asdict(definition_ref(key)), 'source_ids': packet['definition_sources'][key]}
                for key, spec in DEFINITIONS.items()
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )

    staging_db = output / 'staging-audit.sqlite'
    shutil.copyfile(backup_db, staging_db)
    with SynapseStore(staging_db) as st_store:
        delta_res = build_delta(st_store, packet)
        st_store.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        st_store.connection.execute('PRAGMA journal_mode = DELETE')

    with SynapseStore(staging_db, read_only=True) as ro_stage:
        audit_rep, probes, probe_details, info_after = _probe_staging_store(ro_stage, packet, prior_claim_digests)

    if not args.record:
        print(
            _json(
                {
                    'mode': 'audit_only',
                    'ready': True,
                    'definition_count': delta_res['definition_count'],
                    'dangling_refs': list(audit_rep.dangling_refs),
                    'consumer_queries': probes,
                    'policy_digest': POLICY_DIGEST,
                }
            )
        )
        return

    fd, tmp_target_str = tempfile.mkstemp(prefix='.synapse-stage-', suffix='.sqlite', dir=str(db_path.parent))
    os.close(fd)
    tmp_target = Path(tmp_target_str)
    try:
        shutil.copyfile(staging_db, tmp_target)
        os.replace(tmp_target, db_path)
    except Exception:
        if tmp_target.exists():
            tmp_target.unlink()
        raise
    _unlink_sqlite_sidecars(db_path)

    with SynapseStore(db_path, read_only=True) as ro_final:
        audit_rep, probes, probe_details, info_after = _probe_staging_store(ro_final, packet, prior_claim_digests)
        for key, detail in probe_details.items():
            (output / f'{key}-source-context.json').write_text(json.dumps(detail, indent=2) + '\n', encoding='utf-8')

    summary = {
        'store': str(db_path),
        'user_version': info_after['user_version'],
        'definition_count': len(DEFINITIONS),
        'source_claims_reused': len(packet['sources']),
        'prior_claim_revisions_preserved': len(prior_claim_digests),
        'dangling_refs': len(audit_rep.dangling_refs),
        'consumer_queries': probes,
        'scientific_execution': 'not_run',
        'formal_checking': 'not_run',
        'empirical_support': False,
    }
    (output / 'consumer-review.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
