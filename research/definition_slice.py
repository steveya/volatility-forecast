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


def build_delta(base, packet):
    """Compile reviewed normalization records; original source IDs never change."""
    from synapse.graph.dag import EpistemicDAG
    from synapse.models.core import (AtomicStatement, Construct, ConstructKind, KnowledgeOrigin,
        Scope, SourceDocument, SourceAnchor, TopicAssembly, AssemblySection, Symbol, SymbolRole)
    from synapse.models.quantities import QuantityDefinition, QuantityBundle, ScientificTarget
    from synapse.models.assessment import RevisionRef
    from synapse.models.obligations import KnowledgeObligation, KnowledgeRef, CandidateDependency
    from synapse.obligations import capture_reference
    from synapse.quantities import quantity_ref
    if packet.get('format_version') != 1 or set(packet['definition_sources']) != set(DEFINITIONS):
        raise ValueError('Unsupported or incomplete definition mapping')
    for sid, expected in packet['sources'].items():
        if sid not in base.statements or capture_reference(base, KnowledgeRef(kind='statement', node_id=sid)) != expected:
            raise ValueError('Changed or missing source revision: '+sid)
    for key, sources in packet['definition_sources'].items():
        if not sources or len(sources) != len(set(sources)) or any(sid not in packet['sources'] for sid in sources):
            raise ValueError('Missing or invalid definition source mapping: '+key)
    delta = EpistemicDAG()
    module_path = Path(__file__).resolve()
    doc_id = PREFIX+'normalization-source'
    delta.add_document(SourceDocument(doc_id=doc_id, title='VOL-7 reviewed local operational definitions',
        source_type='code_repo', path_or_uri=str(module_path), content_hash=sha256(module_path.read_bytes()).hexdigest(),
        search_keywords=['shared volatility definitions', 'target proxy loss', POLICY_DIGEST]))
    for key, role, clock, description in (
        ('latent_variance',SymbolRole.UNOBSERVED_TRUTH,'t','Unknown conditional variance given F_t'),
        ('latent_second_moment',SymbolRole.UNOBSERVED_TRUTH,'t','Unknown conditional second moment given F_t'),
        ('latent_mean',SymbolRole.UNOBSERVED_TRUTH,'t','Unknown true conditional mean given F_t'),
        ('future_returns',SymbolRole.OBSERVABLE,'t+h','Returns over the future outcome interval; not known at forecast origin t'),
        ('forecast',SymbolRole.OBSERVABLE,'t','Declared origin-measurable forecast; actual timestamp authentication remains unresolved'),
    ):
        delta.add_symbol(Symbol(symbol_id=PREFIX+'symbol:'+key,name=key,latex=key,
            description=description,role=role,measurable_at=clock))
    for key, entry in DEFINITIONS.items():
        cid, sid = PREFIX+'construct:'+key, PREFIX+'statement:'+key
        symbols = {'conditional_variance':['latent_variance'], 'conditional_second_moment':['latent_second_moment'],
            'conditional_stddev':['latent_variance'], 'squared_innovation':['future_returns','latent_mean']}.get(key,
            ['future_returns','forecast'] if entry['role']=='loss' else ['future_returns'])
        delta.add_construct(Construct(construct_id=cid, name=key.replace('_', ' '),
            kind=ConstructKind.LOSS_FUNCTIONAL if entry['role']=='loss' else ConstructKind.MATHEMATICAL_OBJECT,
            scope=Scope.GLOBAL_CANONICAL, latex_definition=entry['formula'],
            uses_symbols=[PREFIX+'symbol:'+s for s in symbols],decision_time='t+h' if entry['role']=='loss' else 't',
            description='Local versioned operational definition; role='+entry['role']+'; scale='+entry['scale']+'; '+entry['conditions']+
                ' No universal equivalence, implementable estimator, scientific support or formal proof is granted.'))
        delta.add_statement(AtomicStatement(statement_id=sid, name='Shared definition: '+key.replace('_',' '),
            scope=Scope.GLOBAL_CANONICAL, primitive_domain='financial_econometrics.volatility.definitions',
            knowledge_origin=KnowledgeOrigin.AI_DEDUCTIVE_SYNTHESIS, context_constructs=[cid],
            conclusion_latex=entry['formula'], conclusion_text='CURATOR OPERATIONAL DEFINITION: '+entry['formula']+'. '+entry['conditions']+
                ' Source correspondence is scoped normalization, not a new source theorem or empirical acceptance.',
            topic_themes=['shared volatility definitions','target proxy loss',key]))
        delta.add_anchor(SourceAnchor(anchor_id=sid+':anchor', statement_id=sid, doc_id=doc_id,
            role_in_doc='definition_property', label_in_doc='DEFINITIONS['+key+']',page_or_cell='DEFINITIONS['+key+']'))
    targets = tuple(ScientificTarget(target_id=PREFIX+'target:'+key, definition=entry['formula'],
        retained_assumptions=(entry['conditions'],)) for key,entry in DEFINITIONS.items() if entry['role']=='target')
    quantities = tuple(QuantityDefinition(**_quantity_payload(key)) for key in DEFINITIONS)
    for key, quantity in zip(DEFINITIONS, quantities):
        assert quantity_ref(quantity).model_dump() == asdict(definition_ref(key))
    bundle = QuantityBundle(bundle_id=PREFIX+'quantity-bundle', description='Symbolic definitions only; no fabricated study, scalar observation, fit or execution.',
        targets=targets, quantity_definitions=quantities,
        statement_refs=tuple(RevisionRef(kind='statement',logical_id=PREFIX+'statement:'+key,
            revision_digest=capture_reference(delta,KnowledgeRef(kind='statement',node_id=PREFIX+'statement:'+key))) for key in DEFINITIONS),
        source_refs=tuple(RevisionRef(kind='statement',logical_id=sid,revision_digest=digest) for sid,digest in packet['sources'].items()))
    delta.add_quantity_bundle(bundle)
    for key, entry in DEFINITIONS.items():
        sid = PREFIX+'statement:'+key
        links = tuple(CandidateDependency(source_ref=KnowledgeRef(kind='statement',node_id=source),
            target_ref=KnowledgeRef(kind='statement',node_id=sid), expected_revision_digest=packet['sources'][source],
            dependency_kind='source',route_id=PREFIX+'source-route:'+key) for source in packet['definition_sources'][key])
        delta.add_obligation(KnowledgeObligation(record_id=PREFIX+'use-gap:'+key,obligation_id=PREFIX+'use-gap:'+key,
            target_ref=KnowledgeRef(kind='statement',node_id=sid), target_revision_digest=capture_reference(delta,KnowledgeRef(kind='statement',node_id=sid)),
            gap_kind='undercut_application_or_transport', dependencies=links,
            description='Operational definition and source mapping are retained. Before scientific use, instantiate the target-world horizon, information, units, measurement and relevant assumptions. '+entry['conditions'],
            permitted_evidence=('reviewed_target_world_mapping','independently_checked_claim_to_procedure_contract','appropriate_empirical_evaluation'),
            artifact_json=_json({'normalization_key':key,'source_claim_ids':packet['definition_sources'][key],
                'policy_digest':POLICY_DIGEST,'scientific_support':False,'formal_checking':'not_run'})))
    delta.add_assembly(TopicAssembly(assembly_id=PREFIX+'assembly',title='Volatility target, proxy and loss definition map',
        objective='Navigate exact source revisions, shared local definitions and unresolved empirical uses.',
        sections=[AssemblySection(section_id=PREFIX+'section:'+role,title=role,
            narrative_goal='Inspect declared definitions with source correspondence and remaining conditions.',
            primary_statement_ids=[PREFIX+'statement:'+key for key,e in DEFINITIONS.items() if e['role']==role],
            composed_construct_ids=[PREFIX+'construct:'+key for key,e in DEFINITIONS.items() if e['role']==role]) for role in ('target','measurement','loss')]))
    return delta


def inspect_definition(dag, key):
    """Read source mappings from the persisted obligation, not a caller's map."""
    from synapse.obligations import capture_reference
    from synapse.models.obligations import KnowledgeRef
    from synapse.quantities import quantity_ref
    sid = PREFIX+'statement:'+key
    statement = dag.statements[sid]
    gap = dag.obligations[PREFIX+'use-gap:'+key]
    bundle = dag.quantity_bundles[PREFIX+'quantity-bundle']
    quantity = next(q for q in bundle.quantity_definitions if q.definition_id==definition_ref(key).logical_id)
    sources = []
    for edge in gap.dependencies:
        if edge.dependency_kind!='source' or edge.target_ref.node_id!=sid:
            raise ValueError('Unexpected source mapping role or endpoint')
        source_id = edge.source_ref.node_id
        source = dag.statements.get(source_id)
        digest = None if source is None else capture_reference(dag,edge.source_ref)
        sources.append({'statement_id':source_id,'expected_revision':edge.expected_revision_digest,
            'observed_revision':digest,'current':digest==edge.expected_revision_digest,
            'conclusion':None if source is None else source.conclusion_text,
            'anchors':[a.model_dump(mode='json') for a in dag.anchors.values() if a.statement_id==source_id]})
    return {'statement_id':sid,'definition_ref':quantity_ref(quantity).model_dump(mode='json'),
        'definition_matches_current_checker':quantity_ref(quantity).model_dump()==asdict(definition_ref(key)),
        'scientific_status':statement.epistemic_vector.model_dump(mode='json'),
        'construct_feasibility':dag.compute_construct_feasibility(PREFIX+'construct:'+key).model_dump(mode='json'),
        'sources':sources,'open_use_obligation':gap.model_dump(mode='json'),'empirical_support':False}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--packet', default=str(Path(__file__).with_name('definition_slice_sources.json')))
    parser.add_argument('--output', required=True)
    parser.add_argument('--record', action='store_true', help='Publish the passive native delta after preview; otherwise audit only')
    args = parser.parse_args()
    from synapse.projects import bind_project
    from synapse.graph.importing import NATIVE_REGISTRIES
    from synapse.obligations import capture_reference, inspect_obligations
    from synapse.models.obligations import KnowledgeRef
    from synapse.workflows import make_change, preview_change, record_change, ArtifactRequest
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    packet = json.loads(Path(args.packet).read_text())
    selected = bind_project(args.project,include_paper_local_seeds=True)
    if selected.status != 'bound':
        raise ValueError(selected.to_dict())
    base = selected.session.load_dag()
    def inventory(graph):
        return {registry:{key:_digest(node.model_dump(mode='json',exclude={'epistemic_vector','execution_receipt'})) for key,node in getattr(graph,registry).items()} for registry,_,_,_ in NATIVE_REGISTRIES}
    previous = inventory(base)
    previous_revisions = {sid:capture_reference(base,KnowledgeRef(kind='statement',node_id=sid)) for sid in base.statements}
    (output/'before-record-hashes.json').write_text(_json(previous))
    (output/'before-claim-revisions.json').write_text(_json(previous_revisions))
    delta = build_delta(base,packet)
    (output/'definition-inventory.json').write_text(json.dumps({key:{**dict(spec),'native_ref':asdict(definition_ref(key)),'source_ids':packet['definition_sources'][key]} for key,spec in DEFINITIONS.items()},indent=2)+'\n')
    change = make_change(operation_id=PREFIX+'record:'+_digest(packet)+':'+POLICY_DIGEST,
        actor='codex-volatility-definition-curator',reason='VOL-7 reviewed normalization of existing source slice; no scientific execution or proof.',
        graph=delta,queries=('shared volatility definitions conditional variance squared return QLIKE target proxy loss',),
        artifacts=(ArtifactRequest(path=output/'native-context.md',kind='context_markdown'),))
    preview = preview_change(selected.session,change)
    (output/'preview.json').write_text(preview.model_dump_json(indent=2)+'\n')
    if not preview.ready or preview.revised:
        raise ValueError(preview.errors or 'Unexpected revision to existing records')
    if not args.record:
        print(_json({'mode':'audit_only','ready':True,'additions':len(preview.added),'policy_digest':POLICY_DIGEST}))
        return
    report = record_change(selected.session,change,preview)
    (output/'workflow.json').write_text(report.model_dump_json(indent=2)+'\n')
    if report.persistence not in {'committed','unchanged'}:
        raise ValueError('Inspect workflow receipt before any retry: '+str(report.errors))
    # Do not republish if optional automatic artifact delivery needs recovery.
    fresh = bind_project(args.project,include_paper_local_seeds=True)
    if fresh.status != 'bound':
        raise ValueError(fresh.to_dict())
    current = fresh.session.load_dag()
    now = inventory(current)
    assert all(now[registry].get(key)==digest for registry,records in previous.items() for key,digest in records.items())
    assert all(capture_reference(current,KnowledgeRef(kind='statement',node_id=sid))==digest for sid,digest in previous_revisions.items())
    assert all(now[registry].get(key)==digest for registry,records in inventory(delta).items() for key,digest in records.items())
    wanted = set(delta.obligations)
    observed = [x for x in inspect_obligations(current) if x.record.record_id in wanted]
    assert {x.record.record_id for x in observed}==wanted and all(x.freshness=='current' for x in observed)
    current.validate_acyclicity()
    # Exercise the query layer on exact definition IDs with their source/gap cones.
    from synapse.rag.retriever import EpistemicGraphRAG, EpistemicQuery
    rag = EpistemicGraphRAG(current)
    probes = {}
    for key in ('conditional_variance','squared_return','realized_stddev','qlike'):
        sid = PREFIX+'statement:'+key
        response = rag.query(EpistemicQuery(query_text=sid,statement_id_prefix=PREFIX+'statement:',include_paper_local_seeds=True,top_k=12))
        text = response.to_agent_markdown()
        assert sid in text and PREFIX+'use-gap:'+key in text
        (output/(key+'-context.md')).write_text(text)
        detail=inspect_definition(current,key)
        assert detail['definition_matches_current_checker'] and detail['sources'] and all(s['current'] for s in detail['sources'])
        assert detail['scientific_status']['deductive_status']=='unproved' and detail['scientific_status']['inferential_status']=='untested'
        if key=='conditional_variance': assert detail['construct_feasibility']['status']=='theoretical_oracle'
        if key in {'squared_return','realized_stddev'}: assert detail['construct_feasibility']['status']=='lookahead_dependent'
        (output/(key+'-source-context.json')).write_text(json.dumps(detail,indent=2)+'\n')
        probes[key] = {'definition_retrieved':True,'use_obligation_retrieved':True,'source_revisions_current':True,'status_unpromoted':True,'feasibility':detail['construct_feasibility']['status']}
    snapshot = fresh.session.binding.snapshot
    summary = {'store':str(fresh.session.binding.store_path),'snapshot':{'store_id':snapshot.store_id,'revision':snapshot.revision},
        'definition_count':len(DEFINITIONS),'source_claims_reused':len(packet['sources']),
        'prior_records_preserved':sum(map(len,previous.values())),'prior_claim_revisions_preserved':len(previous_revisions),
        'current_open_use_obligations':len(observed),'consumer_queries':probes,
        'automatic_consumer_status':report.consumer_review,'scientific_execution':'not_run','formal_checking':'not_run','empirical_support':False}
    (output/'consumer-review.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
