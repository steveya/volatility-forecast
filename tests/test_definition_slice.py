"""Target/proxy/loss reuse boundaries; no market-data experiment."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest

MODULE = Path(__file__).parents[1] / 'research' / 'definition_slice.py'
spec = importlib.util.spec_from_file_location('volatility_definition_slice', MODULE)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def scope(**changes):
    base = mod.ForecastScope(
        target=mod.definition_ref('conditional_variance'),
        measurement=mod.definition_ref('squared_return'),
        loss=mod.definition_ref('mse'),
        horizon='one trading session', unit='decimal log-return squared',
        return_convention='close-to-close log return', information_set='F_t',
        origin_rule='session t close', population='declared instrument panel',
        sample='frozen sample v1', rows=('r1', 'r2'), weights=(1.0, 1.0))
    return replace(base, **changes)


def test_paired_descriptive_comparison_has_no_scientific_promotion():
    result = mod.compare(scope(), scope(), (1.0, 4.0), (2.0, 3.0), (1.0, 2.0))
    assert result.status == 'compatible'
    assert result.scores == (1.0, 2.0)
    assert result.empirical_support is False and result.independent_replication is False
    assert result.retained_conditions and result.policy_digest


@pytest.mark.parametrize('field,value', [
    ('target', 'conditional_stddev'), ('measurement', 'realized_variance'),
    ('loss', 'rmse'), ('horizon', 'five sessions'), ('unit', 'percent squared'),
    ('return_convention', 'simple return'), ('information_set', 'F_t_plus_1'),
    ('origin_rule', 'next session close'), ('population', 'another market'),
    ('sample', 'another sample'), ('rows', ('r2', 'r1')), ('weights', (2.0, 1.0)),
])
def test_same_display_label_cannot_hide_changed_comparison_context(field, value):
    if field in {'target', 'measurement', 'loss'}:
        value = mod.definition_ref(value)
    report = mod.assess(scope(), scope(**{field: value}))
    assert report.status == 'incompatible'
    assert field in ' '.join(report.reasons)


@pytest.mark.parametrize('field', ['horizon', 'unit', 'return_convention', 'information_set', 'origin_rule', 'population', 'sample', 'rows', 'weights'])
def test_equal_missing_metadata_does_not_pass_as_equal_meaning(field):
    report = mod.assess(scope(**{field: None}), scope(**{field: None}))
    assert report.status == 'unresolved'


def test_source_conditions_cannot_be_turned_into_a_caller_verified_flag():
    with pytest.raises(TypeError):
        mod.ForecastScope(**{**scope().__dict__, 'proxy_unbiased': True})
    report = mod.assess(scope(), scope(), requested_use='latent_expected_ranking')
    assert report.status == 'unresolved' and not report.empirical_support


def test_variance_and_volatility_observations_need_a_mapping():
    s = scope(measurement=mod.definition_ref('realized_stddev'))
    assert mod.assess(s, s).status == 'incompatible'
    assert mod.DEFINITIONS['corsi_weekly_volatility']['formula'] != mod.DEFINITIONS['realized_stddev']['formula']


def test_current_definition_revision_is_required():
    old = replace(scope().loss, revision_digest='a' * 64)
    assert mod.assess(scope(loss=old), scope(loss=old)).status == 'unresolved'


@pytest.mark.parametrize('field,value', [('conditions', 'A different retained scientific condition'), ('scale', 'a different scale')])
def test_definition_revision_covers_semantics_beyond_formula(monkeypatch, field, value):
    from types import MappingProxyType
    before = mod.definition_ref('conditional_variance')
    changed = {key:dict(spec) for key,spec in mod.DEFINITIONS.items()}
    changed['conditional_variance'][field] = value
    monkeypatch.setattr(mod, 'DEFINITIONS', MappingProxyType(changed))
    assert mod.definition_ref('conditional_variance') != before


@pytest.mark.parametrize('field,value', [('horizon', 5), ('unit', 42), ('return_convention', True), ('information_set', 8.0), ('origin_rule', ('bad',)), ('population', []), ('sample', {})])
def test_equal_malformed_context_is_incompatible(field, value):
    s = scope(**{field:value})
    assert mod.assess(s, s).status == 'incompatible'


@pytest.mark.parametrize('loss', ['mse', 'rmse', 'qlike'])
def test_zero_proxy_is_allowed_for_the_declared_losses(loss):
    s = scope(loss=mod.definition_ref(loss))
    r = mod.compare(s, s, (0.0, 1.0), (1.0, 1.0), (2.0, 2.0))
    assert r.status == 'compatible' and all(x is not None for x in r.scores)


def test_ratio_qlike_zero_domain_is_distinct():
    s = scope(loss=mod.definition_ref('qlike_ratio'))
    assert mod.compare(s, s, (0.0, 1.0), (1.0, 1.0), (2.0, 2.0)).status == 'incompatible'


@pytest.mark.parametrize('values', [(float('nan'), 1.0), (float('inf'), 1.0), (-1.0, 1.0), (True, 1.0), (1.0,)])
def test_invalid_observations_never_create_a_score(values):
    r = mod.compare(scope(), scope(), values, (1.0, 1.0), (2.0, 2.0))
    assert r.status == 'incompatible' and r.scores is None


@pytest.mark.parametrize('loss', ['qlike', 'qlike_ratio'])
def test_nonpositive_variance_forecast_is_outside_qlike_domain(loss):
    s = scope(loss=mod.definition_ref(loss))
    assert mod.compare(s, s, (1.0, 1.0), (0.0, 1.0), (1.0, 1.0)).status == 'incompatible'


def test_mse_and_rmse_preserve_order_on_matched_data_without_changing_target():
    a = mod.compare(scope(), scope(), (1.0, 4.0), (2.0, 3.0), (1.0, 2.0))
    s = scope(loss=mod.definition_ref('rmse'))
    b = mod.compare(s, s, (1.0, 4.0), (2.0, 3.0), (1.0, 2.0))
    assert b.scores == pytest.approx(tuple(x ** 0.5 for x in a.scores))


def test_native_refs_and_source_corrections_are_checked(tmp_path):
    pytest.importorskip('synapse')
    from synapse.graph.dag import EpistemicDAG
    from synapse.models.core import AtomicStatement
    from synapse.models.obligations import KnowledgeRef
    from synapse.obligations import capture_reference
    from synapse.quantities import quantity_ref
    base = EpistemicDAG()
    base.add_statement(AtomicStatement(statement_id='fixture:source', name='fixture source',
        conclusion_latex='x', conclusion_text='Declared source context'))
    packet = {'format_version': 1, 'sources': {'fixture:source': capture_reference(base, KnowledgeRef(kind='statement', node_id='fixture:source'))},
              'definition_sources': {key: ['fixture:source'] for key in mod.DEFINITIONS}}
    delta = mod.build_delta(base, packet)
    for key in ('conditional_variance', 'conditional_second_moment', 'conditional_stddev'):
        assert delta.compute_construct_feasibility(mod.PREFIX+'construct:'+key).status.value == 'theoretical_oracle'
    assert delta.compute_construct_feasibility(mod.PREFIX+'construct:squared_return').status.value == 'lookahead_dependent'
    bundle = next(iter(delta.quantity_bundles.values()))
    assert len(bundle.quantity_definitions) == len(mod.DEFINITIONS)
    for definition in bundle.quantity_definitions:
        ref = quantity_ref(definition)
        key = definition.definition_id.split(':')[-1]
        assert ref.model_dump() == mod.definition_ref(key).__dict__
    # Inspect a complete persisted graph, preserving the source context separately.
    from synapse.graph.importing import NATIVE_REGISTRIES
    from synapse.storage import SQLiteEpistemicStore
    for registry, _, _, _ in NATIVE_REGISTRIES:
        getattr(base,registry).update(getattr(delta,registry))
    store=SQLiteEpistemicStore(tmp_path/'definitions.sqlite')
    store.save_dag(base)
    loaded=store.load_dag()
    detail=mod.inspect_definition(loaded,'conditional_variance')
    assert detail['sources'][0]['statement_id']=='fixture:source'
    assert detail['sources'][0]['current'] and detail['definition_matches_current_checker']
    assert detail['construct_feasibility']['status']=='theoretical_oracle'
    assert detail['scientific_status']['deductive_status']=='unproved'
    loaded.statements['fixture:source'].conclusion_text='Changed after publication'
    assert not mod.inspect_definition(loaded,'conditional_variance')['sources'][0]['current']
    base.statements['fixture:source'].conclusion_text = 'Changed source interpretation'
    with pytest.raises(ValueError, match='source revision'):
        mod.build_delta(base, packet)
