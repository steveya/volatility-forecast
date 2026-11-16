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
    from synapse.models import Claim
    from synapse.store import SynapseStore

    db_path = tmp_path / 'definitions.sqlite'
    with SynapseStore(db_path) as store:
        src = Claim(
            claim_id='fixture:source',
            claim_kind='theorem',
            mathematical_latex='x',
            title='fixture source',
            exposition='Declared source context',
        )
        store.put_claim(src)
        packet = {
            'format_version': 1,
            'sources': {'fixture:source': src.meaning_digest},
            'definition_sources': {key: ['fixture:source'] for key in mod.DEFINITIONS},
        }
        mod.build_delta(store, packet)

    with SynapseStore(db_path, read_only=True) as ro:
        for key in ('conditional_variance', 'conditional_second_moment', 'conditional_stddev'):
            assert mod.compute_construct_feasibility(ro, mod.PREFIX + 'construct:' + key)['status'] == 'theoretical_oracle'
        assert mod.compute_construct_feasibility(ro, mod.PREFIX + 'construct:squared_return')['status'] == 'lookahead_dependent'

        bundle = ro.get_object(mod.PREFIX + 'quantity-bundle')
        assert bundle is not None
        q_defs = bundle.spec_payload['quantity_definitions']
        assert isinstance(q_defs, list) and len(q_defs) == len(mod.DEFINITIONS)
        for key, ref_dict in bundle.spec_payload['definition_refs'].items():
            assert ref_dict == mod.definition_ref(key).__dict__

        detail = mod.inspect_definition(ro, 'conditional_variance')
        assert detail['sources'][0]['statement_id'] == 'fixture:source'
        assert detail['sources'][0]['current'] and detail['definition_matches_current_checker']
        assert detail['construct_feasibility']['status'] == 'theoretical_oracle'
        assert detail['scientific_status']['epistemic_badge'] == 'conjectured'
        assert detail['scientific_status']['deductive_status'] == 'unproved'
        assert detail['scientific_status']['inferential_status'] == 'untested'

    with SynapseStore(db_path) as store:
        revised = Claim(
            claim_id='fixture:source',
            claim_kind='theorem',
            mathematical_latex='y',
            title='fixture source',
            exposition='Changed after publication',
        )
        store.put_claim(revised, allow_revision=True, reason='Test source revision invalidation')
        assert not mod.inspect_definition(store, 'conditional_variance')['sources'][0]['current']
        with pytest.raises(ValueError, match='source revision'):
            mod.build_delta(store, packet)


def test_definition_slice_cli_audit_only_and_record_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip('synapse')
    import json
    from synapse.models import Claim
    from synapse.store import SynapseStore

    db_path = tmp_path / 'cli_defs.sqlite'
    with SynapseStore(db_path) as store:
        src = Claim(
            claim_id='fixture:source',
            claim_kind='theorem',
            mathematical_latex='x',
            title='fixture source',
        )
        store.put_claim(src)
        packet_path = tmp_path / 'packet.json'
        packet_path.write_text(
            json.dumps(
                {
                    'format_version': 1,
                    'sources': {'fixture:source': src.meaning_digest},
                    'definition_sources': {key: ['fixture:source'] for key in mod.DEFINITIONS},
                }
            ),
            encoding='utf-8',
        )

    out_audit = tmp_path / 'out_audit'
    monkeypatch.setattr(sys, 'argv', ['definition_slice', '--db', str(db_path), '--packet', str(packet_path), '--output', str(out_audit)])
    mod.main()
    with SynapseStore(db_path, read_only=True) as ro:
        assert ro.get_claim(mod.PREFIX + 'statement:conditional_variance') is None

    out_rec = tmp_path / 'out_rec'
    monkeypatch.setattr(sys, 'argv', ['definition_slice', '--db', str(db_path), '--packet', str(packet_path), '--output', str(out_rec), '--record'])
    mod.main()
    summary = json.loads((out_rec / 'consumer-review.json').read_text(encoding='utf-8'))
    assert summary['definition_count'] == len(mod.DEFINITIONS)
    assert summary['dangling_refs'] == 0
    assert summary['empirical_support'] is False
    for probe in summary['consumer_queries'].values():
        assert probe['definition_retrieved'] is True
        assert probe['rag_claim_hit'] is True
    assert not Path(f'{db_path}-wal').exists()
    assert not Path(f'{db_path}-shm').exists()
    assert (out_rec / 'qlike-source-context.json').is_file()


def test_inspect_definition_evaluates_warrants_and_build_delta_blocks_warranted(tmp_path):
    pytest.importorskip('synapse')
    from synapse.models import Claim, Warrant
    from synapse.store import SynapseStore

    db_path = tmp_path / 'warrants.sqlite'
    sid = mod.PREFIX + 'statement:conditional_variance'
    with SynapseStore(db_path) as store:
        src = Claim(claim_id='fixture:source', claim_kind='theorem', mathematical_latex='x', title='fixture source')
        store.put_claim(src)
        packet = {
            'format_version': 1,
            'sources': {'fixture:source': src.meaning_digest},
            'definition_sources': {key: ['fixture:source'] for key in mod.DEFINITIONS},
        }
        mod.build_delta(store, packet)
        target_claim = store.get_claim(sid)
        assert target_claim is not None

        store.put_warrant(
            Warrant(
                warrant_id='fixture:w:stale',
                warrant_kind='formal_lean',
                target_claim_id=sid,
                locked_meaning_digest='0' * 64,
                status='valid',
            )
        )
        stale_detail = mod.inspect_definition(store, 'conditional_variance')
        assert stale_detail['scientific_status']['epistemic_badge'] == 'stale'
        assert stale_detail['scientific_status']['deductive_status'] == 'unproved'
        assert stale_detail['scientific_status']['inferential_status'] == 'untested'
        assert stale_detail['empirical_support'] is False

        with pytest.raises(ValueError, match='warranted'):
            mod.build_delta(store, packet)

        store.put_warrant(
            Warrant(
                warrant_id='fixture:w:emp',
                warrant_kind='empirical_study',
                target_claim_id=sid,
                locked_meaning_digest=target_claim.meaning_digest,
                status='valid',
            )
        )
        blocked_detail = mod.inspect_definition(store, 'conditional_variance')
        assert blocked_detail['scientific_status']['epistemic_badge'] == 'stale'
        assert blocked_detail['empirical_support'] is False

        store.put_warrant(
            Warrant(
                warrant_id='fixture:w:source',
                warrant_kind='deductive_proof',
                target_claim_id='fixture:source',
                locked_meaning_digest=src.meaning_digest,
                status='valid',
            )
        )
        emp_detail = mod.inspect_definition(store, 'conditional_variance')
        assert emp_detail['scientific_status']['epistemic_badge'] == 'warranted'
        assert emp_detail['scientific_status']['inferential_status'] == 'supported'
        assert emp_detail['scientific_status']['deductive_status'] == 'unproved'
        assert emp_detail['empirical_support'] is True

        store.put_warrant(
            Warrant(
                warrant_id='fixture:w:bad',
                warrant_kind='deductive_proof',
                target_claim_id=sid,
                locked_meaning_digest=target_claim.meaning_digest,
                status='falsified',
            )
        )
        bad_detail = mod.inspect_definition(store, 'conditional_variance')
        assert bad_detail['scientific_status']['epistemic_badge'] == 'falsified'
        assert bad_detail['scientific_status']['deductive_status'] == 'falsified'
        assert bad_detail['scientific_status']['inferential_status'] == 'falsified'
        assert bad_detail['empirical_support'] is False


def test_inspect_definition_rejects_tampered_bundle_deps_and_edges(tmp_path):
    pytest.importorskip('synapse')
    from synapse.models import Claim, Edge
    from synapse.store import SynapseStore

    db_path = tmp_path / 'tamper.sqlite'
    sid = mod.PREFIX + 'statement:conditional_variance'
    cid = mod.PREFIX + 'construct:conditional_variance'
    gap_id = mod.PREFIX + 'use-gap:conditional_variance'
    bundle_id = mod.PREFIX + 'quantity-bundle'

    with SynapseStore(db_path) as store:
        src = Claim(claim_id='fixture:source', claim_kind='theorem', mathematical_latex='x', title='fixture source')
        store.put_claim(src)
        packet = {
            'format_version': 1,
            'sources': {'fixture:source': src.meaning_digest},
            'definition_sources': {key: ['fixture:source'] for key in mod.DEFINITIONS},
        }
        mod.build_delta(store, packet)

        bundle = store.get_object(bundle_id)
        tampered_payload = dict(bundle.spec_payload)
        tampered_q = [dict(item) for item in tampered_payload['quantity_definitions']]
        tampered_q[0]['formula'] = '{"formula":"TAMPERED"}'
        tampered_payload['quantity_definitions'] = tampered_q
        store.put_object(bundle.model_copy(update={'spec_payload': tampered_payload}), allow_revision=True, reason='tamper q')
        tampered_detail = mod.inspect_definition(store, 'conditional_variance')
        assert tampered_detail['definition_matches_current_checker'] is False
        assert tampered_detail['definition_ref']['revision_digest'] == mod._digest(tampered_q[0])

        mod.build_delta(store, packet)
        gap = store.get_object(gap_id)
        store.put_object(
            gap.model_copy(update={'spec_payload': {**gap.spec_payload, 'dependencies': []}}),
            allow_revision=True,
            reason='empty deps',
        )
        with pytest.raises(ValueError, match='dependencies'):
            mod.inspect_definition(store, 'conditional_variance')

        uri_deps = [dict(gap.spec_payload['dependencies'][0])]
        uri_deps[0]['source_ref'] = {'kind': 'statement', 'node_id': 'synapse://local/claim/fixture:source'}
        store.put_object(
            gap.model_copy(update={'spec_payload': {**gap.spec_payload, 'dependencies': uri_deps}}),
            allow_revision=True,
            reason='local uri dep',
        )
        ok_uri_detail = mod.inspect_definition(store, 'conditional_variance')
        assert ok_uri_detail['sources'][0]['statement_id'] == 'fixture:source'

        store.put_edge(
            Edge(
                edge_id='fixture:extra-entails',
                source_ref='fixture:source',
                edge_type='entails',
                target_ref=sid,
                assertion_mode='conjectured',
                metadata={'dependency_kind': 'source', 'expected_revision_digest': src.meaning_digest},
            )
        )
        with pytest.raises(ValueError, match='entails'):
            mod.inspect_definition(store, 'conditional_variance')

        con = store.get_object(cid)
        store.put_object(
            con.model_copy(update={'spec_payload': {**con.spec_payload, 'uses_symbols': []}}),
            allow_revision=True,
            reason='clear symbols',
        )
        with pytest.raises(KeyError, match='symbol'):
            mod.compute_construct_feasibility(store, cid)


def test_build_delta_atomic_rollback_and_idempotent_source_updates(tmp_path, monkeypatch):
    pytest.importorskip('synapse')
    from synapse.models import Claim
    from synapse.store import SynapseStore

    db_path = tmp_path / 'atomic.sqlite'
    with SynapseStore(db_path) as store:
        src_a = Claim(claim_id='fixture:a', claim_kind='theorem', mathematical_latex='a', title='source a')
        src_b = Claim(claim_id='fixture:b', claim_kind='theorem', mathematical_latex='b', title='source b')
        store.put_claim(src_a)
        store.put_claim(src_b)
        packet_ab = {
            'format_version': 1,
            'sources': {'fixture:a': src_a.meaning_digest, 'fixture:b': src_b.meaning_digest},
            'definition_sources': {
                key: (['fixture:a', 'fixture:b'] if key == 'conditional_variance' else ['fixture:a'])
                for key in mod.DEFINITIONS
            },
        }

        def fail_assembly(*_args, **_kwargs):
            raise RuntimeError('simulated mid-run failure')

        monkeypatch.setattr(store, 'put_assembly', fail_assembly)
        with pytest.raises(RuntimeError, match='simulated mid-run failure'):
            mod.build_delta(store, packet_ab)
        monkeypatch.undo()

        assert store.get_source(mod.PREFIX + 'normalization-source') is None
        assert store.get_object(mod.PREFIX + 'quantity-bundle') is None
        assert store.get_claim(mod.PREFIX + 'statement:conditional_variance') is None
        assert store.connection.execute(f"SELECT COUNT(*) FROM edges WHERE edge_id LIKE '{mod.PREFIX}%'").fetchone()[0] == 0

        mod.build_delta(store, packet_ab)
        packet_ba = {
            **packet_ab,
            'definition_sources': {
                **packet_ab['definition_sources'],
                'conditional_variance': ['fixture:b', 'fixture:a'],
            },
        }
        mod.build_delta(store, packet_ba)
        packet_b_only = {
            'format_version': 1,
            'sources': {'fixture:a': src_a.meaning_digest, 'fixture:b': src_b.meaning_digest},
            'definition_sources': {
                key: (['fixture:b'] if key == 'conditional_variance' else ['fixture:a'])
                for key in mod.DEFINITIONS
            },
        }
        mod.build_delta(store, packet_b_only)
        rows = store.connection.execute(
            'SELECT source_ref FROM edges WHERE edge_id LIKE ?',
            (f'{mod.PREFIX}source-link:conditional_variance:%',),
        ).fetchall()
        assert [r[0] for r in rows] == ['fixture:b']
        detail = mod.inspect_definition(store, 'conditional_variance')
        assert [s['statement_id'] for s in detail['sources']] == ['fixture:b']


def test_live_migrated_store_compatibility_preserves_meaning_digests(tmp_path, monkeypatch, capsys):
    pytest.importorskip('synapse')
    import json
    from synapse.store import SynapseStore

    live_db = Path('/Users/steveyang/Projects/steveya/volatility-forecast/.synapse/knowledge.sqlite')
    if not live_db.is_file():
        pytest.skip('Live volatility-forecast Synapse store not present on disk')
    real_packet = Path(__file__).parents[1] / 'research' / 'definition_slice_sources.json'

    live_copy = tmp_path / 'live-copy.sqlite'
    with SynapseStore(live_db, read_only=True) as ro_live:
        ro_live.backup_to(live_copy)

    with SynapseStore(live_copy, read_only=True) as ro_before:
        before_digests = {
            str(r[0]): str(r[1])
            for r in ro_before.connection.execute('SELECT claim_id, meaning_digest FROM claims').fetchall()
        }
        for key in mod.DEFINITIONS:
            pre_detail = mod.inspect_definition(ro_before, key)
            assert pre_detail['sources'] and not all(s['current'] for s in pre_detail['sources'])
            assert pre_detail['scientific_status']['epistemic_badge'] == 'conjectured'

    out_audit = tmp_path / 'live_audit'
    monkeypatch.setattr(
        sys,
        'argv',
        ['definition_slice', '--db', str(live_copy), '--packet', str(real_packet), '--output', str(out_audit)],
    )
    mod.main()
    audit_payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert audit_payload['mode'] == 'audit_only'
    assert audit_payload['ready'] is True
    assert audit_payload['dangling_refs'] == []

    with SynapseStore(live_copy, read_only=True) as ro_after_audit:
        assert [
            e
            for e in ro_after_audit.list_edges(edge_type='entails')
            if e.target_ref == mod.PREFIX + 'statement:conditional_variance'
        ] == []

    out_rec = tmp_path / 'live_rec'
    monkeypatch.setattr(
        sys,
        'argv',
        ['definition_slice', '--db', str(live_copy), '--packet', str(real_packet), '--output', str(out_rec), '--record'],
    )
    mod.main()
    rec_payload = json.loads(capsys.readouterr().out)
    assert rec_payload['dangling_refs'] == 0

    assert not Path(f'{live_copy}-wal').exists()
    assert not Path(f'{live_copy}-shm').exists()
    with SynapseStore(live_copy, read_only=True) as ro_after:
        after_digests = {
            str(r[0]): str(r[1])
            for r in ro_after.connection.execute('SELECT claim_id, meaning_digest FROM claims').fetchall()
        }
        assert after_digests == before_digests
        for key in mod.DEFINITIONS:
            detail = mod.inspect_definition(ro_after, key)
            assert detail['sources'] and all(s['current'] for s in detail['sources'])
            assert detail['definition_matches_current_checker'] is True
            assert detail['scientific_status']['epistemic_badge'] == 'conjectured'
            assert detail['scientific_status']['deductive_status'] == 'unproved'
            assert detail['scientific_status']['inferential_status'] == 'untested'
            assert detail['empirical_support'] is False
