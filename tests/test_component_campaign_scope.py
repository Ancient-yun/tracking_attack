import copy
import json
from pathlib import Path
import pytest
from st4rtrack_pgd.run_component_study import checked_config


def config_and_manifest(n=20):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/loss_components.json').read_text())
    config.update(runtime_optimization=True, campaign_expected_clips=n)
    manifest = {'num_frames': 64, 'campaign_expected_clips': n, 'entries': [
        {'dataset': dataset, 'sequence': f'clip{i}'}
        for dataset in ('po_mini', 'ds_mini') for i in range(n // 2)]}
    return config, manifest


def test_declared_balanced_campaign_and_explicit_limited_calibration():
    config, manifest = config_and_manifest()
    checked_config(config, manifest)
    manifest['entries'] = manifest['entries'][:1]
    with pytest.raises(ValueError, match='balanced'):
        checked_config(config, manifest)
    manifest['purpose'] = 'explicitly limited preflight; not full campaign'
    checked_config(config, manifest)


@pytest.mark.parametrize('value', [0, -2, 21, True, 20.0, '20', None])
def test_bad_campaign_declaration_rejected(value):
    config, manifest = config_and_manifest()
    config['campaign_expected_clips'] = value
    with pytest.raises(ValueError, match='campaign_expected_clips'):
        checked_config(config, manifest)


def test_mismatched_missing_and_unbalanced_declarations_rejected():
    config, manifest = config_and_manifest()
    for changed in ({**manifest, 'campaign_expected_clips': 22},
                    {k: v for k, v in manifest.items() if k != 'campaign_expected_clips'}):
        with pytest.raises(ValueError, match='campaign_expected_clips'):
            checked_config(config, changed)
    unbalanced = copy.deepcopy(manifest)
    unbalanced['entries'][-1]['dataset'] = 'po_mini'
    unbalanced['entries'][-1]['sequence'] = 'extra_clip'
    with pytest.raises(ValueError, match='balanced'):
        checked_config(config, unbalanced)


def test_optimization_flag_must_be_boolean():
    config, manifest = config_and_manifest()
    config['runtime_optimization'] = 'true'
    with pytest.raises(ValueError, match='boolean'):
        checked_config(config, manifest)


def test_all_released_frames_declaration_enforced():
    config, manifest = config_and_manifest(8)
    for item in (config, manifest):
        item.update(all_frames=True, num_frames=128, campaign_expected_frames=128)
    checked_config(config, manifest)
    partial = copy.deepcopy(manifest)
    partial['num_frames'] = 64
    with pytest.raises(ValueError, match='128'):
        checked_config(config, partial)
    partial = copy.deepcopy(manifest)
    partial['all_frames'] = False
    with pytest.raises(ValueError, match='all_frames'):
        checked_config(config, partial)
    partial = copy.deepcopy(manifest)
    del partial['campaign_expected_frames']
    with pytest.raises(ValueError, match='128'):
        checked_config(config, partial)
