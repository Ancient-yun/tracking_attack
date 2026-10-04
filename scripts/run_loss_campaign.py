"""Run the declared all-frame loss study, then require artifact audit and analysis."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from st4rtrack_pgd.common import ROOT, resolve_path, write_json
from st4rtrack_pgd.run_component_study import checked_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text(encoding='utf-8'))
    manifest = json.loads(resolve_path(args.manifest).read_text(encoding='utf-8'))
    checked_config(config, manifest)
    if (config.get('all_frames') is not True or config['num_frames'] != 128 or
            len(config['objectives']) != 5 or config['steps'] != 20):
        raise ValueError('Campaign wrapper requires all five PGD-20 objectives on all 128 frames')
    count = config['campaign_expected_clips']
    directory = resolve_path(args.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {'status': 'running', 'all_frames': True, 'frames_per_clip': 128,
              'clips': count, 'expected_conditions': count * 6, 'stages': [],
              'started_at_utc': datetime.now(timezone.utc).isoformat()}
    target = directory / 'campaign_execution.json'
    commands = [
        ('study', [sys.executable, '-u', '-m', 'st4rtrack_pgd.run_component_study',
                   '--config', args.config, '--manifest', args.manifest,
                   '--run-dir', args.run_dir, '--fail-fast', *(['--resume'] if args.resume else [])]),
        ('artifact_audit', [sys.executable, 'scripts/audit_component_artifacts.py',
                            '--run-dir', args.run_dir, '--expected-clips', str(count),
                            '--expected-frames', '128', '--expected-steps', '20']),
        ('analysis', [sys.executable, 'scripts/analyze_component_study.py', '--run-dir', args.run_dir])]
    write_json(target, report)
    for label, command in commands:
        stage = {'stage': label, 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
        report['stages'].append(stage)
        write_json(target, report)
        print(f'CAMPAIGN stage: {label}', flush=True)
        stage_start = time.perf_counter()
        outcome = subprocess.run(command, cwd=ROOT, check=False)
        stage.update(status='complete' if outcome.returncode == 0 else 'failed',
                     exit_code=outcome.returncode, elapsed_seconds=time.perf_counter() - stage_start,
                     finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(target, report)
        if outcome.returncode:
            report.update(status='failed', failed_stage=label, elapsed_seconds=time.perf_counter() - started)
            write_json(target, report)
            return outcome.returncode
    audit = json.loads((directory / 'artifact_validation.json').read_text(encoding='utf-8'))
    analysis = json.loads((directory / 'analysis/study_analysis.json').read_text(encoding='utf-8'))
    if not audit.get('full_campaign_completed') or not analysis.get('all_raw_frames_campaign_completed'):
        report.update(status='failed', failed_stage='completion_scope', error='Whole declared all-frame campaign did not pass final verification')
        write_json(target, report)
        return 1
    report.update(status='complete', elapsed_seconds=time.perf_counter() - started,
                  completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  artifact_audit_passed=True, all_raw_frames_campaign_completed=True)
    write_json(target, report)
    print(f'CAMPAIGN verified: {count} clips, all 128 frames, {count * 6} conditions', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
