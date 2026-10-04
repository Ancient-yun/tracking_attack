"""Compare optimized FP32 two-frame VJPs with saved original 64-frame baselines."""
from __future__ import annotations
import argparse
import json
import statistics
import time
import numpy as np
import torch
from st4rtrack_pgd.common import resolve_path, write_json, sha256
from st4rtrack_pgd.component_attack import component_loss_and_gradient
from st4rtrack_pgd.component_forward import NativeComponentForward, gradient_repeat_diagnostics, load_component_sequence
from st4rtrack_pgd.st4rtrack_forward import St4RTrackForward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--frames', type=int, default=64)
    args = parser.parse_args()
    directory, original_dir = resolve_path(args.output_dir), resolve_path(args.baseline_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = {'passed': False, 'scope': 'FP32 unchanged weights/GT/loss/epsilon; paired compact outputs and real RGB gradient comparison', 'frames': args.frames}
    write_json(directory / 'optimized_report.json', report)
    baseline_report = json.loads((original_dir / 'baseline_report.json').read_text())
    if not baseline_report['passed'] or len(baseline_report['timings']) != 2:
        raise ValueError('Requires two completed original baseline repeats')
    originals = [torch.load(original_dir / f'baseline_repeat{i}.pt', weights_only=True, map_location='cpu') for i in range(2)]
    torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print('OPTIMIZED: loading identical official checkpoint and PO sequence', flush=True)
    sequence, context = load_component_sequence(resolve_path('data/worldtrack_release/po_mini/cab_e_3rd_13.npz'), num_frames=args.frames)
    sequence = sequence.to('cuda')
    base = St4RTrackForward.from_checkpoint(resolve_path('assets/checkpoints/St4RTrack_Seqmode_reweightMax5.pth'), sequence.query_xy, device='cuda')
    forward = NativeComponentForward(base, context, optimize_runtime=True)
    keys = ('native_tracks', 'native_track_valid', 'training_dynamic', 'tracks', 'valid', 'reconstruction_gt_norm', 'reconstruction_valid_counts')
    targets = {key: context[key].to('cuda') for key in keys}
    with torch.no_grad():
        forward.pair_fn(sequence.rgb, 0)
    torch.cuda.synchronize()
    timings, results = [], []
    for repeat in range(2):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = component_loss_and_gradient(forward.pair_fn, sequence.rgb, targets, 'joint_training', pair_frames_fn=forward.pair_frames_fn)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        saved = {'gradient': result.gradient.cpu(), 'outputs': {key: value.cpu() for key, value in result.outputs.items()}, 'loss': result.loss}
        torch.save(saved, directory / f'optimized_repeat{repeat}.pt')
        results.append(saved)
        timings.append({'seconds': seconds, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'loss': result.loss})
        print(f'OPTIMIZED repeat {repeat}: {seconds:.3f}s, loss={result.loss:.8g}', flush=True)
        del result
    output_errors = {}
    for name in originals[0]['outputs']:
        errors = []
        for old in originals:
            for new in results:
                torch.testing.assert_close(new['outputs'][name], old['outputs'][name], rtol=1e-5, atol=1e-6)
                errors.append(float((new['outputs'][name] - old['outputs'][name]).abs().max()))
        output_errors[name] = max(errors)
    for old in originals:
        for new in results:
            np.testing.assert_allclose(new['loss'], old['loss'], rtol=2e-5, atol=2e-5)
    diagnostics = gradient_repeat_diagnostics([entry['gradient'] for entry in results], [entry['gradient'] for entry in originals], relative_l2_cap=1e-3)
    diagnostics['mode'] = 'independent_original_recompute_twice_vs_optimized_local_recompute_twice'
    diagnostics['reference'] = 'saved unchanged original FP32 64-frame recompute gradients'
    diagnostics['native_repeat_label'] = 'optimized repeat difference'
    diagnostics['official_repeat_label'] = 'original repeat difference'
    # The shared numerical diagnostic helper also serves same-graph oracle
    # tests. These graph-retention fields do not apply to saved gradients.
    diagnostics.pop('backward_calls', None)
    diagnostics.pop('retain_graph', None)
    if not diagnostics['passed']:
        report.update({'timings': timings, 'gradient_parity': diagnostics, 'output_max_abs_errors': output_errors})
        write_json(directory / 'optimized_report.json', report)
        raise AssertionError('Optimized RGB gradient does not match baseline repeat-noise gate')
    print('OPTIMIZED: original 64-frame compact values and RGB gradients passed; profiling one gradient', flush=True)
    # Profiling an entire 64-frame pass retains millions of profiler events.
    # Bound event retention to two frames; the timed/verified workload above
    # remains the complete 64-frame objective, and excludes instrumentation.
    short_sequence, short_context = load_component_sequence(resolve_path('data/worldtrack_release/po_mini/cab_e_3rd_13.npz'), num_frames=2)
    short_sequence = short_sequence.to('cuda')
    short_base = St4RTrackForward(base.model, short_sequence.query_xy, checkpoint=base.checkpoint)
    short_forward = NativeComponentForward(short_base, short_context, optimize_runtime=True)
    short_targets = {key: short_context[key].to('cuda') for key in keys}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        result = component_loss_and_gradient(short_forward.pair_fn, short_sequence.rgb, short_targets, 'joint_training', pair_frames_fn=short_forward.pair_frames_fn)
        torch.cuda.synchronize()
    rows = [{'name': event.key, 'count': event.count, 'self_cpu_time_us': event.self_cpu_time_total, 'self_device_time_us': event.self_device_time_total} for event in prof.key_averages()]
    (directory / 'profile_cpu.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=35), encoding='utf-8')
    (directory / 'profile_cuda.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35), encoding='utf-8')
    write_json(directory / 'profile_events.json', rows)
    old_seconds = statistics.median(entry['seconds'] for entry in baseline_report['timings'])
    new_seconds = statistics.median(entry['seconds'] for entry in timings)
    report.update({'passed': True, 'timings': timings, 'baseline_median_seconds': old_seconds,
        'optimized_median_seconds': new_seconds, 'speedup': old_seconds / new_seconds,
        'time_reduction_percent': 100 * (1 - new_seconds / old_seconds),
        'timing_scope': 'CUDA-synchronized 64-frame joint loss gradient; model load, save, metrics and profiler excluded',
        'gradient_parity': diagnostics, 'output_max_abs_errors': output_errors,
        'baseline_source_report_sha256': sha256(original_dir / 'baseline_report.json'),
        'model': forward.metadata, 'profiler_scope_frames': 2, 'precision': 'float32', 'allow_tf32': False, 'autocast': False})
    write_json(directory / 'optimized_report.json', report)
    print(f"OPTIMIZED verified speedup={report['speedup']:.3f}x, time reduction={report['time_reduction_percent']:.1f}%", flush=True)


if __name__ == '__main__':
    main()
