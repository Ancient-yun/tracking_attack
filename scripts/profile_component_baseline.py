"""Bounded baseline runtime/CPU-CUDA synchronization profile, no study launch."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from st4rtrack_pgd.common import resolve_path, write_json
from st4rtrack_pgd.component_attack import component_loss_and_gradient
from st4rtrack_pgd.component_forward import NativeComponentForward, load_component_sequence
from st4rtrack_pgd.st4rtrack_forward import St4RTrackForward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--frames', type=int, default=64)
    args = parser.parse_args()
    directory = resolve_path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print('BASELINE: loading official checkpoint and fixed PO sequence', flush=True)
    sequence, context = load_component_sequence(resolve_path('data/worldtrack_release/po_mini/cab_e_3rd_13.npz'), num_frames=args.frames)
    sequence = sequence.to('cuda')
    base = St4RTrackForward.from_checkpoint(resolve_path('assets/checkpoints/St4RTrack_Seqmode_reweightMax5.pth'), sequence.query_xy, device='cuda')
    forward = NativeComponentForward(base, context)
    keys = ('native_tracks', 'native_track_valid', 'training_dynamic', 'tracks', 'valid', 'reconstruction_gt_norm', 'reconstruction_valid_counts')
    targets = {key: context[key].to('cuda') for key in keys}
    with torch.no_grad():
        forward.pair_fn(sequence.rgb, 0)
    torch.cuda.synchronize()
    timings = []
    for repeat in range(2):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = component_loss_and_gradient(forward.pair_fn, sequence.rgb, targets, 'joint_training')
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        timings.append({'seconds': seconds, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'loss': result.loss})
        torch.save({'gradient': result.gradient.cpu(), 'outputs': {key: val.cpu() for key, val in result.outputs.items()}, 'loss': result.loss}, directory / f'baseline_repeat{repeat}.pt')
        print(f'BASELINE repeat {repeat}: {seconds:.3f}s, loss={result.loss:.8g}', flush=True)
        del result
    report = {'passed': True, 'scope': f'{args.frames} frames, unchanged FP32 official checkpoint, joint loss recompute VJP', 'timings': timings,
        'allow_tf32': False, 'autocast': False, 'precision': 'float32', 'model': base.metadata,
        'profiling': {'completed': False, 'frames': 2}}
    write_json(directory / 'baseline_report.json', report)
    print('BASELINE: profiling a bounded two-frame recompute gradient', flush=True)
    short_sequence, short_context = load_component_sequence(resolve_path('data/worldtrack_release/po_mini/cab_e_3rd_13.npz'), num_frames=2)
    short_sequence = short_sequence.to('cuda')
    short_forward = NativeComponentForward(St4RTrackForward(base.model, short_sequence.query_xy, checkpoint=base.checkpoint), short_context)
    short_targets = {key: short_context[key].to('cuda') for key in keys}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        result = component_loss_and_gradient(short_forward.pair_fn, short_sequence.rgb, short_targets, 'joint_training')
        torch.cuda.synchronize()
    rows = [{'name': event.key, 'count': event.count, 'self_cpu_time_us': event.self_cpu_time_total,
             'self_device_time_us': event.self_device_time_total} for event in prof.key_averages()]
    (directory / 'profile_cpu.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=35), encoding='utf-8')
    (directory / 'profile_cuda.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35), encoding='utf-8')
    write_json(directory / 'profile_events.json', rows)
    report['profiling']['completed'] = True
    write_json(directory / 'baseline_report.json', report)
    print('BASELINE profile completed', flush=True)


if __name__ == '__main__':
    main()
