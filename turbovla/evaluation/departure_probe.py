"""Read-only departure/false-start probe on expert observations, NOT rollout success.

Compare candidates on the same split, selection and thresholds. Speed ablations
change RAW state[0:3] only; they are diagnostic interventions, not recovery labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from turbovla.data.interventions import departure_events


def event_result(expert, prediction, kind):
    expert, prediction = np.asarray(expert), np.asarray(prediction)
    if expert.shape != prediction.shape or expert.ndim != 2 or expert.shape[1] != 23:
        raise ValueError('Expected aligned [T,23] predictions and expert commands')
    if kind == 'move':
        strength = lambda a: np.linalg.norm(a[:, :2], axis=1)
    elif kind == 'turn':
        strength = lambda a: np.abs(a[:, 2])
    else:
        raise ValueError('Separate movement and yaw events')
    starts = np.flatnonzero(strength(expert) > .05)
    if not len(starts):
        return None
    first = int(starts[0])
    hits = np.flatnonzero(strength(prediction)[first:first+16] > .05)
    delay = int(hits[0]) if len(hits) else None
    if delay is None:
        direction = None
    else:
        p = prediction[first+delay]
        reference = expert[first]  # Direction of the actual departure command.
        direction = bool(np.dot(p[:2], reference[:2]) > 0) if kind == 'move' else bool(p[2]*reference[2] > 0)
    return dict(kind=kind, missed=delay is None, delay_frames=delay,
                delay_seconds=delay/30 if delay is not None else None, direction_agrees=direction,
                expert_first_index=first)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--max-events', type=int, default=40)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dinov3')
    parser.add_argument('--bert')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh report path')
    if args.max_events < 1:
        raise ValueError('Positive max-events required')
    import torch
    from turbovla.data.dataset import load_episode
    from turbovla.evaluation.policy import BehaviorPolicy
    metadata = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    ids = metadata[args.split+'_episode_ids']
    all_ids = metadata['train_episode_ids'] + metadata['validation_episode_ids']
    sources = dict(zip(all_ids, metadata.get('data_sources', [])))
    del metadata
    policy = BehaviorPolicy.from_checkpoint(args.checkpoint, args.device, 1, dinov3=args.dinov3, bert=args.bert)
    rows, quiet_rows, ablations = [], [], []
    event_count = 0
    for eid in ids:
        episode = load_episode(args.cache / f'episode_{eid:06d}')
        try:
            if (int(episode['episode_id']) != eid or int(episode['task_id']) != 0 or
                    episode['instruction'] != 'turning_on_radio' or
                    eid not in sources or episode.get('source') != sources[eid]):
                raise ValueError(f'Cache identity/provenance differs: episode {eid}')
            actions = episode['actions']

            def predict(frame, zero_velocity=False):
                state = np.array(episode['states'][frame], copy=True)
                if zero_velocity:
                    state[:3] = 0
                return policy.infer(episode['rgb'][frame], state, episode['instruction'])['action']

            # One fully stationary 16-frame window per episode; no false labels
            # from actually moving expert windows. Selection is expert-only.
            rest = (np.linalg.norm(actions[:, :2], axis=1) <= .01) & (np.abs(actions[:, 2]) <= .02)
            for start in range(0, max(0, len(actions)-15), 16):
                if rest[start:start+16].all():
                    policy.reset()
                    pred = np.stack([predict(t) for t in range(start, start+16)])
                    quiet_rows.append(dict(episode_id=eid, start=start,
                        false_move=bool((np.linalg.norm(pred[:, :2], axis=1) > .05).any()),
                        false_turn=bool((np.abs(pred[:, 2]) > .05).any())))
                    break
            for event in departure_events(actions):
                if event_count >= args.max_events:
                    break
                # Covers ramp to threshold plus 15-frame response window.
                start, stop = event['frame'], min(len(actions), event['frame']+31)
                policy.reset()
                pred = np.stack([predict(t) for t in range(start, stop)])
                for kind in ('move', 'turn'):
                    result = event_result(actions[start:stop], pred, kind)
                    if result is not None:
                        rows.append(dict(episode_id=eid, onset_frame=start, **result))
                event_count += 1
            moving = np.flatnonzero(np.linalg.norm(actions[:, :2], axis=1) > .2)
            if len(moving):
                t = int(moving[min(5, len(moving)-1)])
                policy.reset()
                clean = float(np.linalg.norm(predict(t)[:2]))
                masked = float(np.linalg.norm(predict(t, True)[:2]))
                ablations.append(dict(episode_id=eid, frame=t, clean_xy=clean, zero_velocity_xy=masked,
                                      ratio=masked/clean if clean > 1e-6 else None))
        finally:
            for key in ('rgb', 'states', 'actions', 'depth'):
                mmap = getattr(episode.get(key), '_mmap', None)
                if mmap is not None:
                    mmap.close()
    summary = {}
    for kind in ('move', 'turn'):
        group = [row for row in rows if row['kind'] == kind]
        delays = [row['delay_frames'] for row in group if not row['missed']]
        directions = [row['direction_agrees'] for row in group if row['direction_agrees'] is not None]
        summary[kind] = dict(events=len(group), miss_rate=float(np.mean([r['missed'] for r in group])) if group else None,
                             median_delay_frames=float(np.median(delays)) if delays else None,
                             direction_agreement=float(np.mean(directions)) if directions else None)
    for kind in ('move', 'turn'):
        summary['false_'+kind+'_rate'] = float(np.mean([r['false_'+kind] for r in quiet_rows])) if quiet_rows else None
    report = dict(checkpoint=policy.deployment_metadata, split=args.split, max_events=args.max_events,
                  selection='episode order; event onset requires 15 rest frames; same expert-only selection for all candidates',
                  threshold=.05, response_frames=15, observation_mode='expert observations, first predicted action; not closed-loop',
                  closed_loop_success=None, summary=summary, events=rows, quiet_windows=quiet_rows, velocity_ablation=ablations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
