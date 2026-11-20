"""Independent pilots from the preserved 100k policy; no training unless invoked.

Use --check-only first. --steps counts NEW updates with a fresh optimizer in every
arm, including control. Existing caches, normalization and episode split are fixed.
Run from an isolated git archive to leave the rolled-back Astrago checkout alone.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = '/input/Technyong_TurboVLA/task0-staged-100k-20261118-134121/checkpoints/step_100000.pt'
SOURCE_SHA256 = '893af2ed4b44bd89a6c1ad105db628fe99d64542a8e69c0deca62febbeff9359'
MODES = {
    'control': (0., 0., 'none'),
    'transition': (.25, 0., 'none'),
    'velocity': (0., .2, 'none'),
    'visual': (0., 0., 'mild'),
    'combined': (.25, .2, 'mild'),
}


def pilot_command(args, checkpoint, help_text):
    from scripts.behavior_2026.train_task0_long import resume_settings, training_command
    settings = resume_settings(checkpoint, int(checkpoint['step']) + args.steps)
    settings['lr'] = args.lr
    command = training_command(args, checkpoint, settings, help_text)
    if '--finetune' not in help_text:
        raise ValueError('Use the new isolated intervention code tree, not the old 100k tree')
    command[command.index('--resume')] = '--finetune'
    command[command.index('--decision-pretrain-steps') + 1] = '0'
    fraction, probability, visual = MODES[args.mode]
    command += ['--transition-fraction', str(fraction), '--velocity-mask-probability', str(probability),
                '--visual-augmentation', visual, '--transition-window', '15', '--transition-quiet-frames', '15']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=list(MODES))
    parser.add_argument('--checkpoint', type=Path, default=Path(SOURCE))
    parser.add_argument('--expected-sha256', default=SOURCE_SHA256)
    parser.add_argument('--cache', type=Path, default=Path('/input/Technyong_TurboVLA/caches/task0-rgb-224'))
    parser.add_argument('--out', type=Path, required=True, help='Fresh candidate checkpoint directory')
    parser.add_argument('--steps', type=int, default=10000, help='New optimizer updates, not total ancestral steps')
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dinov3', type=Path)
    parser.add_argument('--bert', type=Path)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.save_every) < 1 or not 0 < args.lr < float('inf'):
        raise ValueError('Positive finite steps/save interval/learning rate required')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Use a fresh output directory; old runs are preserved')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Single-process Python only')
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project))
    from turbovla.evaluation.diagnostics import file_sha256
    digest = file_sha256(args.checkpoint)
    if digest != args.expected_sha256.lower():
        raise ValueError('Source SHA differs from the report; no training started')
    import torch
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    from scripts.behavior_2026.train_task0_long import check_cache, resume_settings
    settings = resume_settings(checkpoint, int(checkpoint['step']) + args.steps)
    from turbovla.models.configuration import TurboVLAConfig
    TurboVLAConfig.from_mapping(checkpoint['model_config'])
    for directory in (args.dinov3 or checkpoint['model_config']['vision']['model_name_or_path'],
                      args.bert or checkpoint['model_config']['text']['model_name_or_path']):
        if not Path(directory).is_dir():
            raise FileNotFoundError(f'Local pretrained directory missing: {directory}')
    ids = checkpoint['train_episode_ids'] + checkpoint['validation_episode_ids']
    sources = checkpoint.get('data_sources', [])
    if len(sources) != len(ids) or any(not source for source in sources):
        raise ValueError('Missing per-episode dataset provenance')
    from turbovla.data.dataset import load_episode
    for eid, source in zip(ids, sources):
        folder = args.cache / f'episode_{eid:06d}'
        if not check_cache(folder, eid, source, 'turning_on_radio', load_episode):
            raise FileNotFoundError(f'Expected existing 100k training cache: {folder}')
    env = {**os.environ, 'PYTHONPATH': str(project) + os.pathsep + os.environ.get('PYTHONPATH', ''),
           'PYTHONUNBUFFERED': '1'}
    help_text = subprocess.run([sys.executable, '-m', 'turbovla.training.trainer', '--help'],
                               cwd=project, env=env, capture_output=True, text=True, check=True).stdout
    command = pilot_command(args, checkpoint, help_text)
    fraction, probability, visual = MODES[args.mode]
    plan = dict(mode=args.mode, source=str(args.checkpoint), source_sha256=digest,
                code_commit=os.environ.get('TURBOVLA_CODE_COMMIT'),
                source_step=int(checkpoint['step']), new_updates=args.steps, optimizer='fresh', lr=args.lr,
                seed=settings['seed'], batch_size=settings['batch_size'], transition_fraction=fraction,
                velocity_mask_probability=probability, velocity_mask_space='raw_zero_before_normalization',
                visual_augmentation=visual, transition_window_frames=15, preceding_rest_frames=15,
                train_episode_ids=checkpoint['train_episode_ids'], validation_episode_ids=checkpoint['validation_episode_ids'],
                out=str(args.out), closed_loop_success=None, expert_replay_verified=None)
    print(json.dumps({'plan': plan}, ensure_ascii=False), flush=True)
    if args.check_only:
        print('CHECK OK: caches inspected; no model updated, no output created', flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training started')
    if args.wandb_mode == 'online':
        import wandb
        wandb.login()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.out.parent / 'source_step_100000.pt'
    if not snapshot.exists():
        shutil.copyfile(args.checkpoint, snapshot)
    if file_sha256(snapshot) != digest:
        raise ValueError('Existing source snapshot differs; choose a fresh candidate directory')
    command[command.index('--finetune') + 1] = str(snapshot)
    with (args.out.parent / 'intervention-plan.json').open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=False)
    env.setdefault('WANDB_NAME', args.out.parent.name)
    del checkpoint
    subprocess.run(command, cwd=project, env=env, check=True)


if __name__ == '__main__':
    main()
