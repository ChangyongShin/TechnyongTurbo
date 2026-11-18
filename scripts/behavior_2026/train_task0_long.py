"""Resume the original RGB task-0 baseline, preparing persistent caches first.

This launcher does not modify model/trainer code or enable staged objectives.
--steps is the TOTAL optimizer step, including the checkpoint's existing steps.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def baseline_settings(checkpoint, steps):
    if 'optimizer' not in checkpoint:
        raise ValueError('Optimizer state is missing; this launcher requires a baseline optimizer resume')
    start = int(checkpoint['step'])
    if steps <= start:
        raise ValueError('--steps must exceed the existing checkpoint step')
    config, training = checkpoint['model_config'], checkpoint['training']
    if config['action'].get('decision_conditioning', False):
        raise ValueError('Use the original task0-batch32-epoch1 checkpoint, not the staged active_policy model')
    for flag in ('task_balanced', 'phase_balanced', 'group_balanced_loss'):
        if training.get(flag, False):
            raise ValueError(f'Not the original baseline: {flag}')
    if training.get('temporal_weight', 0) or training.get('decision_pretrain_steps', 0):
        raise ValueError('Auxiliary objectives differ from the original baseline')
    if checkpoint.get('modality', 'rgb') != 'rgb':
        raise ValueError('This launcher expects the original RGB baseline')
    if training.get('augmentation', 'none') != 'none':
        raise ValueError('Choose the unaugmented task0-batch32-epoch1 baseline')
    if (config['action']['action_dim'], config['action']['state_dim'], config['vision']['num_views']) != (23, 61, 3):
        raise ValueError('Expected the R1Pro 23-action / 61-state / 3-camera contract')
    train, val = checkpoint['train_episode_ids'], checkpoint['validation_episode_ids']
    if len(train) != 180 or len(val) != 20 or len(set(train + val)) != 200 or set(train + val) != set(range(200)):
        raise ValueError('Expected the original disjoint 180/20 task0 split')
    if not config['vision'].get('frozen', True):
        raise ValueError('This launcher preserves the original frozen-vision baseline')
    return {'start_step': start, 'target_step': steps, 'additional_steps': steps - start,
            'batch_size': int(training['batch_size']), 'lr': float(training['lr']),
            'grad_accumulation': int(training['grad_accumulation']), 'workers': int(training.get('workers', 0)),
            'seed': int(checkpoint['seed'])}


def training_command(args, checkpoint, settings, help_text):
    config = checkpoint['model_config']
    command = [sys.executable, '-u', '-m', 'turbovla.training.trainer',
               '--train', *[str(args.cache / f'episode_{eid:06d}') for eid in checkpoint['train_episode_ids']],
               '--validation', *[str(args.cache / f'episode_{eid:06d}') for eid in checkpoint['validation_episode_ids']],
               '--dinov3', str(args.dinov3 or config['vision']['model_name_or_path']),
               '--bert', str(args.bert or config['text']['model_name_or_path']),
               '--resume', str(args.checkpoint), '--out', str(args.out), '--steps', str(args.steps),
               '--batch-size', str(settings['batch_size']), '--lr', str(settings['lr']),
               '--grad-accumulation', str(settings['grad_accumulation']), '--workers', str(settings['workers']),
               '--seed', str(settings['seed']), '--horizon', str(config['action']['horizon']),
               '--image-size', str(checkpoint['preprocessing']['size']),
               '--vision-precision', config['vision']['compute_precision'],
               '--save-every', str(args.save_every), '--device', 'cuda', '--wandb-mode', args.wandb_mode]
    if '--augmentation' in help_text:
        command += ['--augmentation', 'none']
    return command


def check_cache(folder, eid, source, instruction, load_episode):
    if not folder.exists():
        return False
    episode = load_episode(folder)  # Incomplete caches cause an error; never delete them.
    try:
        if (int(episode['episode_id']) != eid or int(episode['task_id']) != 0 or
                episode['instruction'] != instruction or episode.get('source') != source or
                episode['modality'] != 'rgb' or tuple(episode['rgb'].shape[2:4]) != (224, 224)):
            raise ValueError(f'Existing cache identity differs: {folder}')
    finally:
        for key in ('rgb', 'states', 'actions'):
            episode[key]._mmap.close()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('/workspace/TechnyongTurbo/outputs/task0-batch32-epoch1/step_12048.pt'))
    parser.add_argument('--data-root', type=Path, default=Path('/workspace/data/behavior-2026-rgb'))
    parser.add_argument('--cache', type=Path, default=Path('/input/Technyong_TurboVLA/caches/task0-rgb-224'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--dinov3', type=Path)
    parser.add_argument('--bert', type=Path)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--check-only', action='store_true', help='Validate checkpoint/source/settings without preparing caches or training')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use this launcher with single-process Python, not torchrun')
    if args.save_every < 1:
        raise ValueError('--save-every must be positive')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Use a fresh --out directory; existing runs are preserved')
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project))
    env = {**os.environ, 'PYTHONPATH': str(project) + os.pathsep + os.environ.get('PYTHONPATH', ''), 'PYTHONUNBUFFERED': '1'}
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'Original baseline checkpoint is missing: {args.checkpoint}')
    import torch
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    settings = baseline_settings(checkpoint, args.steps)
    from turbovla.models.configuration import TurboVLAConfig
    config = checkpoint['model_config']
    expected = TurboVLAConfig.from_mapping({
        'text': {'model_name_or_path': str(args.bert or config['text']['model_name_or_path']), 'local_files_only': True},
        'vision': {'model_name_or_path': str(args.dinov3 or config['vision']['model_name_or_path']),
                   'num_views': 3, 'input_channels': 3, 'image_size': checkpoint['preprocessing']['size'],
                   'frozen': True, 'local_files_only': True, 'compute_precision': config['vision']['compute_precision']},
        'action': {'action_dim': 23, 'state_dim': 61, 'horizon': config['action']['horizon']}})
    normalized = TurboVLAConfig.from_mapping(config).to_dict()
    normalized['text']['model_name_or_path'] = expected.text.model_name_or_path
    normalized['vision']['model_name_or_path'] = expected.vision.model_name_or_path
    if normalized != expected.to_dict():
        raise ValueError('Current trainer architecture defaults do not match the baseline; nothing trained')
    for directory in (expected.vision.model_name_or_path, expected.text.model_name_or_path):
        if not Path(directory).is_dir():
            raise FileNotFoundError(f'Local pretrained model directory is missing: {directory}')
    source = json.loads((args.data_root / 'turbovla_source.json').read_text(encoding='utf-8'))
    if not checkpoint.get('data_sources') or any(item != source for item in checkpoint['data_sources']):
        raise ValueError('Original dataset provenance does not match the checkpoint')
    from turbovla.data.tasks import load_tasks
    instruction = load_tasks(args.data_root)[0]
    if instruction != 'turning_on_radio':
        raise ValueError(f'Unexpected task0 instruction: {instruction!r}')
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    settings.update(checkpoint=str(args.checkpoint), checkpoint_sha256=digest.hexdigest(), cache=str(args.cache), out=str(args.out),
                    source=source, instruction=instruction, save_every=args.save_every,
                    model_changes=False, optimizer_resumed=True)
    print(json.dumps({'plan': settings}, ensure_ascii=False), flush=True)
    help_text = subprocess.run([sys.executable, '-m', 'turbovla.training.trainer', '--help'],
                               cwd=project, env=env, capture_output=True, text=True, check=True).stdout
    command = training_command(args, checkpoint, settings, help_text)
    if args.check_only:
        print('CHECK OK: caches not prepared and training not started', flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable; training not started')
    if args.wandb_mode == 'online':
        import wandb
        wandb.login()  # Fail before expensive preparation if this terminal has no authentication.
    from turbovla.data.dataset import load_episode, training_normalizer
    missing = []
    all_ids = checkpoint['train_episode_ids'] + checkpoint['validation_episode_ids']
    for eid in all_ids:
        folder = args.cache / f'episode_{eid:06d}'
        if not check_cache(folder, eid, source, instruction, load_episode):
            missing.append(eid)
    if missing:
        print(json.dumps({'prepare_missing_episodes': missing}), flush=True)
        subprocess.run([sys.executable, '-u', '-m', 'turbovla.data.prepare', '--root', str(args.data_root),
                        '--episodes', *map(str, sorted(missing)), '--out', str(args.cache),
                        '--image-size', '224', '--format', 'mmap'], cwd=project, env=env, check=True)
        for eid in missing:
            if not check_cache(args.cache / f'episode_{eid:06d}', eid, source, instruction, load_episode):
                raise FileNotFoundError(f'Preparation did not produce episode {eid}')
    train_paths = [args.cache / f'episode_{eid:06d}' for eid in checkpoint['train_episode_ids']]
    if training_normalizer(train_paths).to_dict() != checkpoint['normalization']:
        raise ValueError('Prepared TRAIN normalization differs from the original; training not started')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print('READY: caches checked; resuming original policy and optimizer', flush=True)
    del checkpoint
    subprocess.run(command, cwd=project, env=env, check=True)


if __name__ == '__main__':
    main()
