import os
import cv2
import dataclasses
import torch
import random
import numpy as np
import bop_dataset as data
import dataset_factory
import models
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from lib import warmup_lr
import config as bop_cfg


def worker_init_fn(*_):
    # each worker should only use one os thread
    # numpy/cv2/torch take advantage of multithreading by default; left
    # unchecked, N workers x all-cores oversubscribes the box and makes
    # the CPU-bound lazy quaternion labels slower than single-threaded.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    cv2.setNumThreads(0)
    torch.set_num_threads(1)

    # random seed
    np.random.seed(bop_cfg.RANDOM_SEED)


def main_worker(rank, world_size, args):
    # Per-run dataset overrides (mono/stereo + fusion ablation). Applied here
    # because spawn re-imports this module in each child, so parent-side
    # globals would not propagate.
    bop_cfg.set_runtime_overrides(
        args.dataset, stereo_mode=args.stereo_mode, fusion=args.fusion)

    # Fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    cv2.setNumThreads(0)

    if args.is_parallel:
        world_rank = int(os.environ['RANK']) if 'RANK' in os.environ else rank
        print('Running DDP on rank {:d}.'.format(world_rank))
        dist.init_process_group(
            'nccl', rank=world_rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        world_rank = rank

    train_dataset = dataset_factory.create_dataset(
        args.dataset, split='train')

    model_config = bop_cfg.ModelConfig.from_dataset_config(
        args.dataset, stereo_mode=args.stereo_mode, fusion=args.fusion)

    if args.is_parallel:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=train_sampler is None,
        pin_memory=True, num_workers=args.num_workers,
        drop_last=True, sampler=train_sampler)

    model = models.MRCNet(model_config).to(rank)
    if not args.no_compile:
        # torch.compile is lazy and can fail on boxes without a working
        # C/C++ toolchain; --no_compile runs eagerly instead.
        model = model.compile()

    if args.pretrained:
        checkpoint = torch.load(
            args.pretrained, map_location=torch.device(rank),
            weights_only=True)
        print('loading pre-trained model from {}'.format(args.pretrained))
        network = checkpoint['network']
        try:
            model.load_state_dict(network)
            print('  loaded strictly (all keys matched).')
        except RuntimeError as exc:
            # Cross-mode (mono<->stereo) or cross-dataset (different class
            # count) init: keep the shared weights, re-init the rest.
            print('  WARNING: strict load failed ({}); loading non-strict '
                  '(mismatched keys re-initialised).'.format(
                      type(exc).__name__))
            missing, unexpected = model.load_state_dict(network, strict=False)
            print('  missing={} unexpected={}'.format(
                sorted(missing)[:8], sorted(unexpected)[:8]))

    if args.is_parallel:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], static_graph=True,
            find_unused_parameters=False, gradient_as_bucket_view=True)

    def group_parameters(model):
        decay = set()
        no_decay = set()
        for mn, md in model.named_modules():
            for pn, p in md.named_parameters():
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(p)
                elif isinstance(md, (torch.nn.BatchNorm2d,
                                     torch.nn.GroupNorm)):
                    no_decay.add(p)
                elif isinstance(md, (torch.nn.Conv2d, torch.nn.Linear,
                                     models.CondConv)):
                    decay.add(p)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameter %s made it into both decay/no_decay sets!" % (
                str(inter_params), )
        assert len(param_dict) - len(union_params) == 0, \
            "parameter %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": list(decay), "weight_decay": bop_cfg.DECAY_WEIGHT},
            {"params": list(no_decay), "weight_decay": 0.0}]
        return optim_groups

    total_batches = len(train_loader) * args.n_epochs
    init_lr = bop_cfg.START_LR
    end_lr = bop_cfg.END_LR
    grouped_parameters = group_parameters(model)
    optimizer = torch.optim.AdamW(grouped_parameters, lr=init_lr)
    scheduler = warmup_lr.CosineAnnealingWarmupRestarts(
        optimizer, total_batches, warmup_steps=args.warmup_step,
        max_lr=init_lr, min_lr=end_lr)
    start_epoch = 0

    if world_rank == 0:
        train_writer = SummaryWriter(log_dir=os.path.join(
            args.log_dir, 'train'))
        if not os.path.exists(args.chkpt_dir):
            os.makedirs(args.chkpt_dir)

    train_step = 0

    def save_checkpoint(tag, epoch, step):
        """Write one checkpoint (rank 0 only); `tag` is the file name."""
        run_config = {
            'dataset': args.dataset,
            'stereo_mode': args.stereo_mode,
            'fusion': args.fusion,
            'n_refine_iters': args.n_refine_iters,
            'seed': args.seed,
            'pretrained': args.pretrained,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
        }
        network = (model.module.state_dict() if args.is_parallel
                   else model.state_dict())
        state_dict = {
            'network': network,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'model_config': dataclasses.asdict(model_config),
            'run_config': run_config,
            'last_epoch': epoch,
            'last_step': step,
        }
        model_path = os.path.join(args.chkpt_dir, tag)
        torch.save(state_dict, model_path)
        print('Saved model to {}'.format(model_path))

    gpu_keys = ['roi_obj_R', 'roi_obj_t', 'roi_mask', 'quat_bin',
                'roi_camK', 'obj_cls', 'fov']
    for epoch in range(start_epoch, args.n_epochs):
        if args.is_parallel:
            train_sampler.set_epoch(epoch)
        # Start main training loop
        with tqdm(desc='Epoch {:03d} - '.format(epoch), unit='iter',
                  total=len(train_loader), disable=rank != 0) as pbar:
            for it, batch in enumerate(train_loader):
                model.train()
                batch_image_roi = torch.concat([
                    batch['roi_image'].to(rank, non_blocking=True),
                    batch['bbox_map'].to(rank, non_blocking=True)], dim=1)

                targets = {key: batch[key].to(rank, non_blocking=True)
                           for key in gpu_keys}

                aux = {'obj_cls': targets['obj_cls'],
                       'fov': targets['fov'],
                       'intrinsics': targets['roi_camK']}
                # Stereo datasets add the right-eye flat tensors via collate;
                # the dataset mode guarantees they are all-or-none per batch.
                if 'roi_image_right' in batch:
                    batch_image_roi_right = torch.concat([
                        batch['roi_image_right'].to(rank, non_blocking=True),
                        batch['bbox_map_right'].to(rank, non_blocking=True)],
                        dim=1)
                    aux['inputs_right'] = batch_image_roi_right
                    aux['intrinsics_right'] = batch['roi_camK_right'].to(
                        rank, non_blocking=True)
                    aux['fov_right'] = batch['fov_right'].to(
                        rank, non_blocking=True)
                    aux['T_right_ref'] = batch['T_right_ref'].to(
                        rank, non_blocking=True)

                predictions = model(
                    batch_image_roi, aux, targets,
                    n_refine_iters=args.n_refine_iters,
                    grad_ckpt=args.refine_grad_ckpt)
                loss_dict = predictions['losses']
                loss = sum(loss_dict.values())
                loss.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr = optimizer.param_groups[0]['lr']

                # Log training progress
                if world_rank == 0:
                    loss_val = sum([v for k, v in loss_dict.items()])
                    train_writer.add_scalar('Loss', loss_val, train_step)
                    train_writer.add_scalar('Learning_rate', lr, train_step)
                    train_writer.add_scalar(
                        'Weight_decay', bop_cfg.DECAY_WEIGHT, train_step)
                    for name, val in loss_dict.items():
                        train_writer.add_scalar(
                            'Loss/' + name, val, train_step)

                pbar.update()
                pbar.set_postfix(loss=loss.item())
                train_step += 1
                scheduler.step()
                if world_rank == 0 and args.save_every_steps > 0 \
                        and train_step % args.save_every_steps == 0:
                    save_checkpoint(
                        'step_{:08d}.pth'.format(train_step), epoch, train_step)
                if args.max_steps and train_step >= args.max_steps:
                    break

            # Save models; this only needs to be done in one replica
            if world_rank == 0 and args.save_every_epochs > 0 \
                    and epoch % args.save_every_epochs == 0:
                save_checkpoint(
                    'epoch_{:03d}.pth'.format(epoch), epoch, train_step)

            if args.max_steps and train_step >= args.max_steps:
                print('Reached --max_steps={}; stopping.'.format(
                    args.max_steps))
                break

    if args.is_parallel:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size for data')
    parser.add_argument('--is_parallel', action='store_true',
                        help='whether to enable multi-gpu training')
    parser.add_argument('--n_epochs', type=int, default=120,
                        help='total number of epochs')
    parser.add_argument('--max_steps', type=int, default=0,
                        help='debug: stop after this many optimizer steps '
                             '(0 = run all epochs)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for parallel data loading')
    parser.add_argument('--dataset', type=str, default='stereobj-1m',
                        help='name of the dataset for training and validation')
    parser.add_argument('--stereo_mode', type=str, default=None,
                        choices=['mono', 'stereo'],
                        help='override DATASET_CONFIG stereo_mode for this run '
                             '(default: the dataset config value)')
    parser.add_argument('--fusion', type=str, default=None,
                        choices=list(bop_cfg.FUSION_MODES),
                        help='stereo fusion operation (only meaningful when '
                             'stereo_mode=stereo; default cross_view)')
    parser.add_argument('--seed', type=int, default=bop_cfg.RANDOM_SEED,
                        help='random seed (single-seed study by default)')
    parser.add_argument('--no_compile', action='store_true',
                        help='skip torch.compile and run the model eagerly')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='override config.DATASET_ROOT (default: env '
                             'MRC_DATASET_ROOT / DATASET_ROOT, else config)')
    parser.add_argument('--eval_root', type=str, default=None,
                        help='override config.EVAL_ROOT (default: env '
                             'MRC_EVAL_ROOT / EVAL_ROOT, else DATASET_ROOT)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory holding tensorboard logs')
    parser.add_argument('--chkpt_dir', type=str, default='./checkpoints',
                        help='directory holding all model checkpoints')
    parser.add_argument('--warmup_step', type=int, default=1000,
                        help='number of steps to warmup learning rate')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='optional checkpoint to initialise the network '
                             'from (fresh init when omitted)')
    parser.add_argument('--n_refine_iters', type=int, default=1,
                        help='number of weight-shared iterative refinement '
                             'passes (1 = single-pass forward)')
    parser.add_argument('--refine_grad_ckpt', action='store_true',
                        help='gradient-checkpoint the refinement synthetic '
                             'branch (overrides config.REFINE_GRAD_CKPT)')
    parser.add_argument('--save_every_steps', type=int, default=0,
                        help='save a checkpoint every N optimizer steps '
                             '(0 = disabled; useful for long epochs where a '
                             'job may time out before the next epoch)')
    parser.add_argument('--save_every_epochs', type=int, default=10,
                        help='save a checkpoint at the end of every N epochs '
                             '(0 = disabled; default 10)')
    args = parser.parse_args()

    # Data/eval roots: CLI > env (MRC_DATASET_ROOT / MRC_EVAL_ROOT) > config.
    # Set before spawn so the exported env reaches the worker processes.
    bop_cfg.set_roots(args.dataset_root, args.eval_root)

    # config.REFINE_GRAD_CKPT is the default; the CLI flag only turns it on.
    args.refine_grad_ckpt = args.refine_grad_ckpt or bop_cfg.REFINE_GRAD_CKPT

    if args.is_parallel:
        # Training on single node
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            world_size = torch.cuda.device_count()
            ngpus = torch.cuda.device_count()
            args.batch_size = int(args.batch_size / world_size)
            args.num_workers = int(args.num_workers / world_size)
            mp.spawn(
                main_worker, args=(world_size, args), nprocs=ngpus, join=True)

        # Training on multiple nodes
        else:
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            args.batch_size = int(args.batch_size / world_size)
            args.num_workers = int(args.num_workers / world_size)
            main_worker(local_rank, world_size, args)
    else:
        main_worker(0, 1, args)
