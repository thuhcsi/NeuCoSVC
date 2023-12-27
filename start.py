import argparse
import logging
import os
import sys
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.tools import ConfigWrapper
from dataset.dataset import SVCDataset

from modules.FastSVC import SVCNN
from modules.discriminator import MelGANMultiScaleDiscriminator

from optimizers.scheduler import StepLRScheduler
from loss.adversarial_loss import GeneratorAdversarialLoss
from loss.adversarial_loss import DiscriminatorAdversarialLoss
from loss.stft_loss import MultiResolutionSTFTLoss
from trainer import Trainer

def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train the FastSVC model."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="dataset root path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="configuration file path.",
    )
    parser.add_argument(
        "--cp_path",
        required=True,
        type=str,
        nargs="?",
        help='checkpoint file path.',
    )
    parser.add_argument(
        "--pretrain",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to load pretrained params. (default="")',
    )
    parser.add_argument(
        "--resume",
        default=False,
        type=bool,
        nargs="?",
        help='whether to resume training from a certain checkpoint.',
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="random seed.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    local_rank = 0
    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        args.world_size = torch.cuda.device_count()
        args.distributed = args.world_size > 1
        if args.distributed:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            print('Using multi-GPUs for training. n_GPU=%d.' %(args.world_size))
            torch.distributed.init_process_group(backend="nccl")

    # random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # suppress logging for distributed training
    if local_rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load and save config
    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))
    config.training_config.rank = local_rank
    config.training_config.distributed = args.distributed
    config.interval_config.out_dir = args.cp_path

    # get dataset
    train_set = SVCDataset(args.data_root, config.data_config.n_samples, config.data_config.sampling_rate, config.data_config.hop_size, 'train')
    valid_set = SVCDataset(args.data_root, config.data_config.n_samples, config.data_config.sampling_rate, config.data_config.hop_size, 'valid')
    dataset = {
        "train": train_set,
        "dev": valid_set,
    }

    # get data loader
    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler

        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=local_rank,
            shuffle=True,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            batch_size=config.data_config.batch_size,
            num_workers=config.data_config.num_workers,
            sampler=sampler["train"],
            pin_memory=config.data_config.pin_memory,
            drop_last=True,
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False,
            batch_size=config.data_config.batch_size,
            num_workers=config.data_config.num_workers,
            sampler=sampler["dev"],
            pin_memory=config.data_config.pin_memory,
        ),
    }

    # define models
    svc_mdl = SVCNN(config).to(device)
    discriminator = MelGANMultiScaleDiscriminator().to(device)
    model = {
        "generator": svc_mdl,
        "discriminator": discriminator,
    }

    # define criterions
    criterion = {
        "gen_adv": GeneratorAdversarialLoss(
            # keep compatibility
            **config.loss_config.generator_adv_loss_params
        ).to(device),
        "dis_adv": DiscriminatorAdversarialLoss(
            # keep compatibility
            **config.loss_config.discriminator_adv_loss_params
        ).to(device),
    }
    criterion["stft"] = MultiResolutionSTFTLoss(
            **config.loss_config.stft,
        ).to(device)

    # define optimizers and schedulers
    optimizer = {
        "generator": torch.optim.Adam(model["generator"].parameters(), lr=config.optimizer_config.lr),
        "discriminator": torch.optim.Adam(model["discriminator"].parameters(), lr=config.optimizer_config.lr),
    }
    scheduler = {
    "generator": StepLRScheduler(optimizer["generator"], step_size=config.optimizer_config.scheduler_step_size, gamma=config.optimizer_config.scheduler_gamma),
    "discriminator": StepLRScheduler(optimizer["discriminator"], step_size=config.optimizer_config.scheduler_step_size, gamma=config.optimizer_config.scheduler_gamma),
    }
    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel
        model["generator"] = DistributedDataParallel(model["generator"])
        model["discriminator"] = DistributedDataParallel(model["discriminator"])

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if args.resume:
        if args.pretrain != "":
            trainer.load_checkpoint(args.pretrain, load_only_params=False, dst_train=args.distributed)
            logging.info(f"Successfully load parameters from {args.pretrain}.")
        else:
            if os.path.isdir(args.cp_path):
                dir_files = os.listdir(args.cp_path)
                cp_files = [fname for fname in dir_files if fname[:11] == 'checkpoint-']
                if len(cp_files) == 0:
                    logging.info(f'No pretrained checkpoints. Training from scratch...')
                else:
                    cp_files.sort(key=lambda fname: os.path.getmtime(f'{args.cp_path}/{fname}'))
                    latest_cp = f'{args.cp_path}/{cp_files[-1]}'
                    trainer.load_checkpoint(latest_cp, load_only_params=False, dst_train=args.distributed)
                    logging.info(f'No pretrain dir specified, use the latest one instead. Successfully load parameters from {latest_cp}.')
            else:
                logging.info(f'No pretrain dir specified. Training from scratch...')
    # run training loop
    try:
        trainer.run()
    finally:
        trainer.save_checkpoint(
            os.path.join(config.interval_config.out_dir, f"checkpoint-{trainer.steps}steps.pkl"), args.distributed
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
