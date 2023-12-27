import logging
import os

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch

from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.tools import save_checkpoint, load_checkpoint

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for FastSVC training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        tensorboard_dir = os.path.join(config.interval_config.out_dir, 'logs')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config.training_config.train_max_steps, desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x, y = batch # x: (mels, pitch, ld, spk_index), y: audio
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        #######################
        #      Generator      #
        #######################
        if self.steps > 0:
            y_ = self.model["generator"](*x)

            # initialize
            gen_loss = 0.0

            # multi-resolution sfft loss
            sc_loss, mag_loss = self.criterion["stft"](y_, y)
            gen_loss += sc_loss + mag_loss
            self.total_train_loss[
                "train/spectral_convergence_loss"
            ] += sc_loss.item()
            self.total_train_loss[
                "train/log_stft_magnitude_loss"
            ] += mag_loss.item()

            # weighting aux loss
            gen_loss *= self.config.loss_config.lambda_aux

            # adversarial loss
            if self.steps > self.config.training_config.discriminator_train_start_steps:
                p_ = self.model["discriminator"](y_.unsqueeze(1))
                adv_loss = self.criterion["gen_adv"](p_)
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()

                # add adversarial loss to generator loss
                gen_loss += self.config.loss_config.lambda_adv * adv_loss

            self.total_train_loss["train/generator_loss"] += gen_loss.item()

            # update generator
            self.optimizer["generator"].zero_grad()
            self.optimizer["discriminator"].zero_grad()
            gen_loss.backward()
            if self.config.training_config.generator_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["generator"].parameters(),
                    self.config.training_config.generator_grad_norm,
                )
            self.optimizer["generator"].step()
            self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config.training_config.discriminator_train_start_steps:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_ = self.model["generator"](*x)

            # discriminator loss
            p = self.model["discriminator"](y.unsqueeze(1))
            p_ = self.model["discriminator"](y_.unsqueeze(1).detach())
            real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
            dis_loss = real_loss + fake_loss
            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config.training_config.discriminator_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config.training_config.discriminator_grad_norm,
                )

            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config.training_config.rank == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

        # needed for shuffle in distributed training
        if self.config.training_config.distributed:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x)

        # initialize
        aux_loss = 0.0

        # multi-resolution stft loss
        sc_loss, mag_loss = self.criterion["stft"](y_, y)
        aux_loss += sc_loss + mag_loss
        self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
        self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()

        # weighting stft loss
        aux_loss *= self.config.loss_config.lambda_aux

        # adversarial loss
        p_ = self.model["discriminator"](y_.unsqueeze(1))
        adv_loss = self.criterion["gen_adv"](p_)
        gen_loss = aux_loss + self.config.loss_config.lambda_adv * adv_loss

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](y.unsqueeze(1))
        p_ = self.model["discriminator"](y_.unsqueeze(1))

        # discriminator loss
        real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
        dis_loss = real_loss + fake_loss

        # add to total eval loss
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        # generate
        x_batch, y_batch = batch
        x_batch = tuple([x.to(self.device) for x in x_batch])
        y_batch = y_batch.to(self.device)
        y_batch_ = self.model["generator"](*x_batch)

        # check directory
        dirname = os.path.join(self.config.interval_config.out_dir, f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(
                figname.replace(".png", "_ref.wav"),
                y,
                self.config.data_config.sampling_rate,
                "PCM_16",
            )
            sf.write(
                figname.replace(".png", "_gen.wav"),
                y_,
                self.config.data_config.sampling_rate,
                "PCM_16",
            )

            if idx >= self.config.interval_config.num_save_intermediate_results:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config.interval_config.save_interval_steps == 0:
            self.save_checkpoint(
                os.path.join(self.config.interval_config.out_dir, f"checkpoint-{self.steps}steps.pkl"), self.config.training_config.distributed
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config.interval_config.eval_interval_steps == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config.interval_config.log_interval_steps == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config.interval_config.log_interval_steps
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config.training_config.train_max_steps:
            self.finish_train = True

    def load_checkpoint(self, cp_path, load_only_params, dst_train):
        self.steps, self.epochs = load_checkpoint(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, checkpoint_path=cp_path, load_only_params=load_only_params, dst_train=dst_train)

    def save_checkpoint(self, cp_path, dst_train):
        save_checkpoint(steps=self.steps, epochs=self.epochs, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, checkpoint_path=cp_path, dst_train=dst_train)
