"""
@ezerilli

MRI module for implementing self-distillation with no labels (DINO) in the case of image-to-image reconstruction.

    M. Caron et al. Emerging Properties in Self-Supervised Vision Transformers. arXiv:2104.14294. 2021.
"""
import copy
import numpy as np
from argparse import ArgumentParser

import torch
from fastmri.models import DinoNet, DinoLoss

from .mri_module import MriModule


class DinoModule(MriModule):
    """
    DINO training module.

    This can be used to train baseline U-Nets from the paper:
        J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. arXiv:1811.08839. 2018.

    The DINO self-supervised approach is inspired by:
        M. Caron et al. Emerging Properties in Self-Supervised Vision Transformers. arXiv:2104.14294. 2021.
    """

    def __init__(
        self,
        epochs,
        niter_per_epochs,
        momentum,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        **kwargs,
    ):
        """
        Args:
            epochs: number of training epochs.
            niter_per_epochs: number of iterations per epochs == number of batches.
            momentum: starting value of the momentum, annealead to 1.0 with a cosine annealing schedule.
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.student = DinoNet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob
        )

        # teacher and student start with the same weights
        self.teacher = copy.deepcopy(self.student)
        # self.teacher_model.load_state_dict(self.student_model.module.state_dict())

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Specific loss for DINO
        self.dino_loss = DinoLoss().cuda()

        # Define the EMA momentum cosine schedule for the teacher momentum upgrade from student params
        iters = np.arange(epochs * niter_per_epochs)
        schedule = 1.0 + 0.5 * (momentum - 1.0) * (1.0 + np.cos(np.pi * iters / len(iters)))
        self.momentum_schedule = schedule[::-1].tolist()

    def forward(self, images):
        return self.student(images).squeeze(1)

    def teacher_forward(self, images):
        return self.teacher(images).squeeze(1)

    def training_step(self, batch, batch_idx):
        images, target, _, _, _, _, _ = batch

        teacher_output = self.teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = self.student(images)
        loss = self.dino_loss(student_output, teacher_output)
        self.log("loss", loss.detach())

        # EMA update for the teacher
        with torch.no_grad():
            m = self.momentum_schedule.pop()  # momentum parameter
            for param_student, param_teacher in zip(self.student.parameters(), self.teacher.parameters()):
                param_teacher.data.mul_(m).add_((1 - m) * param_student.detach().data)

        return loss

    def validation_step(self, batch, batch_idx):
        images, target, mean, std, fname, slice_num, max_value = batch

        teacher_output = self.teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = self.student(images)
        loss = self.dino_loss(student_output, teacher_output)
        output = self(images[0])

        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output * std + mean,
            "target": target * std + mean,
            "val_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        images, _, mean, std, fname, slice_num, _ = batch
        output = self.forward(images[0])
        mean = mean.unsqueeze(1).unsqueeze(2)
        std = std.unsqueeze(1).unsqueeze(2)

        return {
            "fname": fname,
            "slice": slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans",
            default=1,
            type=int,
            help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans",
            default=1,
            type=int,
            help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans",
            default=1,
            type=int,
            help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob",
            default=0.0,
            type=float,
            help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=8,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        parser.add_argument(
            '--momentum_teacher',
            default=0.995,
            type=float,
            help="""Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule.
            We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")

        return parser
