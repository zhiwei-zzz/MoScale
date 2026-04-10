from trainers.base_trainer import BaseTrainer

import os

import torch
import time

from copy import deepcopy

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict, defaultdict
from os.path import join as pjoin


from utils.utils import print_current_loss, print_val_loss
from utils.eval_t2m import evaluation_vqvae
import wandb


def mean_flat(tensor: torch.Tensor, mask=None):
    """
    Take the mean over all non-batch dimensions.
    """
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        # mask = mask.unsqueeze(2)           # [B, T] -> [T, B, 1]
        assert tensor.dim() == 3
        denom = mask.sum() * tensor.shape[-1]
        loss = (tensor * mask).sum() / denom
        return loss
    

def length_to_mask(length, max_len, device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = length.device

    if isinstance(length, list):
        length = torch.tensor(length)
    
    length = length.to(device)
    # max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


class HRVQVAETrainer(BaseTrainer):
    def __init__(self, cfg, vq_model, device):
        self.cfg = cfg
        self.vq_model = vq_model
        self.device = device

        self.logger = SummaryWriter(cfg.exp.log_dir)
        if cfg.training.recons_loss == 'l1':
            self.l1_criterion = torch.nn.L1Loss()
        elif cfg.training.recons_loss == 'l1_smooth':
            self.l1_criterion = torch.nn.SmoothL1Loss()

        if cfg.training.ema:
            self.ema_model = deepcopy(vq_model).to(device)
            self.ema_model.eval()
            self.requires_grad(self.ema_model, False)

    def forward_attn(self, batch_data, vq_model, fk_func):
        _, motions, m_lens = batch_data
        motions = motions.detach().to(self.device).float()
        m_lens = m_lens.detach().to(self.device).long()
        pred_motion, loss_commit, perplexity = vq_model(motions[..., :self.cfg.data.dim_pose], m_lens.clone())
        
        self.motions = motions[..., :self.cfg.data.dim_pose]
        self.pred_motion = pred_motion

        mask = length_to_mask(m_lens, max_len=motions.shape[1])
        loss_rec = mean_flat(
            F.smooth_l1_loss(self.pred_motion, self.motions, reduction='none'),
            mask=mask.unsqueeze(-1)
        )

        
        if self.cfg.training.lambda_fk == 0:
            loss_fk = self.l1_criterion(motions, motions)
        else:
            B, T, _ = motions.shape
            loss_fk = mean_flat(
                F.smooth_l1_loss(fk_func(self.motions).view(B, T, -1), 
                                 fk_func(self.pred_motion).view(B, T, -1), 
                                 reduction='none'),
                mask=mask.unsqueeze(-1))

        loss_global = mean_flat(
            F.smooth_l1_loss(self.pred_motion[..., :4], self.motions[..., :4], reduction='none'),
            mask=mask.unsqueeze(-1)
        )

        loss_vel = mean_flat(
            F.smooth_l1_loss(self.pred_motion[..., 4:67], self.motions[..., 4:67], reduction='none'),
            mask=mask.unsqueeze(-1)
        )

        loss = loss_rec + \
            self.cfg.training.lambda_global * loss_global + \
            self.cfg.training.lambda_fk * loss_fk + \
                self.cfg.training.lambda_commit * loss_commit +\
                self.cfg.training.lambda_expicit * loss_vel

        loss_logs = OrderedDict()
        loss_logs["loss"] = loss.item()
        loss_logs["loss_rec"] = loss_rec.item()
        loss_logs["loss_global"] = loss_global.item()
        loss_logs["loss_commit"] = loss_commit.item()
        loss_logs["loss_fk"] = loss_fk.item()
        loss_logs["loss_vel"] = loss_vel.item()
        loss_logs["perplexity"] = perplexity.item()
        return loss, loss_logs

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep):
        state = {'vq_model': self.vq_model.state_dict(), 'ep': ep}
        if self.cfg.training.ema:
            state["ema_model"] = self.ema_model.state_dict()
        torch.save(state, file_name)

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None, fk_func=None):
        self.vq_model.to(self.device)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.cfg.training.lr, betas=(0.9, 0.99), weight_decay=self.cfg.training.weight_decay)
        

        epoch = 0
        it = 0

        if self.cfg.training.ema:
            self.update_ema(self.ema_model, self.vq_model, decay=0)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model,
                                                              milestones=self.cfg.training.milestones,
                                                              gamma=self.cfg.training.gamma)


        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _ = evaluation_vqvae(
            self.cfg.exp.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, it, best_fid=1000,
            best_div=100, best_top1=0,
            best_top2=0, best_top3=0, best_matching=100,
            eval_wrapper=eval_wrapper, save=False)


        start_time = time.time()
        total_iters = self.cfg.training.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.cfg.training.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        def def_value():
            return 0.0
        logs = defaultdict(def_value, OrderedDict())

        while epoch < self.cfg.training.max_epoch:
            self.vq_model.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.cfg.training.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.cfg.training.warm_up_iter, self.cfg.training.lr)
                loss, loss_log = self.forward_attn(batch_data, self.vq_model, fk_func)
                self.opt_vq_model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vq_model.parameters(), max_norm=0.5)
                self.opt_vq_model.step()

                if self.cfg.training.ema:
                    self.update_ema(self.ema_model, self.vq_model)
                
                for key, val in loss_log.items():
                    logs[key] += val

                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.cfg.training.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.cfg.training.log_every, it)
                        mean_loss[tag] = value / self.cfg.training.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                    wandb.log(
                        {f"Loss/{k}": v for k, v in mean_loss.items()},
                        step=it       # x-axis on the dashboard
                    )

            if it >= self.cfg.training.warm_up_iter:
                self.scheduler.step()

            epoch += 1

            print('Validation time:')
            self.vq_model.eval()
            
            val_logs = defaultdict(def_value, OrderedDict())
            eval_model = self.ema_model if self.cfg.training.ema else self.vq_model
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, loss_log = self.forward_attn(batch_data, eval_model, fk_func)
                    for key, val in loss_log.items():
                        val_logs[key] += val
            mean_loss = OrderedDict()
            for tag, value in val_logs.items():
                self.logger.add_scalar('Val/%s'%tag, value / len(val_loader), epoch)
                mean_loss[tag] = value / len(val_loader)
            
            print_val_loss(mean_loss, epoch)

            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _ = evaluation_vqvae(
                self.cfg.exp.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, it, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1,
                best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, eval_wrapper=eval_wrapper)


            if epoch % self.cfg.training.eval_every_e == 0:
                data = torch.cat([self.motions[:4], self.pred_motion[:4]], dim=0)
                save_dir = pjoin(self.cfg.exp.eval_dir, 'E%04d' % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir)

