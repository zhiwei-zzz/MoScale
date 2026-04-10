import torch
from collections import defaultdict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # type: ignore
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_moscale_transformer
from model.transformer.tools import *
from model.transformer.lr_helper import lr_wd_annealing, filter_params
from model.transformer.amp_sc import AmpOptimizer
from functools import partial


from einops import rearrange, repeat

from trainers.base_trainer import BaseTrainer
import wandb

def def_value():
    return 0.0

class MoScaleTrainer(BaseTrainer):
    def __init__(self, cfg, moscale, vq_model, device):
        self.cfg = cfg
        self.moscale = moscale
        self.vq_model = vq_model
        self.device = device
        self.vq_model.eval()

        self.logger = SummaryWriter(cfg.exp.log_dir)


    def forward(self, batch_data, train=False):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        _loss, _pred_ids, _acc = self.moscale(motion, conds, m_lens, self.vq_model, train)

        return _loss, _acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data, train=True)
        grad_norm, scale_log2 = self.opt_t2m_transformer.backward_clip_step(loss=loss, stepping=True)
        return loss.item(), acc

    def save(self, file_name, ep):
        t2m_trans_state_dict = self.moscale.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            'moscale': t2m_trans_state_dict,
            'ep': ep,
        }
        torch.save(state, file_name)

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.moscale.to(self.device)
        self.vq_model.to(self.device)


        names, paras, para_groups = filter_params(
            self.moscale,
            nowd_keys={
                'cls_token','start_token','task_token','cfg_uncond', 'mask_feature', 'pad_emb', 'txt_pad_emb', 'motion_pad_emb',
                'pos_embed','position_enc_learned','pos_start','start_pos','lvl_embed', 'token_emb',
                'gamma','beta','ada_gss','moe_bias','scale_mul',
                'bias','LayerNorm.weight','layernorm','ln','norm','embedding',
                'q_tokens'
            }
        )

        opt_clz = partial(torch.optim.AdamW, betas=(0.9, 0.98), fused=getattr(self.cfg.training, 'afuse', False))

        opt_kw = dict(lr=self.cfg.training.lr, weight_decay=0.0)
        print(f'[INIT] t2m optim={opt_clz}, opt_kw={opt_kw}\n')

        base_optim = opt_clz(params=para_groups, **opt_kw)

        self.opt_t2m_transformer = AmpOptimizer(
            mixed_precision=self.cfg.training.fp16,
            optimizer=base_optim,
            names=names, paras=paras,
            grad_clip=self.cfg.training.tclip,
            n_gradient_accumulation=self.cfg.training.ac
        )
        del names, paras, para_groups

        epoch = 0
        it = 0

        max_it = self.cfg.training.max_epoch * len(train_loader)
        wp_it = self.cfg.training.wp * len(train_loader)

        start_time = time.time()
        total_iters = self.cfg.training.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.cfg.training.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching = evaluation_moscale_transformer(
                self.cfg.exp.model_dir, eval_val_loader, self.moscale, self.vq_model, self.logger, epoch, it,
                best_fid=1000, best_div=100,
                best_top1=0, best_top2=0, best_top3=0, cond_scale=5,
                best_matching=100, eval_wrapper=eval_wrapper, device=self.device,
                plot_func=plot_eval, save_ckpt=False, save_anim=False
            )
        best_acc = 0.
        global_best_top1 = 0

        early_stop_epoch = self.cfg.training.get('early_stop_epoch', self.cfg.training.max_epoch)
        while epoch < min(self.cfg.training.max_epoch, early_stop_epoch):
            self.moscale.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(self.cfg.training.sche, self.opt_t2m_transformer.optimizer, self.cfg.training.lr,
                                                            self.cfg.training.twd, self.cfg.training.twde, it, wp_it, max_it,
                                                            wp0=self.cfg.training.wp0, wpe=self.cfg.training.wpe,
                                                            step_decay_steps=self.cfg.training.get('step_milestones', None),
                                                            step_decay_rate=self.cfg.training.get('step_gamma', 0.1))

                loss, acc = self.update(batch)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr/min'] += min_tlr
                logs['lr/max'] += max_tlr
                logs['wd/min'] += min_twd
                logs['wd/max'] += max_twd

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

            epoch += 1

            print('Validation time:')
            self.vq_model.eval()
            self.moscale.eval()

            # validate in train mode
            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data, train=True)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            wandb.log(
                {
                    "Val/train_loss": np.mean(val_loss),
                    "Val/train_acc": np.mean(val_acc),
                },
                step=it       # x-axis on the dashboard
            )

            self.logger.add_scalar('Val/train_loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/train_acc', np.mean(val_acc), epoch)

            if np.mean(val_acc) > best_acc:
                print(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                self.save(pjoin(self.cfg.exp.model_dir, 'net_best_acc.tar'), epoch)
                best_acc = np.mean(val_acc)

            # validate in test mode
            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data, train=False)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            wandb.log(
                {
                    "Val/eval_loss": np.mean(val_loss),
                    "Val/eval_acc": np.mean(val_acc),
                },
                step=it       # x-axis on the dashboard
            )

            self.logger.add_scalar('Val/eval_loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/eval_acc', np.mean(val_acc), epoch)

            # begin the evaluation
            if epoch%self.cfg.training.eval_every_e==0:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching = evaluation_moscale_transformer(
                        self.cfg.exp.model_dir, eval_val_loader, self.moscale, self.vq_model, self.logger, epoch, it, best_fid=best_fid,
                        best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, cond_scale=5,
                        best_matching=best_matching, eval_wrapper=eval_wrapper, device=self.device,
                        plot_func=plot_eval, save_ckpt=True, save_anim=False
                    )


                if best_top1 > global_best_top1:
                    global_best_top1 = best_top1
                    self.save(pjoin(self.cfg.exp.model_dir, 'net_best_top1.tar'), epoch)
                    print(f"New global best top1: {global_best_top1:.3f} at epoch {epoch}, saved net_best_top1.tar")