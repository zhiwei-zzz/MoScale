import os
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

from model.vq.hrvqvae import HRVQVAE
from config.load_config import load_config

import numpy as np
from utils.fixseeds import *
from utils.eval_t2m import evaluation_vqvae

from model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.get_opt import get_opt

from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader

def load_vq_model(cfg, device):
    vq_model = HRVQVAE(cfg,
            cfg.data.dim_pose,
            cfg.model.down_t,
            cfg.model.stride_t,
            cfg.model.width,
            cfg.model.depth,
            cfg.model.dilation_growth_rate,
            cfg.model.vq_act,
            cfg.model.use_attn,
            cfg.model.vq_norm)
        

    ckpt = torch.load(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'hrvqvae', cfg.exp.name, 'model', 'net_best_fid.tar'),
                            map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {cfg.exp.name} from epoch {ckpt["ep"]}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model

if __name__ == "__main__":
    cfg = load_config('config/eval_hrvqvae.yaml')
    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'hrvqvae', cfg.vqvae_name)

    repeat_time = cfg.repeat_time

    n_cfg = load_config(pjoin(cfg.exp.checkpoint_dir, 'train_hrvqvae.yaml'))
    n_cfg.exp.device = cfg.exp.device
    n_cfg.exp.checkpoint_dir = cfg.exp.checkpoint_dir
    n_cfg.data.root_dir = cfg.data.root_dir
    cfg = n_cfg
    
    fixseed(cfg.exp.seed)

    if cfg.exp.device != 'cpu':
        torch.cuda.set_device(cfg.exp.device)

    torch.autograd.set_detect_anomaly(True)
    
    device = torch.device(cfg.exp.device)

    cfg.exp.model_dir = pjoin(cfg.exp.checkpoint_dir, 'model')
    cfg.exp.eval_dir = pjoin(cfg.exp.checkpoint_dir, 'animation')
    cfg.exp.log_dir = pjoin(cfg.exp.root_log_dir, cfg.data.name, 'hrvqvae',cfg.exp.name)

    cfg.data.feat_dir = pjoin(cfg.data.root_dir, 'renamed_feats')
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
    data_split_dir = pjoin(cfg.data.root_dir, 'data_split_info1')
    all_caption_path = pjoin(cfg.data.root_dir, 'all_caption_clean.json')

    val_mid_split_file = pjoin(data_split_dir, 'test_fnames.txt')
    val_cid_split_file = pjoin(data_split_dir, 'test_ids.txt')
    
    net = load_vq_model(cfg, device)

    dataset_opt_path = 'checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'), data_root=cfg.data.root_dir)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))

    eval_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=device, data_root=cfg.data.root_dir)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    mpjpe = []
    for i in range(repeat_time):
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe = evaluation_vqvae(
                cfg.exp.model_dir, eval_loader, net, None, i, 0, best_fid=1000,
                best_div=100, best_top1=0,
                best_top2=0, best_top3=0, best_matching=0,
                eval_wrapper=eval_wrapper, save=False, draw=False)

        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        mpjpe.append(best_mpjpe)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    mpjpe = np.array(mpjpe)

    msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}\n" \
                f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}\n" \
                f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}\n" \
                f"\tMPJPE: {np.mean(mpjpe)*10:.4f}, conf. {np.std(mpjpe)*1.96/np.sqrt(repeat_time)*10:.4f}\n\n"
    print(msg_final)
    