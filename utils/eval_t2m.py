import os

import numpy as np
import torch
from utils.metrics import *
from utils.motion_process import recover_from_ric
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import os, json, heapq


class BestCheckpointSaver:
    def __init__(self, out_dir=None, N=5, start_epoch=0,
                 index_name="best_fid_index.json",
                 filename_fmt="ckpt_fid{fid:.5f}_ep{ep}.tar"):
        self.out_dir = None
        self.index_name = index_name
        self.index_path = None
        self.N = int(N)
        self.start_epoch = int(start_epoch)
        self.filename_fmt = filename_fmt
        # heap items: (-fid, ep, path, fid)
        self.heap = []

        if out_dir is not None:
            self.set_out_dir(out_dir)

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.index_path = os.path.join(self.out_dir, self.index_name)

        if os.path.isfile(self.index_path):
            try:
                with open(self.index_path, "r") as f:
                    items = json.load(f)
                self.heap[:] = []
                for it in items:
                    fid = float(it["fid"]); ep = int(it["ep"]); path = it["path"]
                    heapq.heappush(self.heap, (-fid, ep, path, fid))
            except Exception:
                self.heap = []

    def _save_index(self):
        if not self.index_path:
            return
        items = [{"fid": x[3], "ep": x[1], "path": x[2]}
                 for x in sorted(self.heap, key=lambda t: (t[3], t[1]))]
        with open(self.index_path, "w") as f:
            json.dump(items, f, indent=2)

    def _mk_path(self, fid, ep):
        assert self.out_dir is not None, "out_dir not set. Call set_out_dir(...) or pass out_dir to update()."
        fname = self.filename_fmt.format(fid=fid, ep=ep)
        return os.path.join(self.out_dir, fname)

    def should_consider(self, fid, ep):
        if ep < self.start_epoch:
            return False
        if len(self.heap) < self.N:
            return True
        worst_fid = self.heap[0][3]
        return fid < worst_fid

    def update(self, fid, ep, save_fn, out_dir=None):
        """
        save_fn(path, ep) must write the checkpoint to 'path'.
        If self.out_dir is not set yet, pass out_dir on the first call or call set_out_dir(...) earlier.
        """
        if out_dir is not None and self.out_dir is None:
            self.set_out_dir(out_dir)
        if self.out_dir is None:
            raise ValueError("out_dir is not set. Call set_out_dir(...) or pass out_dir=... to update().")

        if ep < self.start_epoch:
            return False

        if len(self.heap) < self.N:
            path = self._mk_path(fid, ep)
            save_fn(path, ep)
            heapq.heappush(self.heap, (-fid, ep, path, fid))
            self._save_index()
            return True

        worst_negfid, worst_ep, worst_path, worst_fid = self.heap[0]
        if fid < worst_fid:
            new_path = self._mk_path(fid, ep)
            save_fn(new_path, ep)
            heapq.heappop(self.heap)
            try:
                if os.path.isfile(worst_path):
                    os.remove(worst_path)
            except Exception:
                pass
            heapq.heappush(self.heap, (-fid, ep, new_path, fid))
            self._save_index()
            return True

        return False

    def current_best(self):
        if not self.heap:
            return None
        best = min(self.heap, key=lambda x: x[3])
        return (best[3], best[1], best[2])


saver = BestCheckpointSaver(N=5, start_epoch=0) 

def length_to_mask(length, max_len, device: torch.device = None) -> torch.Tensor: # type: ignore
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)
    
    length = length.to(device)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


@torch.no_grad()
def evaluation_evaluator(out_dir, eval_val_loader, writer, ep, best_top1, best_top2, best_top3,
                         best_matching, eval_model, device, save_ckpt=True, draw=True):

    def save(file_path, ep):
        state = {
            "latent_enc": eval_model.latent_enc.state_dict(),
            "text_enc": eval_model.text_enc.state_dict(),
            "ep": ep,
        }

        if "motion_enc" in eval_model.state_dict():
            state["motion_enc"] = eval_model.motion_enc.state_dict()

        torch.save(state, file_path)

    motion_annotation_list = []

    R_precision_real = 0

    nb_sample = 0
    matching_score_real = 0
    for batch in eval_val_loader:
        texts, motions, m_lengths = batch

        motions = motions[..., :148]
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_model.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_model.encode_motion(motions, m_lengths, sample_mean=True)

        bs, _ = motions.shape[0], motions.shape[1]


        motion_annotation_list.append(fid_em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample

    matching_score_real = matching_score_real / nb_sample

    msg = "--> \t Eva. Ep %d:, Diversity Real. %.4f, R_precision_real. (%.4f, %.4f, %.4f), matching_score_real. %.4f"%\
          (ep, diversity_real, R_precision_real[0],R_precision_real[1], R_precision_real[2], matching_score_real ) # type: ignore
    print(msg)

    if draw:
        writer.add_scalar('Eval/Diversity', diversity_real, ep)
        writer.add_scalar('Eval/top1', R_precision_real[0], ep) # type: ignore
        writer.add_scalar('Eval/top2', R_precision_real[1], ep)
        writer.add_scalar('Eval/top3', R_precision_real[2], ep)
        writer.add_scalar('Eval/matching_score', matching_score_real, ep)


    if R_precision_real[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision_real[0])
        if draw: print(msg)
        best_top1 = R_precision_real[0]
        if save_ckpt:
            save(os.path.join(out_dir, 'net_best_top1.tar'), ep)

    if R_precision_real[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision_real[1])
        if draw: print(msg)
        best_top2 = R_precision_real[1]

    if R_precision_real[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision_real[2])
        if draw: print(msg)
        best_top3 = R_precision_real[2]

    if matching_score_real > best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_real)
        if draw: print(msg)
        best_matching = matching_score_real
        if save_ckpt:
            save(os.path.join(out_dir, 'net_best_mm.tar'), ep)

    return diversity_real, best_top1, best_top2, best_top3, best_matching



@torch.no_grad()
def evaluation_vqvae(out_dir, val_loader, net, writer, ep, acc_iter, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, save=True, draw=True):
    net.eval()
    device = next(net.parameters()).device

    motion_annotation_list = []
    motion_pred_list = []

    # GPU accumulators for R_precision (3 values for top-1,2,3)
    R_precision_real = torch.zeros(3, device=device)
    R_precision = torch.zeros(3, device=device)

    nb_sample = 0
    matching_score_real = 0.0
    matching_score_pred = 0.0
    mpjpe_sum = torch.tensor(0.0, device=device)
    num_poses = 0

    # Pre-convert std and mean to GPU tensors for inv_transform
    dataset = val_loader.dataset
    std_gpu = torch.from_numpy(dataset.std).float().to(device)
    mean_gpu = torch.from_numpy(dataset.mean).float().to(device)

    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.to(device)
        m_length = m_length.to(device)
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        _, _, all_codes = net.encode(motion, m_length.clone())
        pred_pose_eval = net.decode(all_codes, m_length.clone())

        # GPU-based inv_transform and MPJPE calculation
        all_gt_motions = (motion.detach() * std_gpu + mean_gpu).float()
        all_pred_motions = (pred_pose_eval.detach() * std_gpu + mean_gpu).float()
        all_gt_joints = recover_from_ric(all_gt_motions, 22)
        all_pred_joints = recover_from_ric(all_pred_motions, 22)
        mask = length_to_mask(m_length, motion.shape[1], device=device)

        mpjpe_sum += calculate_mpjpe_batch(all_gt_joints, all_pred_joints, mask)[0][mask].sum()
        num_poses += mask.sum().item()

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        # Keep embeddings on GPU
        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        # GPU-based R_precision and matching score calculations
        temp_R = calculate_R_precision_gpu(et, em, top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix_gpu(et, em).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match.item()

        temp_R = calculate_R_precision_gpu(et_pred, em_pred, top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix_gpu(et_pred, em_pred).trace()
        R_precision += temp_R
        matching_score_pred += temp_match.item()

        nb_sample += bs

    # Concatenate on GPU, then compute statistics
    motion_annotation_gpu = torch.cat(motion_annotation_list, dim=0)
    motion_pred_gpu = torch.cat(motion_pred_list, dim=0)

    # FID needs scipy, so move to CPU for activation statistics
    gt_mu, gt_cov = calculate_activation_statistics_gpu(motion_annotation_gpu)
    mu, cov = calculate_activation_statistics_gpu(motion_pred_gpu)

    # Diversity on GPU
    diversity_real = calculate_diversity_gpu(motion_annotation_gpu, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity_gpu(motion_pred_gpu, 300 if nb_sample > 300 else 100)

    # Move R_precision to CPU for final division
    R_precision_real = (R_precision_real / nb_sample).cpu().numpy()
    R_precision = (R_precision / nb_sample).cpu().numpy()
    mpjpe = (mpjpe_sum / num_poses).item() * 100  # convert to cm

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, mpjpe. %.4f, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (ep, mpjpe, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/mpjpe', mpjpe, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

        wandb.log(
            {
                "Eval/FID":             fid,
                "Eval/mpjpe":             mpjpe,
                "Eval/Diversity":       diversity,
                "Eval/top1":            R_precision[0],
                "Eval/top2":            R_precision[1],
                "Eval/top3":            R_precision[2],
                "Eval/matching_score":  matching_score_pred,
                "Eval/epoch":           ep,
            },
            step=acc_iter  # keeps the x-axis aligned with epochs/steps
        )

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe


@torch.no_grad()
def evaluation_moscale_transformer(out_dir, val_loader, trans, vq_model, writer, ep, acc_iters, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper,device, plot_func,
                           cond_scale=4, save_ckpt=False, save_anim=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            'moscale': t2m_trans_state_dict,
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.to(device).long().detach()

        bs, seq = pose.shape[:2]

        # (b, seqlen)
        mids = trans.generate(clip_text, m_length//4, cond_scale, temperature=1, vq_model=vq_model)
        pred_motions = vq_model.forward_decoder(mids, m_length.clone())

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.to(device).float().detach()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    writer.add_scalar('Eval/FID', fid, ep)
    writer.add_scalar('Eval/Diversity', diversity, ep)
    writer.add_scalar('Eval/top1', R_precision[0], ep)
    writer.add_scalar('Eval/top2', R_precision[1], ep)
    writer.add_scalar('Eval/top3', R_precision[2], ep)
    writer.add_scalar('Eval/matching_score', matching_score_pred, ep)


    wandb.log(
    {
        "Eval/FID": fid,
        "Eval/Diversity": diversity,
        "Eval/top1": R_precision[0],
        "Eval/top2": R_precision[1],
        "Eval/top3": R_precision[2],
        "Eval/matching_score": matching_score_pred,
        "epoch": ep  # store epoch as a separate metric
    },
    step=acc_iters  # x-axis will still align to iteration count
)



    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir,  'net_best_fid.tar'), ep)
    saver.set_out_dir(out_dir)
    saver.update(fid=fid, ep=ep, save_fn=save)


    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching



@torch.no_grad()
def evaluation_moscale(val_loader, vq_model, trans, repeat_id, eval_wrapper,
                                cond_scale, cal_mm=True, sample_time=None, temperature=1, top_p_thres=0.9):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if  cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]

        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(clip_text, m_length//4, cond_scale, temperature=temperature, vq_model=vq_model, sample_time=sample_time, top_p_thres=top_p_thres)
                pred_motions = vq_model.forward_decoder(mids, m_length.clone())

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(clip_text, m_length//4, cond_scale, temperature=temperature, vq_model=vq_model, sample_time=sample_time, top_p_thres=top_p_thres)
            pred_motions = vq_model.forward_decoder(mids, m_length.clone())

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality