import numpy as np
from scipy import linalg
import torch


def calculate_mpjpe_perSample(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq



def calculate_mpjpe_batch(
    gt_joints: torch.Tensor,          # [B, T, J, 3]
    pred_joints: torch.Tensor,        # [B, T, J, 3]
    valid_mask: torch.Tensor,         # [B, T]  (True/1 = valid, False/0 = pad)
):
    """
    Batched MPJPE with per-sample variable lengths via a mask.

    Args:
        gt_joints:  [B, T, J, 3]
        pred_joints:[B, T, J, 3]
        valid_mask: [B, T] boolean or {0,1}; True means this frame is valid.
        align: 
          - "self": subtract each sequence's own pelvis (matches your original code)

    Returns:
        mpjpe_per_frame: [B, T] MPJPE per frame (unreduced; padded frames kept but can be ignored via mask)
        mpjpe_per_sample: [B]   mean MPJPE over valid frames per sample
    """
    assert gt_joints.shape == pred_joints.shape, \
        f"GT shape {gt_joints.shape} != pred shape {pred_joints.shape}"
    assert gt_joints.dim() == 4 and gt_joints.size(-1) == 3, "expected [B,T,J,3]"
    B, T, J, _ = gt_joints.shape
    assert valid_mask.shape == (B, T), f"valid_mask must be [B,T], got {valid_mask.shape}"

    # pelvis joint index 0 (same as your original)
    gt_pelvis  = gt_joints[:, :, 0, :]      # [B, T, 3]
    pred_pelvis= pred_joints[:, :, 0, :]    # [B, T, 3]

    # subtract each stream's own pelvis
    gt_centered   = gt_joints   - gt_pelvis.unsqueeze(2)    # [B,T,1,3] → [B,T,J,3]
    pred_centered = pred_joints - pred_pelvis.unsqueeze(2)

    # per-joint L2, then average joints → per-frame MPJPE
    diff = pred_centered - gt_centered                 # [B,T,J,3]
    mpjpe_per_frame = torch.linalg.norm(diff, dim=-1).mean(dim=-1)  # [B,T]

    # masked mean over frames (avoid div by 0)
    m = valid_mask.to(mpjpe_per_frame.dtype)           # [B,T] in {0.,1.}
    denom = m.sum(dim=1).clamp_min(1.0)                # [B]
    mpjpe_per_sample = (mpjpe_per_frame * m).sum(dim=1) / denom     # [B]

    return mpjpe_per_frame, mpjpe_per_sample



def calculate_mpjpe(preds: torch.Tensor, gts: torch.Tensor, mask: torch.Tensor, only_local: bool = False) -> torch.Tensor:
    """
    Computes MPJPE between predicted and ground truth 3D joint positions.

    Args:
        preds: torch.Tensor of shape (batch_size, num_frames, num_joints, 3)
        gts: torch.Tensor of shape (batch_size, num_frames, num_joints, 3)

    Returns:
        mpjpe: torch.Tensor (scalar), mean per-joint position error
    """
    assert preds.shape == gts.shape, f"Shape mismatch between predictions: {preds.shape} and ground truth: {gts.shape}."

    if only_local:
        preds = preds - preds[..., 0, :].unsqueeze(-2)  # Root-centered predictions
        gts = gts - gts[..., 0, :].unsqueeze(-2)  # Root-centered ground truth
    error = torch.norm(preds - gts, dim=-1)  # Compute L2 distance per joint
    mask = mask.unsqueeze(-1).expand_as(error)

    error[~mask] = 0
    # mpjpe = masked_error.sum() / mask.sum()

    return error.sum(), mask.sum()


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def euclidean_distance_matrix_gpu(matrix1, matrix2):
    """GPU version: matrix1, matrix2 are torch tensors on GPU."""
    d1 = -2 * torch.mm(matrix1, matrix2.T)
    d2 = (matrix1 ** 2).sum(dim=1, keepdim=True)
    d3 = (matrix2 ** 2).sum(dim=1)
    dists = torch.sqrt(torch.clamp(d1 + d2 + d3, min=0))
    return dists


def calculate_top_k_gpu(mat, top_k):
    """GPU version of calculate_top_k."""
    size = mat.shape[0]
    gt_mat = torch.arange(size, device=mat.device).unsqueeze(1).expand(size, size)
    bool_mat = (mat == gt_mat)
    correct_vec = torch.zeros(size, dtype=torch.bool, device=mat.device)
    top_k_list = []
    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec.unsqueeze(1))
    top_k_mat = torch.cat(top_k_list, dim=1)
    return top_k_mat


def calculate_R_precision_gpu(embedding1, embedding2, top_k, sum_all=False):
    """GPU version of calculate_R_precision using euclidean distance."""
    dist_mat = euclidean_distance_matrix_gpu(embedding1, embedding2)
    argmax = torch.argsort(dist_mat, dim=1)
    top_k_mat = calculate_top_k_gpu(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(dim=0).float()
    else:
        return top_k_mat


def calculate_activation_statistics_gpu(activations):
    """
    GPU version: activations is a torch tensor on GPU.
    Computes mean and covariance entirely on GPU, then returns numpy arrays for FID.

    Args:
        activations: torch.Tensor of shape (num_samples, dim_feat) on GPU
    Returns:
        mu: numpy array of shape (dim_feat,)
        cov: numpy array of shape (dim_feat, dim_feat)
    """
    # Use float64 for numerical stability (FID matrix sqrt is sensitive)
    activations = activations.double()

    # Compute mean on GPU
    mu = activations.mean(dim=0)

    # Compute covariance on GPU: cov = (X - mean)^T @ (X - mean) / (n - 1)
    n_samples = activations.shape[0]
    centered = activations - mu.unsqueeze(0)
    cov = centered.T @ centered / (n_samples - 1)

    # Only move to CPU at the very end for FID calculation (needs scipy)
    return mu.cpu().numpy(), cov.cpu().numpy()


def calculate_diversity_gpu(activation, diversity_times):
    """GPU version: activation is a torch tensor on GPU."""
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = torch.randperm(num_samples, device=activation.device)[:diversity_times]
    second_indices = torch.randperm(num_samples, device=activation.device)[:diversity_times]
    dist = torch.linalg.norm(activation[first_indices] - activation[second_indices], dim=1)
    return dist.mean().item()

def cosine_similarity_matrix(matrix1, matrix2):
    matrix1 = matrix1 / np.linalg.norm(matrix1, axis=-1, keepdims=True)
    matrix2 = matrix2 / np.linalg.norm(matrix2, axis=-1, keepdims=True)
    sim_matrix = np.dot(matrix1,  matrix2.T)
    return sim_matrix

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat

# using embedding2 to retreive embedding 1
def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False, is_cosine_sim=False):
    if is_cosine_sim:
        dist_mat = - cosine_similarity_matrix(embedding1, embedding2)
    else:
        dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


# def calculate_matching_score(embedding1, embedding2, sum_all=False):
#     assert len(embedding1.shape) == 2
#     assert embedding1.shape[0] == embedding2.shape[0]
#     assert embedding1.shape[1] == embedding2.shape[1]

#     dist = linalg.norm(embedding1 - embedding2, axis=1)
#     if sum_all:
#         return dist.sum(axis=0)
#     else:
#         return dist



def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


