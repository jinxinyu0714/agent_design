import torch
import torch.nn.functional as F
from utils import farthest_point_sample


def cal_loss(pred, gold, smoothing=True):

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def square_distance(src, dst):
  
    B, N, _ = src.shape[:3]
    _, M, _ = dst.shape[:3]
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
  
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)  ## [B,S]
    view_shape[1:] = [1] * (len(view_shape) - 1)  ## [B,1]
    repeat_shape = list(idx.shape)  ## [B,S]
    repeat_shape[0] = 1  ## [1,S]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)


    if torch.any(idx >= points.size(1)):
        print("Index out of range in idx")

    new_points = points[batch_indices, idx, :]
    # print("new_points shape:", new_points.shape)
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
   
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(xyz, new_xyz)  ### changed
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
  
    sqrdists = square_distance(xyz, new_xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
  
    B, N, C = xyz.shape
    S = npoint
    xyz = xyz.contiguous()  ### keep tensor contiguous

    fps_idx = farthest_point_sample(xyz, npoint).long()  # [B, npoint] [B,S]
    new_xyz = index_points(xyz, fps_idx)  ## [B, S, C=3]
    new_points = index_points(points, fps_idx)  ## [B, S, D]
    idx = knn_point(nsample, xyz, new_xyz)

    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points