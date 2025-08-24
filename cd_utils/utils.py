import torch
import numpy as np


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz[:, :, :3] - centroid[:, :, :3]) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    # return index_points(xyz, centroids)
    return centroids


def square_distance(src, dst):

    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src[:, :, :3], dst[:, :, :3].permute(0, 2, 1))
    dist += torch.sum(src[:, :, :3] ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst[:, :, :3] ** 2, -1).view(B, 1, M)
    return dist


if __name__ == '__main__':
    src = torch.tensor([[[1, 2, 3, 1], [1, 2, 4, 9]]])
    dst = torch.tensor([[[1, 3, 2, 100]]])
    out = square_distance(src, dst)
    print(out.shape)
