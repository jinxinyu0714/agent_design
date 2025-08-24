import torch
from utils import farthest_point_sample, square_distance, index_points
from torch import nn



def query_ball_point(radius, nsample, xyz, new_xyz):

    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    # 对索引按照距离进行排序
    sorted_indices = torch.argsort(sqrdists, dim=-1)
    sorted_group_idx = torch.gather(group_idx, -1, sorted_indices)

    group_first = sorted_group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, sorted_group_idx.size(2))
    mask = sorted_group_idx >= N
    sorted_group_idx[mask] = group_first[mask]

    # 随机选择nsample个点
    rand_indices = torch.randperm(sorted_group_idx.size(2))[:nsample].repeat(B, S, 1).to(device)
    group_idx = torch.gather(sorted_group_idx, 2, rand_indices)

    return group_idx

class Group(nn.Module):
    def __init__(self, num_group, group_size, radius=0.2):
        self.radius = radius
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        batch_size, num_points, C = xyz.shape
        # fps the centers out
        center = farthest_point_sample(xyz, self.num_group)  # B G 3
        # 球查询
        idx = query_ball_point(self.radius, self.group_size, xyz, center)

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size

        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
        grouped_xyz_norm = grouped_xyz - center.view(batch_size, self.num_group, 1, C)
        return grouped_xyz_norm, center


if __name__ == '__main__':
    # 创建输入数据
    num_group = 2
    group_size = 3
    xyz = torch.tensor([[[1, 2, 3, 1], [1, 2, 4, 9], [3, 2, 1, 100], [1, 3, 2, 2000]]], dtype=torch.float)

    # 创建 Group 实例
    group_layer = Group(num_group, group_size)

    # 调用 forward 方法
    neighborhood, center = group_layer(xyz)

    print("Neighborhood shape:", neighborhood.shape)  # 应为 [batch_size, num_group, group_size, C]
    print("Center shape:", center.shape)  # 应为 [batch_size, num_group, C]

