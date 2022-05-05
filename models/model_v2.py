import torch.nn as nn
import numpy as np

from models.model import TransitionUp, TransitionDown
from models.transformer import TransformerBlock


class PTSeg_v2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.nblocks = nblocks
        """downSample"""
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)
        self.trainsition_downs = nn.ModuleList()
        self.transformer_downs = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.trainsition_downs.append(
                TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformer_downs.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))

        """upSample1"""
        upVersion = 0
        points_upsample = int(512 / (2 ** upVersion))
        nblocks_now = nblocks - upVersion
        # print(upVersion, points_upsample, nblocks_now)
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks_now, points_upsample),
            nn.ReLU(),
            nn.Linear(points_upsample, points_upsample),
            nn.ReLU(),
            nn.Linear(points_upsample, 32 * 2 ** nblocks_now)
        )
        self.transformer1 = TransformerBlock(32 * 2 ** nblocks_now, cfg.model.transformer_dim, nneighbor)
        self.transition_ups1 = nn.ModuleList()
        self.transformer_ups1 = nn.ModuleList()
        for i in reversed(range(nblocks_now)):
            channel = 32 * 2 ** i
            self.transition_ups1.append(TransitionUp(channel * 2, channel, channel))
            self.transformer_ups1.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
        upVersion += 1

        """upSample2"""
        points_upsample = int(512 / (2 ** upVersion))
        nblocks_now = nblocks - upVersion
        self.fc3 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks_now, points_upsample),
            nn.ReLU(),
            nn.Linear(points_upsample, points_upsample),
            nn.ReLU(),
            nn.Linear(points_upsample, 32 * 2 ** nblocks_now)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks_now, cfg.model.transformer_dim, nneighbor)
        self.transition_ups2 = nn.ModuleList()
        self.transformer_ups2 = nn.ModuleList()
        for i in reversed(range(nblocks_now)):
            channel = 32 * 2 ** i
            self.transition_ups2.append(TransitionUp(channel * 2, channel, channel))
            self.transformer_ups2.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
        upVersion += 1

        """upSample3"""
        points_upsample = int(512 / (2 ** upVersion))
        nblocks_now = nblocks - upVersion
        self.fc4 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks_now, points_upsample),
            nn.ReLU(),
            nn.Linear(points_upsample, points_upsample),
            nn.ReLU(),
            nn.Linear(points_upsample, 32 * 2 ** nblocks_now)
        )
        self.transformer3 = TransformerBlock(32 * 2 ** nblocks_now, cfg.model.transformer_dim, nneighbor)
        self.transition_ups3 = nn.ModuleList()
        self.transformer_ups3 = nn.ModuleList()
        for i in reversed(range(nblocks_now)):
            channel = 32 * 2 ** i
            self.transition_ups3.append(TransitionUp(channel * 2, channel, channel))
            self.transformer_ups3.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
        upVersion += 1

        """upSample4"""
        points_upsample = int(512 / (2 ** upVersion))
        nblocks_now = nblocks - upVersion
        self.fc5 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks_now, points_upsample),
            nn.ReLU(),
            nn.Linear(points_upsample, points_upsample),
            nn.ReLU(),
            nn.Linear(points_upsample, 32 * 2 ** nblocks_now)
        )
        self.transformer4 = TransformerBlock(32 * 2 ** nblocks_now, cfg.model.transformer_dim, nneighbor)
        self.transition_ups4 = nn.ModuleList()
        self.transformer_ups4 = nn.ModuleList()
        for i in reversed(range(nblocks_now)):
            channel = 32 * 2 ** i
            self.transition_ups4.append(TransitionUp(channel * 2, channel, channel))
            self.transformer_ups4.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))

        self.fc6 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )

    def forward(self, x):
        """downSapmle"""
        xyz = x[..., :3]
        # print(np.shape(xyz))
        points = self.transformer(xyz, self.fc1(x))[0]

        version = 3
        nblocks_now = self.nblocks - version

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.trainsition_downs[i](xyz, points)
            points = self.transformer_downs[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))

        """upSample4(需要从低深度到高深度）"""
        xyz_4 = xyz_and_feats[1][0]
        points_4 = xyz_and_feats[1][1]
        # print(np.shape(xyz_4))
        # print(np.shape(self.fc5(points_4)))
        points_4 = self.transformer4(xyz_4, self.fc5(points_4))[0]

        for i in range(nblocks_now):
            points_4 = self.transition_ups4[i](xyz_4, points_4, xyz_and_feats[i][0], xyz_and_feats[i][1])
            xyz_4 = xyz_and_feats[i][0]
            points_4 = self.transformer_ups4[i](xyz_4, points_4)[0]
        version -= 1

        """upSample3"""
        nblocks_now = self.nblocks - version
        xyz_3 = xyz_and_feats[2][0]
        points_3 = xyz_and_feats[2][1]
        points_3 = self.transformer3(xyz_3, self.fc4(points_3))[0]
        for i in range(nblocks_now):
            if i == 0:
                points_3 = self.transition_ups3[i](xyz_3, points_3, xyz_and_feats[1][0], xyz_and_feats[1][1])
                xyz_3 = xyz_and_feats[1][0]
                points_3 = self.transformer_ups3[i](xyz_3, points_3)[0]
                xyz_3_temp1 = xyz_3
                points_3_temp1 = points_3
            elif i == 1:
                points_temp = points_4 + xyz_and_feats[0][1]
                xyz_temp = xyz_4 + xyz_and_feats[0][0]
                points_3 = self.transition_ups3[i](xyz_3, points_3, xyz_temp, points_temp)
                xyz_3 = xyz_and_feats[0][0]
                points_3 = self.transformer_ups3[i](xyz_3, points_3)[0]
        version -= 1

        """upSample2"""
        nblocks_now = self.nblocks - version
        xyz_2 = xyz_and_feats[3][0]
        points_2 = xyz_and_feats[3][1]
        points_2 = self.transformer2(xyz_2, self.fc3(points_2))[0]
        for i in range(nblocks_now):
            if i == 0:
                points_2 = self.transition_ups2[i](xyz_2, points_2, xyz_and_feats[2][0], xyz_and_feats[2][1])
                xyz_2 = xyz_and_feats[2][0]
                points_2 = self.transformer_ups2[i](xyz_2, points_2)[0]
                xyz_2_temp1 = xyz_2
                points_2_temp1 = points_2
            elif i == 1:
                points_temp = points_3_temp1 + xyz_and_feats[1][1]
                xyz_temp =  xyz_3_temp1 + xyz_and_feats[1][0]
                points_2 = self.transition_ups2[i](xyz_2, points_2, xyz_temp, points_temp)
                xyz_2 = xyz_and_feats[1][0]
                points_2 = self.transformer_ups2[i](xyz_2, points_2)[0]
                xyz_2_temp2 = xyz_2
                points_2_temp2 = points_2
            elif i == 2:
                points_temp = points_3 + xyz_and_feats[0][1]
                xyz_temp = xyz_3 + xyz_and_feats[0][0]
                points_2 = self.transition_ups2[i](xyz_2, points_2, xyz_temp, points_temp)
                xyz_2 = xyz_and_feats[0][0]
                points_2 = self.transformer_ups2[i](xyz_2, points_2)[0]
        version -= 1

        """upSample1"""
        nblocks_now = self.nblocks - version
        xyz_1 = xyz_and_feats[-1][0]
        points_1 = xyz_and_feats[-1][1]
        points_1 = self.transformer1(xyz_1, self.fc2(points_1))[0]
        for i in range(nblocks_now):
            if i == 0:
                points_1 = self.transition_ups1[i](xyz_1, points_1, xyz_and_feats[3][0], xyz_and_feats[3][1])
                xyz_1 = xyz_and_feats[3][0]
                points_1 = self.transformer_ups1[i](xyz_1, points_1)[0]
            elif i == 1:
                points_temp = points_2_temp1 + xyz_and_feats[2][1]
                xyz_temp = xyz_2_temp1 + xyz_and_feats[2][0]
                points_1 = self.transition_ups1[i](xyz_1, points_1, xyz_temp, points_temp)
                xyz_1 = xyz_and_feats[2][0]
                points_1 = self.transformer_ups1[i](xyz_1, points_1)[0]
            elif i == 2:
                points_temp = points_2_temp2 + xyz_and_feats[1][1]
                xyz_temp = xyz_2_temp2 + xyz_and_feats[1][0]
                points_1 = self.transition_ups1[i](xyz_1, points_1, xyz_temp, points_temp)
                xyz_1 = xyz_and_feats[1][0]
                points_1 = self.transformer_ups1[i](xyz_1, points_1)[0]
            elif i == 3:
                points_temp = points_2 + xyz_and_feats[0][1]
                xyz_temp = xyz_2 + xyz_and_feats[0][0]
                points_1 = self.transition_ups1[i](xyz_1, points_1, xyz_temp, points_temp)
                xyz_1 = xyz_and_feats[0][0]
                points_1 = self.transformer_ups1[i](xyz_1, points_1)[0]

        return self.fc6(points_1)





