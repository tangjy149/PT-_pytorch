import os

import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils.pointnet_util import pc_normalize


class MyDataset(Dataset):
    def __init__(self, root='../data/open', npoints=2500, split='train'):
        self.npoints = npoints
        self.root = root
        self.datapath = []
        self.split = split
        for root, dirs, files in os.walk(self.root):
            temp = np.random.choice(files, size=len(files), replace=True)
            self.file_len = len(files)
            if self.split == 'train':
                for f in np.unique(temp):
                    self.datapath.append(os.path.join(self.root, f))
            elif self.split == 'test':
                for f in list(set(files).difference(set(temp))):
                    self.datapath.append(os.path.join(self.root, f))


    def __getitem__(self, index):
        fn = self.datapath[index]
        # print(fn)
        data = np.loadtxt(fn).astype(np.float32)
        cls = np.array([0]).astype(np.int32)
        point_set = data[:, 0:6]
        point_set[:, 3:6] /= 255  # 对rgb进行处理
        seg = data[:, -1].astype(np.int32)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


class MyDatasetVisual():
    def __init__(self, root, npoints=2048, split='test', stride=0.5, padding=0.01, block_size=1.0):
        self.root = root
        self.npoints = npoints
        self.file_list = []
        self.stride = stride
        self.padding = padding
        self.block_size = block_size

        for root, dirs, files in os.walk(self.root):
            self.file_len = len(files)
            for f in files:
                self.file_list.append(os.path.join(self.root, f))

        self.scene_points_list = []
        self.sematic_labels_list = []
        self.vegetable_coord_min, self.vegetable_coord_max = [], []

        for file in self.file_list:
            file_data = np.loadtxt(file)
            points = file_data[:, :3]
            self.scene_points_list.append(file_data[:, :6])
            self.sematic_labels_list.append(file_data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.vegetable_coord_max.append(coord_max)
            self.vegetable_coord_min.append(coord_min)

        assert len(self.scene_points_list) == len(self.sematic_labels_list)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:, :6]
        points[:, 0:3] = pc_normalize(points[:, 0:3])
        points[:, 3:6] /= 255.0
        labels = self.sematic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_vegetable, label_vegetable, index_vegetable = np.array([]), np.array([]), np.array([])
        cls = np.array([0]).astype(np.int32)
        # print(1)
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (
                            points[:, 1] >= s_y - self.padding) & (points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(len(points) / self.npoints)
                point_size = int(num_batch * self.npoints)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))  # 选取差集进行concat
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                # data_batch[:, 0:3] = pc_normalize(data_batch[:, 0:3])
                # data_batch[:, 3:6] /= 255.0
                label_batch = labels[point_idxs].astype(int)

                data_vegetable = np.vstack([data_vegetable, data_batch]) if data_vegetable.size else data_batch
                label_vegetable = np.hstack([label_vegetable, label_batch]) if label_vegetable.size else label_batch
                index_vegetable = np.hstack([index_vegetable, point_idxs]) if index_vegetable.size else point_idxs
        # print(2)
        data_vegetable = data_vegetable.reshape((-1, self.npoints, data_vegetable.shape[1]))
        label_vegetable = label_vegetable.reshape((-1, self.npoints))
        index_vegetable = index_vegetable.reshape((-1, self.npoints))

        return data_vegetable, label_vegetable, cls, index_vegetable


    def __len__(self):
        return len(self.scene_points_list)


if __name__ == '__main__':
    data = MyDataset('../data/open', split='train')
    dataLoader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=False, drop_last=True)
    for point_set, cls, seg in dataLoader:
        print(point_set.shape)
        print(cls.shape)
