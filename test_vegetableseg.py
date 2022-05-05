import importlib
import logging
import os
import shutil
import time

import hydra
import numpy as np
import torch
import yaml
from tqdm import tqdm

from data_utils.general import to_categorical, vegetable_classes
from data_utils.mydataset import MyDatasetVisual

g_part2color = {'1': [0, 255, 0],
                '2': [0, 0, 255],
                '3': [0, 255, 255],
                '0': [255, 255, 0]}


def create_attr_dict(yaml_config):
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


def add_vote(vote_label_pool, point_idx, pred_label):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for i in range(B):
        for j in range(N):
            vote_label_pool[int(point_idx[i, j]), int(pred_label[i, j])] += 1
    return vote_label_pool


def main():
    vegetable_img_root = 'autodl-tmp/take/test'
    vegetable_target_root = 'autodl-tmp/take/visual/'
    vegetable_config = 'config/vegetableseg.yaml'

    img_root = hydra.utils.to_absolute_path(vegetable_img_root)

    with open(vegetable_config, 'r') as f:
        args = AttrDict(yaml.safe_load(f.read()))
    create_attr_dict(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    work_dir = args.work_dir
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    # 日志记录
    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 创建一个handler用于写入文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = work_dir + '/'
    log_name = log_path + rq + '.log'
    log_file = log_name
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    # 定义输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # 添加handler
    logger.addHandler(fh)

    def log_str(s):
        logger.info(s)
        print(s)

    # data load
    args.input_dim = 6 + 1
    args.num_class = 4
    num_category = 1

    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    TEST_DATASET_VISUAL = MyDatasetVisual(root=img_root, npoints=NUM_POINT, split='test')

    # model load
    shutil.copy(hydra.utils.to_absolute_path('models/model.py'), 'data_utils')

    model = getattr(importlib.import_module('models.model'),
                    'PointTransformerSeg')(args).cuda()
    checkpoint = torch.load(str(args.work_dir) + '/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()

    with torch.no_grad():
        scene_id = TEST_DATASET_VISUAL.file_list
        scene_id = [x[27:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_VISUAL)

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s " % (batch_idx + 1, num_batches, scene_id[batch_idx]))

            print('create file')
            # 保存文件
            file_out = open(os.path.join(vegetable_target_root, scene_id[batch_idx] + '_pred.txt'), 'w')

            print('load data')
            # 获取数据和标签
            whole_scene_data = TEST_DATASET_VISUAL.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_VISUAL.sematic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], args.num_class))

            data_vegetable, label_vegetable, cls, index_vegetable = TEST_DATASET_VISUAL[batch_idx]

            num_blocks = data_vegetable.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE

            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))
            batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))

            for sbatch in tqdm(range(s_batch_num), total=s_batch_num):
                start_index = sbatch * BATCH_SIZE
                end_index = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_index - start_index

                # 重新确定数据
                batch_data[0:real_batch_size, ...] = data_vegetable[start_index:end_index, ...]
                batch_label[0:real_batch_size, ...] = label_vegetable[start_index:end_index, ...]
                batch_point_index[0:real_batch_size, ...] = index_vegetable[start_index:end_index, ...]

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().cuda()

                seg_pred = model(
                    torch.cat([torch_data,
                               to_categorical(torch.Tensor(cls).long().cuda(), num_category).repeat(BATCH_SIZE,
                                                                                                    torch_data.shape[1],
                                                                                                    1)], -1))

                batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                           batch_pred_label[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)
            # save txt
            for i in range(whole_scene_label.shape[0]):
                # color = g_part2color[pred_label[i]]

                file_out.write('%f %f %f %f %f %f %d\n' % (
                    whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], whole_scene_data[i, 3],
                    whole_scene_data[i, 4], whole_scene_data[i, 5], pred_label[i]))

            file_out.close()


if __name__ == '__main__':
    main()
