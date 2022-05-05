import importlib
import logging
import os
from pathlib import Path
import shutil

import hydra
import numpy as np
import torch
from tqdm import tqdm
import yaml

from data_utils.dataset import ScannetDatasetWholeScene
from data_utils.general import sem_label_to_cat
from data_utils.indoor3d_util import g_class2color


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
    """
    封装字典操作，包括返回和设置
    :param dict:
    :return: 获取则返回值，设置则对字典进行操作
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    """
    实现一个投票器，从而确定权重
    :param vote_label_pool: 投票箱，存储索引和label对应结果
    :param point_idx: 索引
    :param pred_label: 标签
    :param weight: 权重
    :return: 返回总体投票箱
    """
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for i in range(B):
        for j in range(N):
            if weight[i, j] != 0 and not np.isinf(weight[i, j]):
                vote_label_pool[int(point_idx[i, j]), int(pred_label[i, j])] += 1
    return vote_label_pool


def main():
    # 参数获取
    with open('config/test_semseg.yaml', 'r') as f:
        args = AttrDict(yaml.safe_load(f.read()))
    create_attr_dict(args)

    # 环境配置
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    work_dir = args.work_dir
    visual_dir = Path(work_dir + '/visual/')
    visual_dir.mkdir(exist_ok=True)
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    # 日志
    log_file = os.path.join(work_dir, 'log.txt')
    logger = logging.getLogger("model")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    print(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)

    def log_str(s):
        logger.info(s)
        print(s)

    # data load
    root = hydra.utils.to_absolute_path('data/s3dis')
    NUM_CLASSES = 13
    NUM_POINT = args.num_point
    BATCH_SIZE = args.batch_size

    TEST_DATESET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area,
                                                        block_points=NUM_POINT)
    log_str("the number of test data is : %d " % len(TEST_DATESET_WHOLE_SCENE))

    # model load
    args.num_class = 13
    args.input_dim = 9
    shutil.copy(hydra.utils.to_absolute_path('models/model.py'), '.')

    MODEL = getattr(importlib.import_module('models.model'),
                    'PointTransformerSeg')(args).cuda()
    checkpoint = torch.load(str(work_dir) + '/best_model.pth')
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL = MODEL.eval()

    with torch.no_grad():
        scene_id = TEST_DATESET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]  # 去除末尾四位
        num_batches = len(TEST_DATESET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_str("evaluation whole scene")

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s " % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            # 保存可视化文件
            if args.visual:
                file_out = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                file_out_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            # 获取场景数据和标签
            whole_scene_data = TEST_DATESET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATESET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_samplepoint_weights, scene_point_index = TEST_DATESET_WHOLE_SCENE[
                    batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))
                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_samplepoint_weights = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    # 获取每个batch的首尾索引
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx

                    # 根据新确定的batchsize大小 将数据进行输入
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_samplepoint_weights[0:real_batch_size, ...] = scene_samplepoint_weights[start_idx:end_idx,
                                                                        ...]
                    batch_data[:, :, 3:6] /= 1.0  # rgb?

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    # torch_data = torch_data.transpose(2, 1)
                    seg_pred = MODEL(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_samplepoint_weights[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)

            # evaluation
            # 整合数据
            for cls in range(NUM_CLASSES):
                total_seen_class_tmp[cls] += np.sum((whole_scene_label == cls))
                total_correct_class_tmp[cls] += np.sum((pred_label == cls) & (whole_scene_label == cls))
                total_iou_deno_class_tmp[cls] += np.sum((pred_label == cls) | (whole_scene_label == cls))
                total_seen_class[cls] += total_seen_class_tmp[cls]
                total_correct_class[cls] += total_correct_class_tmp[cls]
                total_iou_deno_class[cls] += total_iou_deno_class_tmp[cls]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_str('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))

            # save test result
            file_name = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(file_name, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()

            for i in range(whole_scene_label.shape[0]):
                color = g_class2color[pred_label[i]]  # 查找对应的物体rgb
                color_gt = g_class2color[whole_scene_label[i]]

                if args.visual:
                    file_out.write('v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                        color[2]))
                    file_out_gt.write('v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                        color_gt[1],
                        color_gt[2]))
            if args.visual:
                file_out.close()
                file_out_gt.close()

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
        iou_per_class_str = '------IoU------\n'
        for i in range(NUM_CLASSES):
            iou_per_class_str += 'class %s , IoU: %.3f \n' % (
                sem_label_to_cat[i] + ' ' * (14 - len(sem_label_to_cat[i])),
                total_correct_class[i] / float(total_iou_deno_class[i]))

        log_str(iou_per_class_str)
        log_str('Eval point avg class IoU: %f ' % np.mean(IoU))
        log_str('Eval whole scene point avg class acc : %f ' % (
            np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float) + 1e-6)))
        log_str(
            'Eval whole scene point acc: %f ' % (np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

if __name__ == '__main__':
    main()