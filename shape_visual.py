import importlib
import os
import shutil

import hydra
import numpy as np
import torch
import yaml
from tqdm import tqdm

from data_utils.general import to_categorical, vegetable_classes
from data_utils.mydataset import MyDataset


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


class Generate_txt_and_img:
    """
    生成预测文件txt和可视化图像img
    获取数据路径，加载模型，生成txt和img
    """

    def __init__(self, data_root, target_root, num_classes, model_dict, testDataLoader, color_map=None):
        """
        初始类数据
        :param data_root:点云数据路径
        :param target_root: 生成数据路径
        :param num_classes: 大类
        :param model_dict: 模型字典权重
        :param testDataloader: 数据读取
        :param color_map: 颜色映射
        :return: None
        """
        self.data_root = data_root
        self.target_root = target_root
        self.testDataLoader = testDataLoader
        self.num_classes = num_classes
        self.color_map = color_map
        self.heat_map = False  # 控制是否输出heatmap
        self.label_path_txt = os.path.join(self.target_root, 'label_txt')  # 存放label的txt文件
        self.make_dir(self.label_path_txt)

        # 加载权重
        self.model_name = []
        self.model = []
        self.model_weight_path = []

        for k, v in model_dict.items():
            self.model_name.append(k)
            self.model.append(v[0])
            self.model_weight_path.append(v[1])

        self.load_checkpoint_for_models(self.model_name, self.model, self.model_weight_path)

        # mkdir
        self.all_pred_image_path = []
        self.all_pred_txt_path = []
        for n in self.model_name:
            self.make_dir(os.path.join(self.target_root, n + '_predict_txt'))
            self.make_dir(os.path.join(self.target_root, n + '_predict_image'))
            self.all_pred_txt_path.append(os.path.join(self.target_root, n + '_predict_txt'))
            self.all_pred_image_path.append(os.path.join(self.target_root, n + '_predict_image'))

        self.generate_predict_to_txt()  # 生成预测的txt

    def make_dir(self, root):
        if os.path.exists(root):
            print('has exsited')
        else:
            os.mkdir(root)

    # shapenet seg_classes vegetable vegetable_classes
    def generate_predict_to_txt(self):
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in vegetable_classes.keys():
            for label in vegetable_classes[cat]:
                seg_label_to_cat[label] = cat

        for batch_id, (points, label, target) in tqdm(enumerate(self.testDataLoader),
                                                      total=len(self.testDataLoader),
                                                      smoothing=0.9):
            for n, model, pred_path in zip(self.model_name, self.model, self.all_pred_txt_path):

                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                xyz_points = points[:, :, :6].cpu()

                seg_pred = model(torch.cat([points, to_categorical(
                    label, num_category).repeat(1, points.shape[1], 1)], -1))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros(
                    (cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(
                        logits[:, vegetable_classes[cat]], 1) + vegetable_classes[cat][0]

                points_file = np.concatenate([xyz_points, torch.tensor(cur_pred_val[:, :, None])], axis=2).squeeze(0)
                result_path = os.path.join(pred_path, f'{n}_{batch_id}.txt')
                np.savetxt(result_path, points_file, fmt='%.04f')

    def load_checkpoint_for_models(self, name, model, checkpoints):
        assert checkpoints is not None, '权重缺失'
        assert model is not None, '未实例化模型'

        for n, m, c in zip(name, model, checkpoints):
            # print(c)
            weight_dict = torch.load(os.path.join(c))
            m.load_state_dict(weight_dict['model_state_dict'])
            print('load down')


if __name__ == '__main__':
    # 参数管理
    shapenet_img_root = 'autodl-tmp/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    shapenet_target_root = 'autodl-tmp/predict/partseg'
    shapenet_config = 'config/partseg.ymal'

    vegetable_img_root = 'data/open/test/'
    vegetable_target_root = 'data/predict/vegetableseg'
    vegetable_config = 'config/vegetableseg.yaml'

    img_root = hydra.utils.to_absolute_path(vegetable_img_root)

    with open(vegetable_config, 'r') as f:
        args = AttrDict(yaml.safe_load(f.read()))
    create_attr_dict(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # 导入模型

    # shapenet
    # args.input_dim = (6 if args.normal else 3) + 16
    # args.num_class = 50
    # num_category = 16
    # num_part = args.num_class

    # vegetable
    args.input_dim = 6 + 1
    args.num_class = 4
    num_category = 1
    num_part = args.num_class

    shutil.copy(hydra.utils.to_absolute_path('models/model.py'), 'data_utils')

    model = getattr(importlib.import_module('models.model'),
                    'PointTransformerSeg')(args).cuda()
    model = model.eval()

    # shapenet
    # TEST_DATASET = PartNormalDataset(
    #     root=img_root, npoints=args.num_point, split='test', normal_channel=args.normal)
    # testDataLoader = torch.utils.data.DataLoader(
    #     TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

    TEST_DATASET = MyDataset(root=vegetable_img_root, npoints=args.num_point, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0, drop_last=True)
    color_map = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, args.num_class))}

    model_dict = {
        'PT': [model, 'log/PT/vegetableseg/best_model.pth']
    }

    c = Generate_txt_and_img(img_root, vegetable_target_root, args.num_class, model_dict, testDataLoader, color_map)
