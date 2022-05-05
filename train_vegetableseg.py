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

from data_utils import provider
from data_utils.general import to_categorical
from data_utils.general import vegetable_classes
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


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def main():
    with open('config/vegetableseg.yaml', 'r') as f:
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

    root_train = hydra.utils.to_absolute_path('data/open/train')
    root_test = hydra.utils.to_absolute_path('data/open/test')

    TRAIN_DATASET = MyDataset(root=root_train, npoints=args.num_point, split='train')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=0, drop_last=True)
    TEST_DATASET = MyDataset(root=root_test, npoints=args.num_point, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=0, drop_last=True)

    """model load"""
    args.input_dim = 6 + 1
    args.num_class = 4
    num_category = 1
    num_part = args.num_class
    shutil.copy(hydra.utils.to_absolute_path('models/model.py'), '.')

    # model = getattr(importlib.import_module('models.model'), 'PointTransformerSeg')(args).cuda()
    model = getattr(importlib.import_module('models.model_v2'), 'PTSeg_v2')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load(str(work_dir) + '/best_model.pth')
        # print(checkpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_str("Use Pretrain model")
    except:
        log_str("No exsiting model")
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )685er
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learing_rate, momentum=0.9
        )

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_str('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        # 调整学习率和batchnormal
        lr = max(args.learning_rate * (args.lr_decay **
                                       (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_str('Learing rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum upadated to:%f' % momentum)
        model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
        model = model.train()

        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(
                points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)

            points, label, target = points.float().cuda(
            ), label.long().cuda(), target.long().cuda()
            optimizer.zero_grad()
            # print(np.shape(points))
            # print(np.shape(label))
            seg_pred = model(torch.cat([points, to_categorical(
                label, num_category).repeat(1, points.shape[1], 1)], -1))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(
                correct.item() / (args.batch_size * args.num_point))
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_str('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in vegetable_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in vegetable_classes.keys():
                for label in vegetable_classes[cat]:
                    seg_label_to_cat[label] = cat

            model = model.eval()

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                          smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(
                ), label.long().cuda(), target.long().cuda()
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

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (
                        np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(vegetable_classes[cat]))]
                    for l in vegetable_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - vegetable_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - vegetable_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            for cat in sorted(shape_ious.keys()):
                logger.info('eval mIoU of %s %f' %
                            (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        log_str('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            log_str('Save model...')
            savepath = os.path.join(work_dir, 'best_model.pth')
            log_str('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_str('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_str('Best accuracy is: %.5f' % best_acc)
        log_str('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_str('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1

    shutil.copy(savepath, 'log/PT/vegetabelseg/best_model.pth')


if __name__ == '__main__':
    main()
