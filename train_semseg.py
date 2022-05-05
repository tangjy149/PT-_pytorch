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
from data_utils.dataset import S3DISDataset
from data_utils.general import sem_label_to_cat


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


def main():
    # 参数获取
    with open('config/semseg.yaml', 'r') as f:
        args = AttrDict(yaml.safe_load(f.read()))  # 将semseg中的参数读出
    create_attr_dict(args)

    # 根据获取参数对环境设备进行配置
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
    log_path = work_dir
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

    # 数据加载
    root = hydra.utils.to_absolute_path('data/s3dis')
    NUM_CLASSES = 13
    NUM_POINT = args.num_point
    BATCH_SIZE = args.batch_size

    print("loading train data...")
    TRAIN_DATASET = S3DISDataset(
        split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0,
        transform=None)
    print("loading test data...")
    TEST_DATASET = S3DISDataset(
        split='test', data_root=root, num_point=NUM_POINT, test_area=1, block_size=1.0, sample_rate=1.0, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    # 模型加载
    args.num_class = 0
    args.input_dim = 9
    shutil.copy(hydra.utils.to_absolute_path('models/model.py'), '.')

    model = getattr(importlib.import_module('models.model'),
                    'PointTransformerSeg')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    # checkpoint = torch.load(str(work_dir) + '/best_model.pth')
    # if checkpoint:
    #     print(checkpoint)
    #     start_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     log_str("Use Pretrain model")
    # else:
    #     log_str("No exsiting model")
    #     start_epoch = 0

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
        )
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

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
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
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        model = model.train()

        # learing in one epoch
        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            # points = points.transpose(2, 1)

            # print(np.shape(points))
            # print(np.shape(torch.cat([points, (torch.eye(NUM_CLASSES)).repeat(1, points.shape[1], 1).cuda()], -1)))
            # seg_pred = model(torch.cat([points, (torch.eye(NUM_CLASSES)[1:9].repeat(1, points.shape[1], 1)).reshape(8, points.shape[1], 13).cuda()], -1))
            seg_pred = model(points)
            # seg_pred = model(torch.cat(points, -1))
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss.item()
        log_str('Training mean loss: %f' % (loss_sum / num_batches))
        log_str('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(work_dir) + '/model.pth'
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(state, savepath)
            log_str('Saving model')

        # 评估分割场景
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            label_weights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            model = model.eval()

            log_str('--- EPOCH %03d EVALUATION ---' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                # points = points.transpose(2, 1)

                seg_pred = model(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()  # 开辟连续的内存空间，保证运算
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                # 累计计算损失
                loss = criterion(seg_pred, target)
                loss_sum += loss.item()
                pred_val = np.argmax(pred_val, 2)  # 发挥过往损失值最大的索引
                correct = np.sum((pred_val == batch_label))  # 比对正确率
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                label_weights += tmp

                # 计算总体正确值
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((batch_label == l) & (pred_val == l))  # 交并比的分子
                    total_iou_deno_class[l] += np.sum((batch_label == l) | (pred_val == l))  # 交并比的分母

            label_weights = label_weights.astype(np.float32) / np.sum(label_weights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
            log_str('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_str('eval point avg class IoU: %f' % (mIoU))
            log_str('eval point acc: %f ' % (total_correct / float(total_seen)))
            log_str('eval point avg class acc: %f ' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

            # 输出每一种类别的IoU
            iou_per_class_str = '------IoU------\n'
            for i in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    sem_label_to_cat[i] + ' ' * (14 - len(sem_label_to_cat[i])), label_weights[i - 1],
                    total_correct_class[i] / float(total_iou_deno_class[i]))

            log_str(iou_per_class_str)
            log_str('Eval mean loss: %f ' % (loss_sum / num_batches))
            log_str('Eval acc : %f ' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                print('Save model...')
                savepath = str(work_dir) + '/best_model.pth'
                log_str('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_str('Saving model....')
            log_str('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    main()
