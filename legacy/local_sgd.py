"""
==========================
@author:Zhu Zehan
@time:2021/5/11:19:37
@email:12032045@zju.edu.cn
==========================
"""
import argparse
from collections import defaultdict
import random
import numpy as np
import load as cifar

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torchvision
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='PyTorch ImageNet 2012 Training')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=50, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:33069', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--f', default=20, type=int, metavar='N',
                    help='communication internal')


class CIFARLoader():   # 定义数据加载器的类
    def __init__(self):
        self.train_data = cifar.x_train
        self.train_label = cifar.y_train
        self.test_data = cifar.x_test
        self.test_label = cifar.y_test

        self.train_data = self.train_data / 255.0
        self.train_label = self.train_label
        self.test_data = self.test_data / 255.0
        self.test_label = self.test_label


def main():
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    data_loader = CIFARLoader()
    indices_per_participant = sample_dirichlet_train_data(data_loader, 8)
    mp.spawn(main_worker, nprocs=8, args=(indices_per_participant, args))  # 开启8个进程， 每个进程执行main_worker（）函数，且将参数传给该函数


def main_worker(gpu, indices_per_participant, args):
    """
    进程初始化
    """
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=8, rank=gpu)

    """
    模型加载，定义criterion，optimizer优化器,
    """
    model = nn.Linear(3 * 32 * 32, 10)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
    cudnn.benchmark = True

    """
    加载数据集，构建数据集的加载器
    """
    data_loader = CIFARLoader()
    indices = indices_per_participant[gpu]
    random.shuffle(indices)

    """
    进入循环训练
    """
    for epoch in range(120):
        model.train()
        steps = list(range(125))
        random.shuffle(steps)
        for step in steps:
            images = data_loader.train_data[indices[step * 50:(step + 1) * 50]]
            target = data_loader.train_label[indices[step * 50:(step + 1) * 50]]
            torch_images = torch.from_numpy(images).float().cuda(gpu, non_blocking=True)
            torch_target = torch.from_numpy(target).long().cuda(gpu, non_blocking=True)
            torch_images = torch_images.view(-1, 32 * 32 * 3)
            output = model(torch_images)
            loss = criterion(output, torch_target)
            if gpu == 6:
                print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.f == 0:
                sync_models(model)

        PATH = './Weights_local_sgd/GPU{}/Logistic_{}.pth'.format(gpu, epoch)
        torch.save(model.state_dict(), PATH)


def sample_dirichlet_train_data(data_loader, no_participants):   #  定义数据划分
    probabilities = np.array([[625, 625, 625, 625, 625, 625, 625, 625], [625, 625, 625, 625, 625, 625, 625, 625],
                              [625, 625, 625, 625, 625, 625, 625, 625], [625, 625, 625, 625, 625, 625, 625, 625],
                              [625, 625, 625, 625, 625, 625, 625, 625], [625, 625, 625, 625, 625, 625, 625, 625],
                              [625, 625, 625, 625, 625, 625, 625, 625], [625, 625, 625, 625, 625, 625, 625, 625],
                              [625, 625, 625, 625, 625, 625, 625, 625], [625, 625, 625, 625, 625, 625, 625, 625]])

    cifar_classes = {}
    for ind, label in enumerate(data_loader.train_label):
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]
    per_participant_list = defaultdict(list)
    no_classes = len(cifar_classes.keys())
    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = probabilities[n]
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

    return per_participant_list


def flatten_tensors(tensors):
    """
    将高纬张量展平
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    将展平的张量按照tensors的shape恢复成高纬张量
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def sync_models(model):
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.reduce_op.SUM)
        param.data.div_(8.0)


if __name__ == '__main__':
    main()



