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
import torch.utils.data
import torch.utils.data.distributed

from gragh import My_Graph


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
parser.add_argument('--dist-url', default='tcp://127.0.0.1:33014', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--topo', default=2, type=int, metavar='N',
                    help='Netwok connect topo')


class CIFARLoader():
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
    输入网络拓扑的邻接矩阵
    """
    matrix = np.array([[1 / 3, 0, 0, 1 / 3, 0, 0, 1 / 3, 0],
                       [0, 1 / 3, 0, 0, 1 / 3, 0, 1 / 3, 0],
                       [0, 0, 1 / 3, 0, 1 / 3, 1 / 3, 0, 0],
                       [0, 1 / 3, 0, 1 / 3, 0, 1 / 3, 0, 0],
                       [1 / 3, 0, 1 / 3, 0, 1 / 3, 0, 0, 0],
                       [0, 0, 0, 1 / 3, 0, 1 / 3, 0, 1 / 3],
                       [0, 1 / 3, 0, 0, 0, 0, 1 / 3, 1 / 3],
                       [1 / 3, 0, 1 / 3, 0, 0, 0, 0, 1 / 3]])

    Weight_matrix = torch.from_numpy(matrix)
    graph = My_Graph(rank=gpu, world_size=dist.get_world_size(), weight_matrix=matrix)
    out_edges, in_edges = graph.get_edges()

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
    定义SGP算法的各个发送接收缓冲区
    """
    send_buffer = []
    grad_send_buffer = []
    tracking_grad = []
    last_self_grad = []
    for p in model.parameters():
        cp = p.clone().detach_()
        cp = cp.cuda(gpu)
        send_buffer.append(cp)
        grad_send_buffer.append(cp)
        tracking_grad.append(cp)
        last_self_grad.append(cp)

    in_msg = torch.cat([flatten_tensors(send_buffer), flatten_tensors(tracking_grad)])
    placeholder = torch.cat([flatten_tensors(send_buffer), flatten_tensors(tracking_grad)])

    """
    进入循环训练
    """
    first_step = 0
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

            """
            利用收到的yk之和，加上当前梯度值，减去上一次的梯度值，得到新的tracking_grad(第一次迭代例外)
            """
            if first_step == 0:
                first_step = 1
                for p, tracking_grad_elem in zip(model.parameters(), tracking_grad):
                    tracking_grad_elem.data.copy_(p.grad.data)

            else:
                for p, tracking_grad_elem in zip(model.parameters(), tracking_grad):
                    tracking_grad_elem.data.add_(p.grad.data)
                for last_self_grad_elem, tracking_grad_elem in zip(last_self_grad, tracking_grad):
                    tracking_grad_elem.data.add_(-1.0 * last_self_grad_elem)

            """
            将当前的梯度值赋值给last_self_grad
            """
            for p, last_self_grad_elem in zip(model.parameters(), last_self_grad):
                last_self_grad_elem.data.copy_(p.grad.data)

            """
            利用新的tracking_grad进行梯度更新
            """
            for p, r in zip(model.parameters(), tracking_grad):
                p.grad.copy_(r.data)
            optimizer.step()

            """
            准备发送的信息
            """
            for p, send_buffer_elem in zip(model.parameters(), send_buffer):
                send_buffer_elem.data.copy_(p.data)
            out_msg = torch.cat([flatten_tensors(send_buffer), flatten_tensors(tracking_grad)])

            """
            非阻塞发送数据
            """
            for out_edge in out_edges:
                assert gpu == out_edge.src
                weight = Weight_matrix[out_edge.dest, gpu].to(gpu)
                dist.broadcast(tensor=out_msg.mul(weight.type(out_msg.dtype)),
                               src=out_edge.src, group=out_edge.process_group, async_op=True)
            """
            阻塞接收数据
            """
            in_msg.zero_()
            for in_edge in in_edges:
                dist.broadcast(tensor=placeholder, src=in_edge.src, group=in_edge.process_group)
                in_msg.add_(placeholder)

            """
            将接收到的数据拆分成模型参数和梯度
            """
            in_msg_param, in_msg_grad = in_msg.narrow(0, 0, 30730), in_msg.narrow(0, 30730, 30730)

            """
            融合模型参数
            """
            for r, p in zip(unflatten_tensors(in_msg_param, send_buffer), model.parameters()):
                p.data.mul_(Weight_matrix[gpu, gpu].to(gpu).type(p.data.dtype))
                p.data.add_(r)
            """
            融合tracking_grad
            """
            for r, p in zip(unflatten_tensors(in_msg_grad, grad_send_buffer), tracking_grad):
                p.data.mul_(Weight_matrix[gpu, gpu].to(gpu).type(p.data.dtype))
                p.data.add_(r)

        PATH = './Weights_TRKING/GPU{}/Logistic_{}.pth'.format(gpu, epoch)
        torch.save(model.state_dict(), PATH)


def sample_dirichlet_train_data(data_loader, no_participants):
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


def get_average(table):
    sum = torch.zeros_like(table[0])
    for item in table:
        sum += item

    return sum / len(table)


if __name__ == '__main__':
    main()


