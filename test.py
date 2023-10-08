#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-net', type=str, default='vgg16', help='net type')
    # parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-weights', type=str, \
                        default=".\\checkpoint\\vgg16\\Monday_02_October_2023_22h_56m_27s\\vgg16-31-regular.pth", help='the weights file you want to test')
    parser.add_argument('-weights_pred', type=str, \
                        default=".\\checkpoint\\vgg16\\Monday_02_October_2023_22h_56m_27s\\vgg16-69-regular_prd.pth", help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)
    net_pred = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights))
    net_pred.load_state_dict(torch.load(args.weights_pred))
    # print(net)
    net.eval()
    net_pred.eval()


    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    correct_pred_1 = 0.0
    correct_pred_5 = 0.0
    total_pred = 0

    test_correct = 0
    test_pred_correct = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                # print('GPU INFO.....')
                # print(torch.cuda.memory_summary(), end='')

            # 正常训练模型计算
            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)
            labels = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(labels).float()
            #compute top 5
            correct_5 += correct[:, :5].sum()
            #compute top1
            correct_1 += correct[:, :1].sum()

            # 预测模型计算
            output_pred = net_pred(image)
            _, pred_pred = output_pred.topk(5, 1, largest=True, sorted=True)
            labels = label.view(label.size(0), -1).expand_as(pred_pred)
            correct = pred_pred.eq(labels).float()
            #compute top 5
            correct_pred_5 += correct[:, :5].sum()
            #compute top1
            correct_pred_1 += correct[:, :1].sum()

            _, test_preds = output.max(1)
            test_correct += test_preds.eq(label).sum()
            _, train_pred_preds = output_pred.max(1)
            test_pred_correct += train_pred_preds.eq(label).sum()

    # if args.gpu:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')

    lengthoftraindataset = len(cifar100_test_loader.dataset)
    acc_test = test_correct.float() / lengthoftraindataset
    acc_pred_test = test_pred_correct.float() / lengthoftraindataset

    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Test Acc:", acc_test)
    print("Pred Top 1 err: ", 1 - correct_pred_1 / len(cifar100_test_loader.dataset))
    print("Pred Top 5 err: ", 1 - correct_pred_5 / len(cifar100_test_loader.dataset))
    print("Test Acc:", acc_pred_test)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
