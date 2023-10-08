# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import data_predfunc as dprd

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, get_train_val_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights


UPDATA_PARAMS = 3
WAIT_STPE = 4
PARAM_BIAS_VGG16 = 1
PARAM_WEIGHTS_VGG16 = 0
PARAM_BIAS_RESNET18 = 2
PARAM_FC_BIAS_RESNET18 = 61

LRPRED_EPOCH_THRD = 100
global_acc_time = 0
global_w_count_resnet18 = 0
# 构造GPU训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def paramupdate(model, epoch_times):
    global global_acc_time
    global global_w_count_resnet18
    # 限制预测仅在前LRPRED_EPOCH_THRD epoch可用
    if global_acc_time < WAIT_STPE:
        if LRPRED_EPOCH_THRD > epoch_times > args.warm:
            # 用预测的方式变更
            with torch.no_grad():
                for iter, param in enumerate(model.parameters()):
                    if args.net == 'vgg16':
                        # 获取bias
                        if iter % 2 == PARAM_BIAS_VGG16:
                            dprd.layer_blist[int(iter / 2)][:, global_acc_time] = torch.clone(param.data)

                        # 获取weight
                        if iter % 2 == PARAM_WEIGHTS_VGG16:
                            dprd.layer_wlist[int(iter / 2)][global_acc_time] = torch.clone(param.data)

                    elif args.net == 'resnet18':
                        # 获取bias
                        if iter % 3 == PARAM_BIAS_RESNET18 or iter == PARAM_FC_BIAS_RESNET18:
                            dprd.layer_blist[int(iter / 3)][:, global_acc_time] = torch.clone(param.data)

                        # 获取weight
                        elif iter % 3 != PARAM_BIAS_RESNET18:
                            dprd.layer_wlist[global_w_count_resnet18][global_acc_time] = torch.clone(param.data)
                            global_w_count_resnet18 += 1
                        else:
                            print("你的代码出问题了，正常不应该进入这个else语句")

                    elif args.net == 'googlenet':
                        # 获取bias
                        if iter in dprd.b_position_list:
                            dprd.param_list[iter][:, global_acc_time] = torch.clone(param.data)

                        # 获取weight
                        if iter in dprd.w_position_list:
                            dprd.param_list[iter][global_acc_time] = torch.clone(param.data)

                    else:
                        print("你进入了一个新网络，没有配置的情况下就不应该开启Linear prediction")

            global_w_count_resnet18 = 0
            global_acc_time += 1
            if global_acc_time == UPDATA_PARAMS:
                for iter, param in enumerate(model.parameters()):
                    if args.net == 'vgg16':
                        # 更新预测bias
                        if iter % 2 == PARAM_BIAS_VGG16:
                            new_param_b = dprd.n_step_predict(dprd.layer_blist[int(iter / 2)][:, 0],
                                                              dprd.layer_blist[int(iter / 2)][:, 1],
                                                              dprd.layer_blist[int(iter / 2)][:, 2])
                            new_param_b = new_param_b.to(device)
                            param.data = torch.clone(new_param_b)
                            new_param_b = new_param_b.to('cpu')
                        # 更新预测w
                        if iter % 2 == PARAM_WEIGHTS_VGG16:
                            new_param_w = dprd.n_step_predict(dprd.layer_wlist[int(iter / 2)][0],
                                                              dprd.layer_wlist[int(iter / 2)][1],
                                                              dprd.layer_wlist[int(iter / 2)][2])
                            new_param_w = new_param_w.to(device)
                            param.data = torch.clone(new_param_w)
                            new_param_w = new_param_w.to('cpu')

                    elif args.net == 'resnet18':
                        # 更新预测bias
                        if iter % 3 == PARAM_BIAS_RESNET18 or iter == PARAM_FC_BIAS_RESNET18:
                            new_param_b = dprd.n_step_predict(dprd.layer_blist[int(iter / 3)][:, 0],
                                                              dprd.layer_blist[int(iter / 3)][:, 1],
                                                              dprd.layer_blist[int(iter / 3)][:, 2])
                            new_param_b = new_param_b.to(device)
                            param.data = torch.clone(new_param_b)
                            new_param_b = new_param_b.to('cpu')
                        # 更新预测w
                        elif iter % 3 != PARAM_BIAS_RESNET18:
                            new_param_w = dprd.n_step_predict(dprd.layer_wlist[global_w_count_resnet18][0],
                                                              dprd.layer_wlist[global_w_count_resnet18][1],
                                                              dprd.layer_wlist[global_w_count_resnet18][2])
                            new_param_w = new_param_w.to(device)
                            param.data = torch.clone(new_param_w)
                            new_param_w = new_param_w.to('cpu')
                            global_w_count_resnet18 += 1

                    elif args.net == 'googlenet':
                        # 更新预测bias
                        if iter in dprd.b_position_list:
                            new_param_b = dprd.n_step_predict(dprd.param_list[iter][:, 0],
                                                              dprd.param_list[iter][:, 1],
                                                              dprd.param_list[iter][:, 2])
                            new_param_b = new_param_b.to(device)
                            param.data = torch.clone(new_param_b)
                            new_param_b = new_param_b.to('cpu')

                        # 更新预测w
                        if iter in dprd.w_position_list:
                            new_param_w = dprd.n_step_predict(dprd.param_list[iter][0],
                                                              dprd.param_list[iter][1],
                                                              dprd.param_list[iter][2])
                            new_param_w = new_param_w.to(device)
                            param.data = torch.clone(new_param_w)
                            new_param_w = new_param_w.to('cpu')

                global_acc_time += 1  # 预测轮的数据不进行保存，保存预测轮之后两轮数据在进行one step predictoin

    else:
        global_acc_time = 0   # 预测之后等候一轮再重新开启预测
        global_w_count_resnet18 = 0

def train(epoch):

    start = time.time()
    net.train()
    if args.compare:
        net_lrpred.train()

    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.to(device)
            images = images.to(device)
        # 正常训练
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        # 预测训练
        if args.compare:
            optimizer_lrpred.zero_grad()
            outputs_lrpred = net_lrpred(images)
            loss_lrpred = loss_function_lrpred(outputs_lrpred, labels)
            loss_lrpred.backward()
            optimizer_lrpred.step()
            # 预测训练参数更新
            paramupdate(net_lrpred, epoch)
        else:
            loss_lrpred = loss

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1    # 求总共训练循环次数

        # 添加w和b的梯度范数到tensorboard
        last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        if batch_index % 20 == 0:
            print('Training Epoch: {epoch}. iter:{batch_index} [{trained_samples}/{total_samples}]\
                   \tLoss: {:0.4f}\tLoss_lrpred: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                loss_lrpred.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                batch_index=batch_index+1,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        if args.compare:
            writer.add_scalar('Train/loss_Prediction', loss_lrpred.item(), n_iter)
            writer.add_scalar('Compare/loss—loss_Prediction', loss.item() - loss_lrpred.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()
            if args.compare:
                warmup_scheduler_lrpred.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()
    if args.compare:
        net_lrpred.eval()

    test_loss = 0.0    # cost function error
    correct = 0.0
    if args.compare:
        lrpred_test_loss = 0.0  # cost function error
        lrpred_correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.to(device)
            labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)
        # 计算误差和精度
        test_loss += loss.item()

        if args.compare:
            outputs_lrpred = net_lrpred(images)
            loss_lrpred = loss_function_lrpred(outputs_lrpred, labels)
            lrpred_test_loss += loss_lrpred.item()
        else:
            lrpred_test_loss = test_loss


        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        if args.compare:
            _, lrpreds = outputs_lrpred.max(1)
            lrpred_correct += lrpreds.eq(labels).sum()
        else:
            lrpred_correct = correct

    finish = time.time()

    # 求平均误差和精度并显示
    lengthofdataset = len(cifar100_test_loader.dataset)
    avg_loss = test_loss / lengthofdataset
    lrpred_avg_loss = lrpred_test_loss / lengthofdataset
    acc_test = correct.float() / lengthofdataset
    lrpred_acc_test = lrpred_correct.float() / lengthofdataset

    # 暂时频闭GPU信息
    # if args.gpu:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, lrpred Average loss: {:.4f}, \
           Accuracy: {:.4f}, lrpred Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch, avg_loss, lrpred_avg_loss, acc_test, lrpred_acc_test, finish - start)
    )
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', avg_loss, epoch)
        writer.add_scalar('Test/Accuracy', acc_test, epoch)
        if args.compare:
            writer.add_scalar('Test_lrpred/Average loss', lrpred_avg_loss, epoch)
            writer.add_scalar('Test_lrpred/Accuracy', lrpred_acc_test, epoch)
            writer.add_scalar('Test Compare/AvgLoss(pred-norm)', lrpred_avg_loss-avg_loss, epoch)
            writer.add_scalar('Test Compare/Acc(pred-norm)', lrpred_acc_test-acc_test, epoch)

    return correct.float() / len(cifar100_test_loader.dataset)    # 返回正确率

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-net', type=str, default='vgg16', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')    # 开启GPU加速
    parser.add_argument('-compare',  default=True, help='compare normal and linear pred model')  # 是否进行两个模型比较

    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.002, help='initial learning rate')
    # 是否要保存模型，以备中断后继续训练
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)
    if args.compare:
        net_lrpred = get_network(args)

    # data preprocessing:
    # cifar100_test_loader实际上是验证集，这里为了减少代码修改影响范围，所以保留test的名字
    cifar100_training_loader, cifar100_test_loader = get_train_val_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    # cifar100_test_loader = get_test_dataloader(
    #     settings.CIFAR100_TRAIN_MEAN,
    #     settings.CIFAR100_TRAIN_STD,
    #     num_workers=4,
    #     batch_size=args.b,
    #     shuffle=False
    # )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.5) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    if args.compare:
        loss_function_lrpred = nn.CrossEntropyLoss()
        optimizer_lrpred = optim.SGD(net_lrpred.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        train_scheduler_lrpred = optim.lr_scheduler.MultiStepLR(optimizer_lrpred, milestones=settings.MILESTONES, gamma=0.5)
        warmup_scheduler_lrpred = WarmUpLR(optimizer_lrpred, iter_per_epoch * args.warm)

    # 得到checkpoint文件夹路径
    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    # 获取checkpoint的.pth路径
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.to(device)
    writer.add_graph(net, input_tensor)

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))    # 获取最佳weights路径
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH+1):
        if epoch > args.warm:
            train_scheduler.step(epoch)
            if args.compare:
                train_scheduler_lrpred.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        # if epoch > settings.MILESTONES[1] and best_acc < acc:
        #     weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
        #     print('saving weights file to {}'.format(weights_path))
        #     torch.save(net.state_dict(), weights_path)
        #     best_acc = acc
        #     continue

        if settings.MILESTONES[2] > epoch > settings.MILESTONES[1]:
            if not epoch % settings.SAVE_EPOCH:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                weights_pred_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular_prd')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
                torch.save(net_lrpred.state_dict(), weights_pred_path)

    writer.close()
