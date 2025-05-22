from __future__ import print_function
import os
import sys
import logging
import datetime
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import tools
import dataset_class
from model import CNN, get_resnet18, get_resnet34
from loss import get_otdis_loss
from utils import setup_logger

# Ignore warnings
warnings.filterwarnings('ignore')

# Argument parsing
parser = argparse.ArgumentParser(description="Robust Training with Noisy Labels")
parser.add_argument('--n', type=int, default=0, help="Number of runs")
parser.add_argument('--d', type=str, default='output', help="Description")
parser.add_argument('--p', type=int, default=0, help="Print mode")
parser.add_argument('--c', type=int, default=10, help="Number of classes")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--split_percentage', type=float, default=0.9, help="Train-validation split ratio")
parser.add_argument('--result_dir', type=str, default='/output/results_ours_hard/', help="Directory for result logs")
parser.add_argument('--noise_rate', type=float, default=0.3, help="Label corruption rate")
parser.add_argument('--noise_type', type=str, default='symmetric', choices=('symmetric', 'asymmetric'), help="Noise type")
parser.add_argument('--num_gradual', type=int, default=10, help="Gradual drop epochs")
parser.add_argument('--dataset', type=str, default='cifar10', help="Dataset to use")
parser.add_argument('--n_epoch', type=int, default=200, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=128, help="Scale of batch size")
parser.add_argument('--gamma', type=float, default=0.5, help="Negative learning intensity")
parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer type")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--print_freq', type=int, default=100, help="Print frequency")
parser.add_argument('--num_workers', type=int, default=4, help="DataLoader worker count")
parser.add_argument('--epoch_decay_start', type=int, default=80, help="Epoch to start learning rate decay")
parser.add_argument('--model_type', type=str, default='CNN', choices=('CNN', 'ResNet18', 'ResNet34'), help="Type of backbone to use")
parser.add_argument('--fr_type', type=str, default='type_1', help="Forget rate type")
parser.add_argument('--num_iter_per_epoch', type=int, default=400, help="Number of iterations per epoch")
parser.add_argument('--co_lambda', type=float, default=1e-4, help="Co-regularization coefficient")
parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
parser.add_argument('--channel', type=int, default=3, help="Input image channels")
parser.add_argument('--time_step', type=int, default=7, help="Time step interval for reinitialization")
args = parser.parse_args()

# Set device and seeds
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# Hyperparameters
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
MOMENTUM1 = 0.9
MOMENTUM2 = 0.1

# Learning rate schedule
lr_schedule = [LEARNING_RATE] * args.n_epoch
beta1_schedule = [MOMENTUM1] * args.n_epoch
for epoch in range(args.epoch_decay_start, args.n_epoch):
    lr_schedule[epoch] = LEARNING_RATE * (args.n_epoch - epoch) / (args.n_epoch - args.epoch_decay_start)
    beta1_schedule[epoch] = MOMENTUM2

co_lambda_schedule = args.co_lambda * np.linspace(1, 0, args.epoch_decay_start)

# Directory setup
SAVE_PATH = os.path.join(args.result_dir, args.dataset, args.model_type)
os.makedirs(SAVE_PATH, exist_ok=True)

# Learning rate adjustment
def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_schedule[epoch]
        param_group['betas'] = (beta1_schedule[epoch], 0.999)

# Accuracy computation
def accuracy(logits, targets, topk=(1,)):
    pred = logits.softmax(dim=1).topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0).mul_(100.0 / targets.size(0)) for k in topk]

# Load dataset with configuration
def load_data(args):
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'mnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.n_epoch = 200
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = dataset_class.mnist_dataset(True, transform, tools.transform_target, args.dataset, args.noise_type, args.noise_rate, args.split_percentage, args.seed)
        val_dataset = dataset_class.mnist_dataset(False, transform, tools.transform_target, args.dataset, args.noise_type, args.noise_rate, args.split_percentage, args.seed)
        test_dataset = dataset_class.mnist_test_dataset(transform, tools.transform_target)
    elif args.dataset == 'cifar10':
        args.channel = 3
        args.feature_size = 3 * 32 * 32
        args.num_classes = 10
        args.n_epoch = 200
        train_dataset = dataset_class.cifar10_dataset(True, transform, tools.transform_target, args.dataset, args.noise_type, args.noise_rate, args.split_percentage, args.seed)
        val_dataset = dataset_class.cifar10_dataset(False, transform, tools.transform_target, args.dataset, args.noise_type, args.noise_rate, args.split_percentage, args.seed)
        test_dataset = dataset_class.cifar10_test_dataset(transform, tools.transform_target)
    elif args.dataset == 'cifar100':
        args.channel = 3
        args.feature_size = 3 * 32 * 32
        args.num_classes = 100
        args.n_epoch = 200
        train_dataset = dataset_class.cifar100_dataset(True, transform, tools.transform_target, args.dataset, args.noise_type, args.noise_rate, args.split_percentage, args.seed)
        val_dataset = dataset_class.cifar100_dataset(False, transform, tools.transform_target, args.dataset, args.noise_type, args.noise_rate, args.split_percentage, args.seed)
        test_dataset = dataset_class.cifar100_test_dataset(transform, tools.transform_target)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    return train_dataset, val_dataset, test_dataset

# Training function
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2, prev_loss_1, prev_loss_2, sn_1, sn_2, noise_or_not):
    model1.train()
    model2.train()

    total_correct_1 = total_correct_2 = 0
    total_batches = 0
    pure_ratio_1_list = []
    pure_ratio_2_list = []
    updated_loss_1 = []
    updated_loss_2 = []
    updated_idx_1 = []
    updated_idx_2 = []

    for i, (data, labels, indices) in enumerate(train_loader):
        if i >= args.num_iter_per_epoch:
            break

        batch_start = i * BATCH_SIZE
        batch_end = (i + 1) * BATCH_SIZE

        data, labels = data.cuda(), labels.cuda()
        indices = indices.cpu().numpy().transpose()

        logits1 = model1(data)
        logits2 = model2(data)

        acc1, = accuracy(logits1, labels, topk=(1,))
        acc2, = accuracy(logits2, labels, topk=(1,))
        total_correct_1 += acc1
        total_correct_2 += acc2
        total_batches += 1

        if epoch < args.epoch_decay_start:
            loss_1, loss_2, pr1, pr2, idx_1, idx_2, loss1_vals, loss2_vals = get_otdis_loss(
                epoch, prev_loss_1[batch_start:batch_end], prev_loss_2[batch_start:batch_end],
                sn_1[batch_start:batch_end], sn_2[batch_start:batch_end], logits1, logits2,
                labels, indices, noise_or_not, co_lambda_schedule[epoch], 2., args.gamma)
        else:
            loss_1, loss_2, pr1, pr2, idx_1, idx_2, loss1_vals, loss2_vals = get_otdis_loss(
                epoch, prev_loss_1[batch_start:batch_end], prev_loss_2[batch_start:batch_end],
                sn_1[batch_start:batch_end], sn_2[batch_start:batch_end], logits1, logits2,
                labels, indices, noise_or_not, 0., 2., args.gamma)

        updated_loss_1.extend(loss1_vals)
        updated_loss_2.extend(loss2_vals)
        updated_idx_1.extend(np.array(idx_1) + i * BATCH_SIZE)
        updated_idx_2.extend(np.array(idx_2) + i * BATCH_SIZE)
        pure_ratio_1_list.append(100 * pr1)
        pure_ratio_2_list.append(100 * pr2)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        if (i + 1) % args.print_freq == 0:
            avg_pr1 = np.mean(pure_ratio_1_list)
            avg_pr2 = np.mean(pure_ratio_2_list)
            print(f"Epoch [{epoch+1}/{args.n_epoch}], Iter [{i+1}/{len(train_loader)}] Acc1: {acc1:.2f} Acc2: {acc2:.2f} Loss1: {loss_1.item():.4f} Loss2: {loss_2.item():.4f} PR1: {avg_pr1:.2f} PR2: {avg_pr2:.2f}")

    train_acc1 = float(total_correct_1) / total_batches
    train_acc2 = float(total_correct_2) / total_batches

    return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list, updated_loss_1, updated_loss_2, updated_idx_1, updated_idx_2

def evaluate(test_loader, model1, model2):
    model1.eval()
    model2.eval()

    correct_1 = correct_2 = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = data.cuda()
            labels = labels.cuda()

            # Model 1 evaluation
            logits1 = model1(data)
            pred1 = logits1.softmax(dim=1).argmax(dim=1)
            correct_1 += (pred1 == labels).sum().item()

            # Model 2 evaluation
            logits2 = model2(data)
            pred2 = logits2.softmax(dim=1).argmax(dim=1)
            correct_2 += (pred2 == labels).sum().item()

            total_samples += labels.size(0)

    acc1 = 100.0 * correct_1 / total_samples
    acc2 = 100.0 * correct_2 / total_samples
    return acc1, acc2

# Main training and evaluation loop
def main(args):
    run_name = f"{args.dataset}_{args.model_type}_{args.noise_type}_{args.noise_rate}_{args.seed}"
    log_path = os.path.join(SAVE_PATH, f"{run_name}.log")
    logger = setup_logger(log_path)
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(log_path):
        os.rename(log_path, log_path + f".bak-{now}")

    print(args)
    print("Loading dataset...")
    train_set, val_set, test_set = load_data(args)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=args.num_workers, drop_last=True, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=args.num_workers, drop_last=True, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=args.num_workers, drop_last=True, shuffle=False)

    noise_mask = train_set.noise_or_not

    print("Initializing models...")
    # load models
    if args.model_type == 'CNN':
        model1 = CNN(input_channel=args.channel, n_outputs=args.c).cuda()
        model2 = CNN(input_channel=args.channel, n_outputs=args.c).cuda()
    elif args.model_type == 'ResNet18':
        model1 = get_resnet18(input_channel=args.channel, num_classes=args.c).cuda()
        model2 = get_resnet18(input_channel=args.channel, num_classes=args.c).cuda()
    elif args.model_type == 'ResNet34':
        model1 = get_resnet34(input_channel=args.channel, num_classes=args.c).cuda()
        model2 = get_resnet34(input_channel=args.channel, num_classes=args.c).cuda()
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    optimizer1 = optim.Adam(model1.parameters(), lr=LEARNING_RATE)
    optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE)

    logger.info(f"{epoch} {train_acc1:.4f} {train_acc2:.4f} {val_acc1:.4f} {val_acc2:.4f} {test_acc1:.4f} {test_acc2:.4f} {mean_pr1:.2f} {mean_pr2:.2f}")

    val_acc_log = []
    test_acc_log = []
    final_test_acc = []

    prev_loss_1 = np.zeros((len(train_set), 1))
    prev_loss_2 = np.zeros((len(train_set), 1))
    sn_1 = torch.ones((len(train_set), 1))
    sn_2 = torch.ones((len(train_set), 1))

    loop = trange(args.n_epoch, desc="Training Epochs")
    for epoch in loop:
        if epoch % args.time_step == 0:
            prev_loss_1.fill(0)
            prev_loss_2.fill(0)
            sn_1 = torch.ones((len(train_set), 1))
            sn_2 = torch.ones((len(train_set), 1))

        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)

        train_acc1, train_acc2, pr1_list, pr2_list, loss_hist_1, loss_hist_2, idx_hist_1, idx_hist_2 = train(
            train_loader, epoch, model1, optimizer1, model2, optimizer2,
            prev_loss_1, prev_loss_2, sn_1, sn_2, noise_mask)

        val_acc1, val_acc2 = evaluate(val_loader, model1, model2)
        test_acc1, test_acc2 = evaluate(test_loader, model1, model2)

        val_acc_log.append((val_acc1 + val_acc2) / 2)
        test_acc_log.append((test_acc1 + test_acc2) / 2)
        if epoch > args.n_epoch - 10:
            final_test_acc.extend([test_acc1, test_acc2])

        mean_pr1 = np.mean(pr1_list)
        mean_pr2 = np.mean(pr2_list)

        prev_loss_1 = np.concatenate((prev_loss_1, np.array(loss_hist_1).reshape(-1, 1)), axis=1)
        prev_loss_2 = np.concatenate((prev_loss_2, np.array(loss_hist_2).reshape(-1, 1)), axis=1)

        update_sn = lambda sn, idx: sn + torch.from_numpy(np.eye(len(train_set))[idx].sum(axis=0)).reshape(-1, 1)
        sn_1 = update_sn(sn_1, idx_hist_1)
        sn_2 = update_sn(sn_2, idx_hist_2)

        loop.set_postfix({
        'Val1': f"{val_acc1:.2f}%",
        'Val2': f"{val_acc2:.2f}%",
        'Test1': f"{test_acc1:.2f}%",
        'Test2': f"{test_acc2:.2f}%"
        })
        with open(log_path, 'a') as f:
            f.write(f"{epoch} {train_acc1:.4f} {train_acc2:.4f} {val_acc1:.4f} {val_acc2:.4f} {test_acc1:.4f} {test_acc2:.4f} {mean_pr1:.2f} {mean_pr2:.2f}\n")

    best_val_epoch = np.argmax(val_acc_log)
    best_test_acc = test_acc_log[best_val_epoch]
    last_test_avg = np.mean(final_test_acc)
    return best_test_acc, last_test_avg

if __name__ == '__main__':
    best_acc_list = []
    last_acc_list = []
    for i in range(args.n):
        args.seed = i + 1
        args.output_dir = os.path.join('/output', args.d, str(args.noise_rate))
        os.makedirs(args.output_dir, exist_ok=True)
        if args.p == 0:
            log_file = os.path.join(args.output_dir, f"{args.noise_type}_{args.dataset}_{args.seed}.log")
            logger = setup_logger(log_file)
        else:
            logger = setup_logger(None)  
        best_acc, last_acc = main(args)
        best_acc_list.append(best_acc)
        last_acc_list.append(last_acc)

    print('Best Acc:')
    print(np.mean(best_acc_list))
    print(np.std(best_acc_list, ddof=1))
    print('Last Ten Acc:')
    print(np.mean(last_acc_list))
    print(np.std(last_acc_list, ddof=1))