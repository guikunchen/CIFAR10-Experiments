# coding: utf-8
import torch.optim as optim
import torch.nn as nn
import torch
import csv
import time

from models.mobilenet_v2 import mobilenet_v2
from models.mobilenet_v1 import mobilenet_v1
from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.vgg import vgg16
from data.dataloader import get_dataloader
from test import test


num_class = 10
train_bs = 256
test_bs = 128
device = 'cuda:1'
num_epoch = 200
dirs = ['resnet50', 'resnet101']


def train():
    # create dataloader
    trainloader, testloader = get_dataloader(train_bs, test_bs)
    for dir in dirs:
        # create model and criterion
        if dir == 'resnet18':
            model = resnet18(num_classes = num_class)
        elif dir == 'resnet34':
            model = resnet34(num_classes = num_class)
        elif dir == 'resnet50':
            model = resnet50(num_classes = num_class)
        elif dir == 'resnet101':
            model = resnet101(num_classes = num_class)
        # model.load_state_dict(torch.load("weights.pth"))
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        # define variable
        start_time = time.time()
        max_val_acc = 0.
        step = 0
        correct, total = 0, 0
        train_loss, counter = 0, 0
        logging_step = 100
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        log_file_name = './logs/' + dir + '/' + time_str + '.csv'
        model_file_name = './checkpoints/' + dir + '/' + time_str + '.pth'
        # write header
        with open(log_file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "train_loss", "val_loss", "acc", "val_acc"])
        # start training
        for epoch in range(num_epoch):
            epoch_start_time = time.time()

            # update lr
            if epoch == 0:
                optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay = 4e-5, momentum = 0.9)
            elif epoch == 75:
                optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay = 4e-5, momentum = 0.9)
            elif epoch == 150:
                optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay = 4e-5, momentum = 0.9)

            # iteration over all train data
            for (inputs, labels) in trainloader:
                # push data into cuda
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # count acc,loss on trainset
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                train_loss += loss.item()
                counter += 1

                # recording
                if step % logging_step == 0:
                    # shift to eval mode
                    model.eval()
                    # get acc,loss on trainset
                    acc = correct / total
                    train_loss /= counter

                    # test
                    val_loss, val_acc = test(model, testloader, criterion, device)

                    print('iteration %d , epoch %d:  loss: %.4f  val_loss: %.4f  acc: %.4f  val_acc: %.4f'
                        % (step, epoch, train_loss, val_loss, acc, val_acc))

                    # save logs and weights
                    with open(log_file_name, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([step, train_loss, val_loss, acc, val_acc])
                    if val_acc > max_val_acc:
                        torch.save(model.state_dict(), model_file_name)
                        max_val_acc = val_acc

                    # reset counters
                    correct, total = 0, 0
                    train_loss, counter = 0, 0
                    # shift to train mode
                    model.train()

                step += 1
            print("epoch time %.4f min" % ((time.time() - epoch_start_time) / 60))

        end_time = time.time()
        print("total time %.1f h" % ((end_time - start_time) / 3600))


if __name__ == "__main__":
    train()
