from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch

def get_dataloader(train_bs, test_bs, path='./dataset/cifar10'):
    # load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # create dataset and dataloader
    trainset = CIFAR10(path, transform=transform_train, train=True, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True, num_workers=4)

    testset = CIFAR10(path, transform=transform_test, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=4)
    return trainloader, testloader