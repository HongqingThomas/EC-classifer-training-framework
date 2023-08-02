import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader,random_split

import src.models as models
from src.utils.datasets import get_dataset
from src.EC_Func.EC_Classifier_Ball_Tree import EpistemicClassifier
import tqdm
import importlib
from matplotlib import pyplot as plt
from torchvision.models import resnet18

from src.EC_Func.EC_setup import EC

class train_utils(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = self.cfg['mode']
        self.batch_size = self.cfg['train']['batch_size']
        self.train_ratio = self.cfg['train']['train_ratio']
        self.num_epochs = self.cfg['train']['num_epochs']
        self.pretrained = self.cfg['train']['pretrain']
        self.lr = self.cfg['train']['lr']
        self.opt = self.cfg['train']['opt']
        self.momentum = self.cfg['train']['momentum']
        self.weight_decay = self.cfg['train']['weight_decay']
        self.lr_scheduler = self.cfg['train']['lr_scheduler']
        self.gamma = self.cfg['train']['gamma']


    def setup(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        self.dataset = get_dataset(cfg=self.cfg, device=self.device)
        self.train_size = int(len(self.dataset) * self.train_ratio)
        self.test_size = len(self.dataset) - self.train_size
        self.train_dataset, self.test_dataset = random_split(self.dataset, [self.train_size, self.test_size])
        self.train_dataloaders = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_dataloaders = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # define model
        if self.mode == 'IR' or self.mode == 'visible':
            # self.net = resnet18(num_classes=2)
            self.net = getattr(models, 'CNN')(self.cfg)
            self.net = self.net.to(self.device)
            parameter_list = [{"params": self.net.parameters(), "lr": self.lr}]
            # for fine-tune
            # parameter_list = filter(lambda p: p.requires_grad, net.parameters())
        # elif self.mode == 'visible':
        # else:

        if self.opt == 'sgd':
            self.optimizer = optim.SGD(parameter_list, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.opt == 'adam':
            self.optimizer = optim.Adam(parameter_list, lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.gamma)

        self.criterion = nn.CrossEntropyLoss()


    def train(self):
        self.loss_list = []
        self.acc_list = []
        for epoch in range(self.num_epochs):
            # if epoch%10 == 0:
            print("epoch:", epoch)
            self.net.train()
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, self.num_epochs - 1) + '-'*5)
            for i, data in enumerate(self.train_dataloaders):
                labels = data[2]
                img = data[1]

                # print("train data:", img.shape)

                self.optimizer.zero_grad()

                outputs = self.net(img)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            self.test()
            torch.save(self.net.state_dict(), 'saved_models/RGB_model.pth')

        # plot
        plt.subplot(1, 2, 1)
        plt.plot(torch.range(1, len(self.loss_list)), self.loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('train loss')

        plt.subplot(1, 2, 2)
        plt.plot(torch.range(1, len(self.acc_list)), self.acc_list)
        plt.xlabel('Iteration')
        plt.ylabel('acc')
        plt.title('accuracy')

        # plt.savefig('./single_object_rgb_cnn.png')
        # plt.show()


    def test(self):
        # self.net.eval()
        correct = 0
        total = 0
        avg_test_loss = 0.0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
          for i, data in enumerate(self.test_dataloaders):
              labels = data[2]
              img = data[1]

              outputs = self.net(img)

              # calc acc
              predicted = torch.max(outputs, 1)[1]
              # if i % 10 == 0:
              #     # print("outputs:", outputs)
              #     print("predicted:", predicted)
              #     print("labels:", labels)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()

              # loss
              avg_test_loss += self.criterion(outputs, labels) / len(self.test_dataloaders)
        self.loss_list.append(avg_test_loss.detach().to("cpu").numpy())
        self.acc_list.append(correct / total)
        print('TESTING:')
        print(f'Accuracy: {100 * correct / total:.2f} %')
        print(f'Average loss: {avg_test_loss:.3f}')


    def EC_classifier(self):
        # self.EC = EpistemicClassifier(self.net, self.layer_interest, metric='minkowski',p=2)
        # self.EC.fit(self.train_dataset[0], self.train_dataset[1])
        self.EC = EC(self.cfg, self)
        self.EC.EC_fit()
        pass
