import torch
import logging
from torch import nn
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
import argparse
import random

from src.EC_Func.EC_Classifier_Ball_Tree import EpistemicClassifier
from src.EC_Func.EC_setup import EC
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset

import numpy as np
import torch

from src.utils import config
from src.train import train_utils
import src.models as models
from src.utils.datasets import get_dataset

class TestUtils(object):
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def __init__(self, cfg, trainer):
        self.cfg = cfg
        self.mode = self.cfg['mode']
        self.batch_size = self.cfg['test']['batch_size']
        self.test_ratio = self.cfg['test']['test_ratio']
        self.net = trainer.net
        self.k = self.cfg['test']['k-fold cross-validation']
        self.trainer = trainer

    def setup(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # self.cfg['data']['input_folder'] = self.cfg['data']['test_input_folder']

        # self.dataset = get_dataset(cfg=self.cfg, device=self.device)
        self.dataset = self.trainer.train_dataset
        # print("***************************************************************")
        # print("test_dataset:", len(self.dataset))
        # print("self.dataset:", self.dataset)

        # k-fold cross validation
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        fold_indices = kf.split(range(len(self.dataset)))
        self.train_dataloaders_list = []
        self.test_dataloaders_list = []
        for fold_idx, (train_indices, test_indices) in enumerate(fold_indices):
            # Create DataLoader for training and testing subsets for the current fold
            train_subset = Subset(self.dataset, train_indices)
            test_subset = Subset(self.dataset, test_indices)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=True)
            self.train_dataloaders_list.append(train_loader)
            self.test_dataloaders_list.append(test_loader)
        # print(len(self.train_dataloaders_list))

        # self.train_size = int(len(self.dataset) * self.test_ratio)
        # self.test_size = len(self.dataset) - self.train_size
        # self.train_dataset, self.test_dataset = random_split(self.dataset, [self.train_size, self.test_size])
        # self.train_dataloaders = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        # self.test_dataloaders = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # # load model
        # if self.mode == 'visible':
        #     self.net = getattr(models, 'CNN')(self.cfg)
        #     self.net.load_state_dict(torch.load('saved_models/RGB_model.pth'))
        # elif self.mode == 'IR':
        #     self.net = getattr(models, 'IRCNN')(self.cfg)
        #     self.net.load_state_dict(torch.load('saved_models/IR_model.pth'))

        # self.net = self.net.to(self.device)
        self.net.load_state_dict(torch.load('saved_models/RGB_model.pth'))
        self.criterion = nn.CrossEntropyLoss()

        # onnx_name = 'CNN.onnx'
        # dummy_input = torch.randn(16, 3,120,144)
        # torch.onnx.export(self.net, dummy_input, onnx_name, input_names=['input'], output_names=['output'])

        # import tf2onnx
        # import onnxruntime as rt
        #
        # # 加载模型
        # model = ResNet50(weights='imagenet')
        # # 设置输入大小
        # spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
        # # 保存onnx路径
        # output_path = model.name + ".onnx"
        # # 转换
        # tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    def test(self):
        correct = 0
        total = 0
        avg_test_loss = 0.0
        self.net.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.test_dataloaders_list[0]), desc="Testing data"):
                labels = data[2].long()
                img = data[1]

                outputs = self.net(img)

                # calculate accuracy
                predicted = torch.max(outputs, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # calculate loss
                avg_test_loss += self.criterion(outputs, labels) / len(self.test_dataloaders_list[0])
        print("img", img.shape)
        print('TESTING:')
        print(f'Accuracy: {100 * correct / total:.2f} %')
        print(f'Average loss: {avg_test_loss:.3f}')

    def EC_classifier(self):
        # self.EC = EpistemicClassifier(self.net, self.layer_interest, metric='minkowski',p=2)
        # self.EC.fit(self.train_dataset[0], self.train_dataset[1])
        self.EC = EC(self.cfg, self)
        self.EC.EC_fit()
        # stanley modify
        # self.EC.EC_fit_multilayer()

    def EC_validate_setup(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.train_dataset = self.trainer.train_dataset
        self.cfg['data']['input_folder'] = self.cfg['data']['test_input_folder']
        self.test_dataset = get_dataset(cfg=self.cfg, device=self.device)
        self.train_dataloaders = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_dataloaders = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.net.load_state_dict(torch.load('saved_models/RGB_model.pth'))
        self.criterion = nn.CrossEntropyLoss()

    def validate_test(self):
        correct = 0
        total = 0
        avg_test_loss = 0.0
        self.net.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.test_dataloaders), desc="Testing data"):
                labels = data[2].long()
                img = data[1]

                outputs = self.net(img)

                # calculate accuracy
                predicted = torch.max(outputs, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # calculate loss
                avg_test_loss += self.criterion(outputs, labels) / len(self.test_dataloaders_list[0])
        print("img", img.shape)
        print('validate TESTING:')
        print(f'validate Accuracy: {100 * correct / total:.2f} %')
        print(f'validate Average loss: {avg_test_loss:.3f}')

    def EC_classifier_validate(self):
        self.EC = EC(self.cfg, self)
        self.EC.EC_validate_fit()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Arguments for running the RGB IR-Fusion.')
#     parser.add_argument('config', type=str, help='Path to config file.')
#     args = parser.parse_args()
#
#     cfg = config.load_config(args.config)
#     test_utils = TestUtils(cfg)
#     test_utils.setup_seed(42)
#     test_utils.setup()
#     test_utils.test()
#     test_utils.EC_classifier()
