import glob
import os

import cv2
import numpy as np
import torch
import csv
import pandas as pd
from torch.utils.data import Dataset


def get_dataset(cfg, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, device='cuda:0'
                 ):
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.name = cfg['dataset']
        self.mode = cfg['mode']
        self.device = device
        self.input_folder = cfg['data']['input_folder']

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        self.label = torch.from_numpy(np.array(self.label))
        label = self.label[index]
        W, H = self.cfg['data']['image_size']['w'], self.cfg['data']['image_size']['h']
        if self.name == 'IR_image_single':
            iR_path = self.ir_paths[index]
            ir_data = cv2.imread(iR_path)
            ir_data = cv2.resize(ir_data, (W, H))
            ir_data = torch.from_numpy(ir_data)
            ir_data = ir_data.to(torch.float32)
            ir_data = ir_data.permute(2, 1, 0)
            return index, ir_data.to(self.device), label.to(self.device)
        elif self.name == 'RGB_image_single':
            color_path = self.color_paths[index]
            color_data = cv2.imread(color_path)
            color_data = cv2.resize(color_data, (W, H))
            color_data = torch.from_numpy(color_data)
            color_data = color_data.to(torch.float32)
            color_data = color_data.permute(2, 1, 0)
            return index, color_data.to(self.device), label.to(self.device)
        elif self.name == 'fusion_image_single':
            iR_path = self.ir_paths[index]
            ir_data = cv2.imread(iR_path)
            ir_data = cv2.resize(ir_data, (W, H))
            ir_data = torch.from_numpy(ir_data)
            ir_data = ir_data.to(torch.float32)
            ir_data = ir_data.permute(2, 1, 0)
            color_path = self.color_paths[index]
            color_data = cv2.imread(color_path)
            color_data = cv2.resize(color_data, (W, H))
            color_data = torch.from_numpy(color_data)
            color_data = color_data.to(torch.float32)
            color_data = color_data.permute(2, 1, 0)
            return index, color_data.to(self.device), ir_data.to(self.device), label.to(self.device)


class single_IR_image(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(single_IR_image, self).__init__(cfg, device)
        self.input_folder = os.path.join(self.input_folder)
        # self.color_paths = sorted(glob.glob(os.path.join(
        #     self.input_folder, 'visible', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.ir_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'thermal', '*.JPG')), key=lambda x: int(os.path.basename(x)[4:-4]))
        self.n_img = len(self.ir_paths)
        self.read_csv()

    def read_csv(self):
        labels = []
        file_path = os.path.join(self.input_folder, 'label.csv')
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                label = row[0]  # Assuming the label is in the first column
                labels.append(int(label))
        self.label = labels


class single_RGB_image(BaseDataset):
    def __init__(self, cfg, device='cuda:0'):
        super(single_RGB_image, self).__init__(cfg, device)
        self.input_folder = os.path.join(self.input_folder)

        self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, 'visible', '*')),
                                  key=lambda x: int(os.path.basename(x).split('.')[0]))
        self.color_paths = [path for path in self.color_paths if path.endswith('.jpg') or path.endswith('.jpeg')]
        # self.color_paths = sorted(glob.glob(os.path.join(
        #     self.input_folder, 'visible','*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        # self.ir_paths = sorted(glob.glob(os.path.join(
        #     self.input_folder, 'thermal', '*.JPG')), key=lambda x: int(os.path.basename(x)[4:-4]))
        self.n_img = len(self.color_paths)
        self.read_csv()

    def read_csv(self):
        labels = []
        file_path = os.path.join(self.input_folder, 'label.csv')
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                label = row[0]  # Assuming the label is in the first column
                labels.append(int(label))
        # print("labels:", labels)
        self.label = labels


dataset_dict = {
    "IR_image_single": single_IR_image,
    "RGB_image_single": single_RGB_image,
    # "IR_image_multi": multi_IR_image,
    # "RGB_image_multi": multi_RGB_image,
    # "fusion_image_multi": multi_fusion_image,
}
