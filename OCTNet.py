import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import cv2
import torch
import torch.nn as nn
from src.dataset import Datasetgzb
from src.model import get_torchvision_model
from src.transformation import inference_transformation


class OCTANetwork():

    def __init__(self, model_dir, model_type):
        self.inference_transformation = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
        self.model = get_torchvision_model(model_type, True, 1, 0)
        state_dict = torch.load(model_dir)
        state_dict = state_dict["state"]
        self.model.load_state_dict(state_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess(self, img_name_list):
        test_set = Datasetgzb(img_name_list, 256, self.inference_transformation)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
        return test_loader

    def forword(self, imgPath):
        img_name_list = [imgPath]
        test_loader = self.preprocess(img_name_list)
        with torch.no_grad():
            for step, images in enumerate(test_loader):
                images = images.to(self.device, dtype=torch.float)
                outputs = self.model(images)
                mask = outputs.reshape((256, 256))
                mask = (mask.to("cpu").detach().numpy() > 0.5) * 1
                mask[mask > 0.5] = 255
                mask[mask < 0.5] = 0
                mask = mask.astype(np.uint8)
                return mask


if __name__ == '__main__':
    net = OCTANetwork('./models/Se_resnext50-920eef84.pth', 'Se_resnext50')
    img = r'F:\2_PycharmWorks\OCTAReadProject\imgfiles\33/1.bmp'
    mask = net.forword(img)
    cv2.imwrite(r'F:\2_PycharmWorks\OCTAReadProject\imgfiles/1.jpg', mask)

