#!/usr/bin/env python

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
#import constant
from torch.autograd import Variable
from PIL import Image
from io import BytesIO
from typing import Optional


classes = [
    "c_0",
    "c_1",
    "c_2",
    "c_3",
    "c_4",
    "c_5",
    "c_6",
    "c_7",
    "c_8",
    "c_9",
    "c_10",
    "c_11",
    "c_12",
    "c_13",
    "c_14",
    "c_15",
    "c_16",
    "c_17",
    "c_18",
    "c_19",
    "c_20",
    "c_21",
    "c_22",
    "c_23",
    "c_24",
    "c_25",
    "c_26",
    "c_27",
    "c_28",
    "c_29",
    "c_30",
    "c_31",
    "c_32",
    "c_33",
    "c_34",
    "c_35",
    "c_36",
    "c_37",
]
NUMBER_OF_CLASSES = len(classes)

# CPU or GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    """Convolutional Neural Network which does the raining."""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 30 * 30, 1024)
        self.fc2 = nn.Linear(1024, NUMBER_OF_CLASSES)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class PlantDiseaseClassifier(object):
    """An interface which classifies the given image."""

    def __init__(self, model_path: str = None):
        """Returns a Classifier instance.
        Args:
          model_path (str):
            Model path that is going to be loaded for classification.
        Returns:
          Classifier"""

        self.img_size = 128
        self.tranform = transforms.Compose(
            [transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()]
        )
        if model_path:
            self.__load_model(path=model_path)
        else:
            print("Provide a path to trained model!")

    def __load_model(self, path: str):
        self.model = CNN()
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.model.eval()

    def __load_image(self, image_path: str):
        image = Image.open(image_path)
        tensor = self.tranform(image).float()
        tensor = Variable(tensor, requires_grad=True)
        tensor = tensor.to(DEVICE)
        return tensor

    def __load_image_from_bytes(self, image_data: bytes):
        image = Image.open(BytesIO(image_data))
        tensor = self.tranform(image).float()
        tensor = Variable(tensor, requires_grad=True)
        tensor = tensor.to(DEVICE)
        return tensor

    def __batch_data(self, tensor):
        return tensor[None, ...]

    def classify(
        self, image_path: Optional[str] = None, image_data: Optional[bytes] = None
    ):
        """Returns a prediction class.
        Args:
          image_path (Optional[str]):
            Image path that is going to be used in classification.
          image_data (Optional[bytes]):
            Image data that is going to be used in classification.
        Returns:
          Prediction class (str)"""

        tensor = None
        if image_path:
            tensor = self.__load_image(image_path=image_path)
        elif image_data:
            tensor = self.__load_image_from_bytes(image_data=image_data)
        if tensor is None:
            raise Exception("Please provide image path or data of your image!")

        output = self.model(self.__batch_data(tensor))
        predicted = torch.argmax(output)
        classes = classes
        prediction_class = classes[int(predicted.item())]
        return prediction_class

