import torch

import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, output_size):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False, num_classes=output_size)

    def forward(self, x):
        return self.resnet(x)

    def extract_features(self, x):
        # Disable gradients to save memory and computation
        with torch.no_grad():
            features = self.resnet.conv1(x)
            features = self.resnet.bn1(features)
            features = self.resnet.relu(features)
            features = self.resnet.maxpool(features)

            features = self.resnet.layer1(features)
            features = self.resnet.layer2(features)
            features = self.resnet.layer3(features)
            features = self.resnet.layer4(features)

            features = self.resnet.avgpool(features)
            features = torch.flatten(features, 1)

        return features
    
class ResNet50(nn.Module):
    def __init__(self, output_size):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=False, num_classes=output_size)

    def forward(self, x):
        return self.resnet(x)

    def extract_features(self, x):
        # Disable gradients to save memory and computation
        with torch.no_grad():
            features = self.resnet.conv1(x)
            features = self.resnet.bn1(features)
            features = self.resnet.relu(features)
            features = self.resnet.maxpool(features)

            features = self.resnet.layer1(features)
            features = self.resnet.layer2(features)
            features = self.resnet.layer3(features)
            features = self.resnet.layer4(features)

            features = self.resnet.avgpool(features)
            features = torch.flatten(features, 1)

        return features