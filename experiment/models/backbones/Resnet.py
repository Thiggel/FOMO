import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, model_type, output_size):
        super(ResNet, self).__init__()
        model_dict = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
        }
        self.resnet = model_dict[model_type](pretrained=False, num_classes=output_size)
        self.num_features = self.resnet.fc.in_features

    def forward(self, x):
        return self.resnet(x)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def extract_features(self, x):
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


class ResNet18(ResNet):
    def __init__(self, output_size):
        super().__init__("resnet18", output_size)


class ResNet50(ResNet):
    def __init__(self, output_size):
        super().__init__("resnet50", output_size)


class ResNet101(ResNet):
    def __init__(self, output_size):
        super().__init__("resnet101", output_size)
