import torch
import torch.nn as nn
import torchvision.models as models


# By using CNN
class Encoder(nn.Module):

    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        resnet = model.resnet50(pretrained=True)
        for item in resnet:
            item.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)

    class Decoder(nn.Module):


