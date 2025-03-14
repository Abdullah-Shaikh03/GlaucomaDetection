import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block: 
    Adaptively recalibrates channel-wise feature responses by explicitly 
    modeling interdependencies between channels.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class HybridBlock(nn.Module):
    """
    Hybrid Convolutional Block: 
    Integrates multiple network design paradigms to enhance feature extraction.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(HybridBlock, self).__init__()
        self.vgg_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
        self.dense_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        )
        self.inception_1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride)
        self.inception_3x3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1, stride=stride)
        self.inception_5x5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2, stride=stride)
        self.inception_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        )
        self.se = SEBlock(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        vgg_out = self.vgg_conv(x)
        res_out = self.res_conv(x)
        dense_out = self.dense_conv(x)
        inception_out = torch.cat([
            self.inception_1x1(x),
            self.inception_3x3(x),
            self.inception_5x5(x),
            self.inception_pool(x)
        ], dim=1)
        combined = vgg_out + res_out + dense_out + inception_out
        out = self.se(combined)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer_norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out

class HybridNet(nn.Module):
    """
    HybridNet: A neural network architecture combining multiple 
    advanced deep learning design principles.
    """
    def __init__(self, num_classes=2):
        super(HybridNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 128, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, 256, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, 512, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, 1024, blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Stacking Ensemble of Classifiers
        self.fc1 = nn.Linear(1024, num_classes)  # First classifier
        self.fc2 = nn.Linear(1024, num_classes)  # Second classifier
        self.fc3 = nn.Linear(1024, num_classes)  # Third classifier
        
        # Final aggregation layer
        self.ensemble_fc = nn.Linear(num_classes * 3, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Creates a sequence of HybridBlocks."""
        layers = [HybridBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(HybridBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Parallel classifier outputs
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out3 = self.fc3(x)
        
        # Concatenate classifier outputs and pass through ensemble layer
        ensemble_out = torch.cat([out1, out2, out3], dim=1)
        final_out = self.ensemble_fc(ensemble_out)
        return final_out


def get_model(device):
    """Utility function to instantiate and move model to appropriate device."""
    model = HybridNet(num_classes=2)
    model.to(device)
    return model

    
def export_model_params(model, output_file='params.txt'):
    """
    Export model parameters to a text file.
    
    Args:
        model (torch.nn.Module): The PyTorch model to export
        output_file (str, optional): Path to the output text file. Defaults to 'params.txt'.
    """
    with open(output_file, 'w') as f:
        # Write total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        f.write(f"Total number of parameters: {total_params:,}\n\n")
        
        # Write detailed parameter information
        f.write("Model:\n")
        f.write(str(model) + "\n\n")


def count_parameters(model):
    """
    Count the total number of parameters in a PyTorch model.

    Args:
    - model: PyTorch model

    Returns:
    - Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Import pretrained DensNet121
class DenseNet121Model(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(DenseNet121Model, self).__init__()
        self.model = models.densenet121(pretrained=pretrained)
        
        # Modify the classifier for custom number of classes
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


class ResNet50Model(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(ResNet50Model, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)

        # Modify the classifier for custom number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    

class VGG19Model(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(VGG19Model, self).__init__()
        self.model = models.vgg19(pretrained=pretrained)

        # Modify the classifier for custom number of classes
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def get_model(device):
    """
    Utility function to instantiate and move model to appropriate device.
    
    Args:
    - device: Computation device (CPU/GPU)
    
    Returns:
    - Initialized HybridNet model
    """

    # HybridNet
    # model = HybridNet(num_classes=2)

    # DenseNet121
    model = DenseNet121Model(num_classes=2)

    # ResNet50
    # model = ResNet50Model(num_classes=2, pretrained=True)

    # VGG-19
    # model = VGG19Model(num_classes=2, pretrained=True)


    model.to(device)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    print(f"Toatl number of trainable parameters: {count_parameters(model)}")
