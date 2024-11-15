import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
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

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, drop_prob=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout(0.2)
        self.stochastic_depth = StochasticDepth(drop_prob)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample:
            residual = self.downsample(x)
        out = self.stochastic_depth(out) + residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map

class DARTSNet(nn.Module):
    def __init__(self, num_classes=2, num_blocks=[3, 4, 6, 3], stochastic_depth_rate=0.1):
        super(DARTSNet, self).__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Calculate drop probs for each layer
        total_blocks = sum(num_blocks)
        drop_probs = [x * stochastic_depth_rate for x in torch.linspace(0, 1, total_blocks)]
        
        self.layer1 = self._make_layer(64, num_blocks[0], drop_probs[:num_blocks[0]])
        self.layer2 = self._make_layer(128, num_blocks[1], drop_probs[num_blocks[0]:sum(num_blocks[:2])], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], drop_probs[sum(num_blocks[:2]):sum(num_blocks[:3])], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], drop_probs[sum(num_blocks[:3]):], stride=2)
        
        # Attention blocks
        self.attention1 = AttentionBlock(64)
        self.attention2 = AttentionBlock(128)
        self.attention3 = AttentionBlock(256)
        self.attention4 = AttentionBlock(512)
        
        # Global average pooling and fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, drop_probs, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample, drop_probs[0]))
        self.in_channels = out_channels
        for drop_prob in drop_probs[1:]:
            layers.append(ResidualBlock(out_channels, out_channels, drop_prob=drop_prob))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.attention1(x)
        x = self.layer2(x)
        x = self.attention2(x)
        x = self.layer3(x)
        x = self.attention3(x)
        x = self.layer4(x)
        x = self.attention4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def get_model(device):
    model = DARTSNet(num_classes=2)
    model.to(device)
    return model

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    # print(model)
    
    # # Test with a sample input
    # sample_input = torch.randn(1, 3, 224, 224).to(device)
    # output = model(sample_input)
    # print(f"Output shape: {output.shape}")