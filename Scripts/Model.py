import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block: 
    Adaptively recalibrates channel-wise feature responses by explicitly 
    modeling interdependencies between channels.
    
    Key Features:
    - Global average pooling to capture channel-wise global context
    - Dimensionality reduction and expansion via fully connected layers
    - Sigmoid activation to generate channel-wise scaling factors
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # Global context capture via adaptive average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Channel attention mechanism
        self.fc = nn.Sequential(
            # Dimensionality reduction
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # Dimensionality restoration with sigmoid for scaling
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Batch size, channels, height, width
        b, c, _, _ = x.size()
        
        # Global average pooling and channel-wise statistics
        y = self.avg_pool(x).view(b, c)
        
        # Generate channel-wise attention weights
        y = self.fc(y).view(b, c, 1, 1)
        
        # Recalibrate input features by channel-wise scaling
        return x * y.expand_as(x)

class HybridBlock(nn.Module):
    """
    Hybrid Convolutional Block: 
    Integrates multiple network design paradigms to enhance feature extraction.
    
    Combines:
    - VGG-style convolutions
    - ResNet-style residual connections
    - DenseNet-style dense connections
    - Inception-style multi-scale processing
    - Squeeze-and-Excitation attention mechanism
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(HybridBlock, self).__init__()
        
        # VGG-style standard convolution
        # Captures local spatial features with 3x3 receptive field
        self.vgg_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        
        # ResNet-style residual connection
        # Handles identity mapping or channel/spatial dimension changes
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
        
        # DenseNet-style dense connection
        # Enables feature reuse and alleviates vanishing gradient problem
        self.dense_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        )
        
        # Inception-style multi-scale feature extraction
        # Captures features at different receptive field sizes
        self.inception_1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride)
        self.inception_3x3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1, stride=stride)
        self.inception_5x5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2, stride=stride)
        self.inception_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        )
        
        # Squeeze-and-Excitation for channel-wise attention
        self.se = SEBlock(out_channels)
        
        # Batch normalization and ReLU for feature normalization and non-linearity
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Parallel feature extraction paths
        vgg_out = self.vgg_conv(x)
        res_out = self.res_conv(x)
        dense_out = self.dense_conv(x)
        
        # Inception-style multi-scale feature concatenation
        inception_out = torch.cat([
            self.inception_1x1(x),
            self.inception_3x3(x),
            self.inception_5x5(x),
            self.inception_pool(x)
        ], dim=1)
        
        # Feature fusion via summation
        combined = vgg_out + res_out + dense_out + inception_out
        
        # Channel-wise feature recalibration
        out = self.se(combined)
        
        # Normalization and activation
        out = self.bn(out)
        out = self.relu(out)
        return out

class HybridNet(nn.Module):
    """
    HybridNet: A neural network architecture combining multiple 
    advanced deep learning design principles.
    
    Key Architectural Characteristics:
    - Progressive feature learning
    - Controlled spatial downsampling
    - Adaptive channel expansion
    - Hybrid block design
    """
    def __init__(self, num_classes=2):
        super(HybridNet, self).__init__()
        
        # Initial convolution for raw feature extraction
        # Large kernel size (7x7) for initial broad feature capture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer progression with increasing complexity
        # Each layer strategically increases channel count and reduces spatial dimensions
        self.layer1 = self._make_layer(64, 128, blocks=3, stride=1)    # Initial feature learning
        self.layer2 = self._make_layer(128, 256, blocks=4, stride=2)   # Enhanced feature extraction
        self.layer3 = self._make_layer(256, 512, blocks=6, stride=2)   # Deep feature representation
        self.layer4 = self._make_layer(512, 1024, blocks=3, stride=2)  # Final feature compression
        
        # Global average pooling for spatial dimension reduction
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification layer
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """
        Creates a sequence of HybridBlocks with controlled channel and spatial progression.
        
        Args:
        - in_channels: Input channel count
        - out_channels: Output channel count
        - blocks: Number of HybridBlocks in the layer
        - stride: Spatial downsampling factor
        
        Returns:
        - Sequential container of HybridBlocks
        """
        layers = []
        # First block handles potential channel expansion and spatial reduction
        layers.append(HybridBlock(in_channels, out_channels, stride))
        
        # Subsequent blocks maintain channel count and refine features
        for _ in range(1, blocks):
            layers.append(HybridBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial feature extraction and downsampling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Progressive feature learning through hybrid layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global spatial compression
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Final classification
        x = self.fc(x)
        return x
    
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


def get_model(device):
    """
    Utility function to instantiate and move model to appropriate device.
    
    Args:
    - device: Computation device (CPU/GPU)
    
    Returns:
    - Initialized HybridNet model
    """
    model = HybridNet(num_classes=2)
    model.to(device)
    return model

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = get_model(device)
#     print(model)

