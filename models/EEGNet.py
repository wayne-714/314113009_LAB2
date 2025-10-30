import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs
    
    Reference: Lawhern et al., 2018
    """
    
    def __init__(self, activation='elu', dropout_rate=0.25, F1=16, D=2, F2=32, 
                 num_channels=2, num_classes=2, kernel_length=64):
        super(EEGNet, self).__init__()
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # Get activation function
        if activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.01)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Block 1: Temporal Convolution
        # Input: (batch, 1, C, T) where C=2, T=750
        # Output: (batch, F1, C, T)
        self.firstconv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernel_length),
                stride=1,
                padding=(0, kernel_length // 2),
                bias=False
            ),
            nn.BatchNorm2d(F1)
        )
        
        # Block 2: Depthwise Spatial Convolution
        # Depthwise: each input channel has its own set of filters
        # Input: (batch, F1, C, T)
        # Output: (batch, F1*D, 1, T)
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(num_channels, 1),  # Spatial filtering
                stride=1,
                groups=F1,  # Key: Depthwise convolution
                bias=False
            ),
            nn.BatchNorm2d(F1 * D),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=dropout_rate)
        )
        
        # Block 3: Separable Convolution
        # Separable = Depthwise + Pointwise
        # Input: (batch, F1*D, 1, T//4)
        # Output: (batch, F2, 1, T//32)
        self.separableConv = nn.Sequential(
            # Depthwise convolution (spatial)
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, 16),
                stride=1,
                padding=(0, 8),
                groups=F1 * D,  # Key: Depthwise
                bias=False
            ),
            # Pointwise convolution (1x1)
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(F2),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=dropout_rate)
        )
        
        # Classification Layer
        # After all convolutions and pooling: (batch, F2, 1, T//32)
        # For T=750: 750 // 4 // 8 = 23
        # Flatten size = F2 * 1 * 23 = 32 * 23 = 736
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(736, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 1, C, T)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Block 1: Temporal convolution
        x = self.firstconv(x)
        
        # Block 2: Depthwise spatial convolution
        x = self.depthwiseConv(x)
        
        # Block 3: Separable convolution
        x = self.separableConv(x)
        
        # Classification
        x = self.classify(x)
        
        return x


class DeepConvNet(nn.Module):
    """
    DeepConvNet: Deep Convolutional Network for EEG Classification
    
    Reference: Schirrmeister et al., 2017
    """
    
    def __init__(self, activation='elu', dropout_rate=0.5, num_channels=2, num_classes=2):
        super(DeepConvNet, self).__init__()
        
        # Get activation function
        if activation == 'elu':
            act = nn.ELU(alpha=1.0)
        elif activation == 'relu':
            act = nn.ReLU()
        elif activation == 'leakyrelu':
            act = nn.LeakyReLU(0.01)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Layer 1: Temporal + Spatial Convolution
        # Input: (batch, 1, C, T) where C=2, T=750
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), stride=1, bias=True),
            nn.Conv2d(25, 25, kernel_size=(num_channels, 1), stride=1, bias=True),
            nn.BatchNorm2d(25),
            act,
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(p=dropout_rate)
        )
        
        # Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 10), stride=1, bias=True),
            nn.BatchNorm2d(50),
            act,
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(p=dropout_rate)
        )
        
        # Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 10), stride=1, bias=True),
            nn.BatchNorm2d(100),
            act,
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(p=dropout_rate)
        )
        
        # Layer 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 10), stride=1, bias=True),
            nn.BatchNorm2d(200),
            act,
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(p=dropout_rate)
        )
        
        # Calculate flatten size dynamically
        self.flatten_size = self._get_flatten_size(num_channels)
        
        # Fully connected layer
        self.fc = nn.Linear(self.flatten_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_flatten_size(self, num_channels):
        """Calculate flattened feature size."""
        # Simulate forward pass
        x = torch.zeros(1, 1, num_channels, 750)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.view(1, -1).size(1)
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Test EEGNet
    print("=" * 80)
    print("Testing EEGNet")
    print("=" * 80)
    model = EEGNet(activation='elu', dropout_rate=0.25)
    x = torch.randn(8, 1, 2, 750)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test DeepConvNet
    print("\n" + "=" * 80)
    print("Testing DeepConvNet")
    print("=" * 80)
    model = DeepConvNet(activation='elu', dropout_rate=0.5)
    x = torch.randn(8, 1, 2, 750)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")