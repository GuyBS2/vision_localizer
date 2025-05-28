import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class FeatureExtractor(nn.Module):
    """
    MobileNetV2-based feature extractor.

    Converts a PIL image into a 1280-dimensional L2-normalized feature vector.
    Suitable for visual similarity tasks.
    """

    def __init__(self, device=None):
        """
        Initialize the feature extractor and preprocessing pipeline.

        Args:
            device: 'cuda' or 'cpu'. If None, selects automatically.
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained MobileNetV2 and remove classification head
        backbone = models.mobilenet_v2(pretrained=True)
        self.feature_net = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

        self.to(self.device)
        self.eval()

        # Define image preprocessing steps
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Required size for MobileNetV2
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a PIL image and prepare it as input to the network.

        Args:
            image: Input image (PIL format)

        Returns:
            Preprocessed image as a batch tensor on the correct device.
        """
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized features from a preprocessed image tensor.

        Args:
            image_tensor: Preprocessed image tensor of shape [1, 3, 224, 224]

        Returns:
            Feature vector of shape [1280], normalized and on CPU
        """
        with torch.no_grad():
            features = self.feature_net(image_tensor)           # [1, 1280, 7, 7]
            pooled = self.pool(features).squeeze()              # [1280]
            normed = nn.functional.normalize(pooled, dim=0)     # L2 normalization
        return normed.cpu()
