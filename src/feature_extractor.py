import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class FeatureExtractor(nn.Module):

    # MobileNetV2-based feature extractor.
    # Outputs a 1280-dim L2-normalized feature vector for each image.

    def __init__(self, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained MobileNetV2 and remove the classification head
        backbone = models.mobilenet_v2(pretrained=True)
        self.feature_net = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.to(self.device)
        self.eval()

        # Image preprocessing pipeline
        # Required input size for MobileNetV2
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:

        # Preprocess a PIL image into a batch-ready tensor.

        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:

        # Run the image tensor through the network and return L2-normalized features.

        with torch.no_grad():
            features = self.feature_net(image_tensor)           # [1, 1280, 7, 7]
            pooled = self.pool(features).squeeze()              # [1280]
            normed = nn.functional.normalize(pooled, dim=0)     # L2 norm
        return normed.cpu()                                     # Always return on CPU for saving
