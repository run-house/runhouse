from PIL import Image
from torchvision import transforms


def get_transform():
    transform = transforms.Compose(
        [
            transforms.Resize(
                (28, 28), interpolation=Image.BILINEAR
            ),  # Resize to 28x28 using bilinear interpolation
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # Normalize with mean=0.5, std=0.5 for general purposes
        ]
    )
    return transform
