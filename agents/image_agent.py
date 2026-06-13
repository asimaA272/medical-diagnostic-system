import torch
from torchvision import transforms
from PIL import Image
import pydicom
import numpy as np
import io

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def image_agent(file_bytes: bytes, filename: str):
    """
    Agent 1: Medical scan preprocess karta hai
    DICOM aur normal images dono support karta hai
    """
    try:
        # DICOM file check karo
        if filename.endswith('.dcm'):
            dicom = pydicom.dcmread(io.BytesIO(file_bytes))
            pixel = dicom.pixel_array.astype(float)
            pixel = (pixel / pixel.max() * 255).astype(np.uint8)
            if len(pixel.shape) == 2:
                pixel = np.stack([pixel]*3, axis=-1)
            img = Image.fromarray(pixel)
        else:
            img = Image.open(io.BytesIO(file_bytes)).convert('RGB')

        tensor = transform(img).unsqueeze(0)
        return tensor, img

    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")