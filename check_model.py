import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import urllib.request
import numpy as np

CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'No Finding']

# Google se real X-ray download
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Chest_Xray_PA_3-8-2010.png/800px-Chest_Xray_PA_3-8-2010.png"
urllib.request.urlretrieve(url, "temp_xray.png")
print("✅ X-ray image download ho gayi!")

# Model load
model = models.resnet50(weights=None)
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 5))
model.load_state_dict(torch.load('model/final_model.pth', map_location='cpu'))
model.eval()
print("✅ Model load ho gaya!")

# Transform + Predict
t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open("temp_xray.png").convert('RGB')
with torch.no_grad():
    p = torch.softmax(model(t(img).unsqueeze(0)), dim=1)[0]

print("\n=== MODEL RESULT ===")
for i, cls in enumerate(CLASSES):
    bar = '#' * int(p[i] * 40)
    print(f'{cls:<18} {p[i]*100:5.1f}%  {bar}')
print(f"\n==> FINAL DIAGNOSIS: {CLASSES[p.argmax()]} ({p.max()*100:.1f}%)")
