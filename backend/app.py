from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io

app = FastAPI()

model = torch.load("model/final_model_v2pth", map_location="cpu")
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    # yahan tum apna preprocessing aur prediction code likho
    result = {"diagnosis": "Pneumonia", "confidence": 0.92}
    return result
