from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
import sys
import os

sys.path.append(os.path.dirname(__file__))

from src.model import load_trained_model
from src.dataset import get_val_transforms
from agents.detection_agent import detection_agent
from agents.differential_agent import differential_agent
from agents.evidence_agent import evidence_agent
from agents.report_agent import report_agent
from agents.fda_agent import fda_agent
from agents.claude_agent import claude_agent
from agents.vision_agent import vision_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "final_model_v2.pth")
transform = get_val_transforms()
model = None

def get_model():
    global model
    if model is None:
        model = load_trained_model(MODEL_PATH, device="cpu")
    return model

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "Backend is running!"}

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        m = get_model()
        probs = detection_agent(tensor, m)
        ranked = differential_agent(probs)
        primary_disease = ranked[0]["disease"]
        evidence = evidence_agent(primary_disease)
        report = report_agent(ranked, evidence)
        fda_data = fda_agent(primary_disease)
        claude_data = claude_agent(primary_disease, ranked, evidence)
        vision_data = vision_agent(contents)

        return {
            "success": True,
            "result": {
                "primary_diagnosis": report["primary_diagnosis"],
                "confidence": report["confidence"],
                "differential_diagnoses": ranked,
                "supporting_literature": evidence,
                "formatted_report": report["formatted_report"],
                "fda_drugs": fda_data,
                "ai_analysis": claude_data,
                "vision_analysis": vision_data
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}