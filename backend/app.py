from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Backend is running!"}

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    return {
        "success": True,
        "result": {
            "primary_diagnosis": "Pneumonia",
            "confidence": "87.3%",
            "differential_diagnoses": [
                {"disease": "Pneumonia", "confidence": 87},
                {"disease": "Pleural Effusion", "confidence": 45},
                {"disease": "Cardiomegaly", "confidence": 23},
                {"disease": "No Finding", "confidence": 12}
            ],
            "supporting_literature": [
                {"pmid": "32743479", "year": 2020, "title": "Deep learning for chest X-ray diagnosis"},
                {"pmid": "31959783", "year": 2020, "title": "AI detection of pneumonia in chest radiographs"}
            ],
            "formatted_report": "CHEST X-RAY REPORT\n==================\nPrimary Finding: Pneumonia\nConfidence: 87.3%\nRecommendation: Immediate medical attention required."
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)