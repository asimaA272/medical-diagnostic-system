from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
import os

from agents.image_agent import image_agent
from agents.detection_agent import detection_agent, load_model
from agents.differential_agent import differential_agent
from agents.evidence_agent import evidence_agent
from agents.report_agent import report_agent

app = FastAPI(
    title="Autonomous Medical Imaging Diagnostic System",
    description="AI-powered chest X-ray analysis with 5 agents",
    version="1.0.0"
)

# Static files serve karo
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model load karo
MODEL = load_model("model/final_model.pth")
print("✅ Model loaded!")

@app.get("/", response_class=HTMLResponse)
def home():
   with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # 5 Agents pipeline
        tensor, img     = image_agent(contents, file.filename)
        probs           = detection_agent(tensor, MODEL)
        ranked          = differential_agent(probs)
        evidence        = evidence_agent(ranked[0]['disease'])
        report          = report_agent(ranked, evidence)

        return JSONResponse({
            "success": True,
            "pipeline": {
                "agent1_image": "✅ Scan preprocessed",
                "agent2_detection": "✅ Anomalies detected",
                "agent3_differential": "✅ Diagnoses ranked",
                "agent4_evidence": "✅ Literature cited",
                "agent5_report": "✅ Report generated"
            },
            "result": report
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })

@app.get("/health")
def health():
    return {"status": "running", "model": "ResNet50", "agents": 5}


# ✅ Correct main block
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
