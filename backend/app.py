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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {"diagnosis": "Pneumonia", "confidence": 0.92}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)