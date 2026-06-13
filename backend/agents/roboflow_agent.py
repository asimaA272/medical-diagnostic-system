import requests
import base64

ROBOFLOW_KEY = "7EaCaY9tbGfbCnB0PMAB"

def roboflow_agent(image_bytes: bytes):
    """Roboflow Detection Agent"""
    try:
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        r = requests.post(
            "https://detect.roboflow.com/chest-disease-detection/1",
            params={"api_key": ROBOFLOW_KEY},
            data=image_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15
        )
        result = r.json()
        predictions = result.get('predictions', [])

        return {
            "detections": [
                {
                    "class":      p['class'],
                    "confidence": round(p['confidence'] * 100, 1),
                    "location": {
                        "x": p['x'], "y": p['y'],
                        "width": p['width'], "height": p['height']
                    }
                }
                for p in predictions[:5]
            ],
            "total_found": len(predictions)
        }
    except Exception as e:
        return {"detections": [], "error": str(e)}