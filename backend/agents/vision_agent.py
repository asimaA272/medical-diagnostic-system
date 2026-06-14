import openai
import base64
from PIL import Image
import io
OPENAI_API_KEY = "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"


def vision_agent(image_bytes: bytes):
    """GPT-4o Vision — Real API"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """You are a professional radiologist. 
                        Analyze this chest X-ray and provide:
                        1. Main findings
                        2. Any abnormalities detected
                        3. Severity assessment
                        4. Professional recommendations
                        Be concise and professional."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }],
            max_tokens=500
        )
        
        return {
            "vision_findings": response.choices[0].message.content,
            "powered_by": "GPT-4o Vision — Real API"
        }
        
    except Exception as e:
        return {
            "vision_findings": f"Error: {str(e)}",
            "powered_by": "GPT-4o Vision"
        }