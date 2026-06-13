from groq import Groq

GROQ_API_KEY = "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"

def claude_agent(diagnosis: str, ranked: list, evidence: list):
    """Claude 3.5 Style — Groq API FREE"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        prompt = f"""You are a medical AI assistant analyzing chest X-ray results.

Primary Diagnosis: {diagnosis}
Confidence Scores:
{chr(10).join([f"- {r['disease']}: {r['confidence']}%" for r in ranked])}

PubMed Evidence Found: {len(evidence)} papers

Please provide:
1. Clinical interpretation
2. Recommended next steps
3. Risk assessment
4. Treatment considerations"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert medical AI assistant specializing in radiology."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500
        )
        
        return {
            "claude_analysis": {
                "primary_finding": diagnosis,
                "clinical_response": response.choices[0].message.content,
                "powered_by": "Groq LLaMA (Claude 3.5 Compatible)"
            }
        }
        
    except Exception as e:
        return {
            "claude_analysis": {
                "primary_finding": diagnosis,
                "error": str(e),
                "powered_by": "Groq API"
            }
        }