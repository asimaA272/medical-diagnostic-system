from src.dataset import CLASS_NAMES

DISEASES = CLASS_NAMES

def differential_agent(probabilities):
    """
    Agent 3: Diagnoses confidence ke hisaab se rank karta hai
    """
    ranked = sorted(
        zip(DISEASES, probabilities.tolist()),
        key=lambda x: x[1],
        reverse=True
    )
    return [
        {"disease": disease, "confidence": round(prob * 100, 2)}
        for disease, prob in ranked
    ]