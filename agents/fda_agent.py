import requests

def fda_agent(diagnosis: str):
    """FDA OpenFDA API - Completely FREE"""
    try:
        r = requests.get(
    "https://api.fda.gov/drug/label.json",
    params={
        'search': f'indications_and_usage:"{diagnosis}"',
        'limit': 3
    },
    timeout=30  # 10 se 30 karo
 )
        
        data = r.json()
        results = []
        if 'results' in data:
            for item in data['results'][:3]:
                results.append({
                    'drug_name': item.get('openfda', {}).get(
                        'brand_name', ['N/A'])[0],
                    'generic_name': item.get('openfda', {}).get(
                        'generic_name', ['N/A'])[0],
                    'usage': item.get(
                        'indications_and_usage', ['N/A'])[0][:150]
                })
        return results if results else [
            {"drug_name": "No data", "generic_name": "N/A", "usage": "N/A"}
        ]
    except Exception as e:
        return [{"drug_name": "FDA Error", "generic_name": str(e), "usage": "N/A"}]