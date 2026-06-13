import requests

def evidence_agent(diagnosis: str):
    """
    Agent 4: PubMed FREE API se medical literature cite karta hai
    """
    try:
        # Search
        search = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                'db': 'pubmed',
                'term': f"chest xray {diagnosis} diagnosis",
                'retmax': 3,
                'retmode': 'json'
            },
            timeout=10
        )
        ids = search.json()['esearchresult']['idlist']

        if not ids:
            return [{"pmid": "N/A", "title": "No references found"}]

        # Details fetch karo
        summary = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={
                'db': 'pubmed',
                'id': ','.join(ids),
                'retmode': 'json'
            },
            timeout=10
        ).json()['result']

        return [
            {
                "pmid": pid,
                "title": summary[pid].get('title', 'N/A'),
                "authors": summary[pid].get('sortfirstauthor', 'N/A'),
                "year": summary[pid].get('pubdate', 'N/A')[:4]
            }
            for pid in ids if pid in summary
        ]

    except Exception as e:
        return [{"pmid": "Error", "title": str(e)}]