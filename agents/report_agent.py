from datetime import datetime

def report_agent(ranked_diagnoses: list, evidence: list):
    primary = ranked_diagnoses[0]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    disease = primary['disease']
    if disease == 'No Finding':
        display_name = 'Normal - Koi Bimari Nahi Mili'
    else:
        display_name = disease

    report = {
        "report_id": f"AMIDS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": now,
        "primary_diagnosis": display_name,
        "confidence": f"{primary['confidence']}%",
        "differential_diagnoses": ranked_diagnoses,
        "supporting_literature": evidence,
        "formatted_report": f"""
╔══════════════════════════════════════════════════╗
║      AUTONOMOUS MEDICAL IMAGING REPORT          ║
║      Generated: {now}            ║
╠══════════════════════════════════════════════════╣
║  PRIMARY DIAGNOSIS : {display_name:<29}║
║  CONFIDENCE        : {str(primary['confidence'])+'%':<29}║
╠══════════════════════════════════════════════════╣
║  DIFFERENTIAL DIAGNOSES:                        ║
║  1. {ranked_diagnoses[0]['disease']:<20} {str(ranked_diagnoses[0]['confidence'])+'%':<25}║
║  2. {ranked_diagnoses[1]['disease']:<20} {str(ranked_diagnoses[1]['confidence'])+'%':<25}║
║  3. {ranked_diagnoses[2]['disease']:<20} {str(ranked_diagnoses[2]['confidence'])+'%':<25}║
╠══════════════════════════════════════════════════╣
║  SUPPORTING LITERATURE:                         ║
║  {evidence[0]['title'][:48]:<48}║
╠══════════════════════════════════════════════════╣
║  ⚠ AI-assisted report. Must be confirmed by    ║
║    a licensed radiologist before clinical use.  ║
╚══════════════════════════════════════════════════╝
        """
    }
    return report