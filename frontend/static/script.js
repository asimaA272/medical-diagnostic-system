// File preview
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file && !file.name.endsWith('.dcm')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById('preview');
            preview.src = e.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

// Main analyze function
async function analyze() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files[0]) {
        alert('Please select an image first!');
        return;
    }

    // Loading show karo
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/diagnose', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.success) {
            const result = data.result;

            // Primary diagnosis
            document.getElementById('primaryDiag').innerHTML = `
                <div class="primary-name">${result.primary_diagnosis}</div>
                <div class="primary-conf">Confidence: ${result.confidence}</div>
            `;

            // Differential diagnoses bars
            let diffHtml = '';
            result.differential_diagnoses.forEach(d => {
                const barWidth = Math.max(d.confidence * 3, 5);
                diffHtml += `
                    <div class="diagnosis-bar">
                        <span style="width:160px;font-size:0.9rem">${d.disease}</span>
                        <div class="bar" style="width:${barWidth}px"></div>
                        <span style="color:#64b5f6">${d.confidence}%</span>
                    </div>`;
            });
            document.getElementById('diffDiag').innerHTML = diffHtml;

            // PubMed Evidence
            let evHtml = '';
            result.supporting_literature.forEach(e => {
                evHtml += `
                    <div class="ref-item">
                        <span class="pmid">PMID ${e.pmid}</span>
                        <span style="color:#90caf9"> (${e.year})</span>
                        <p style="margin-top:4px">${e.title}</p>
                    </div>`;
            });
            document.getElementById('evidence').innerHTML = evHtml;

            // Formatted report
            document.getElementById('report').textContent = result.formatted_report;

            // Results dikhao
            document.getElementById('results').style.display = 'block';

            // Scroll to results
            document.getElementById('results').scrollIntoView({behavior: 'smooth'});

        } else {
            alert('Error: ' + data.error);
        }

    } catch(err) {
        alert('Connection error: ' + err.message);
    }

    document.getElementById('loading').style.display = 'none';
}