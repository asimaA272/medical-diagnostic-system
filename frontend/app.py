from flask import Flask, render_template, request
import requests

app = Flask(__name__)
BACKEND_URL = "https://medical-backend.onrender.com/predict"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["xray"]
        response = requests.post(BACKEND_URL, files={"file": file})
        result = response.json()
        return render_template("result.html", prediction=result)
    return render_template("index.html")