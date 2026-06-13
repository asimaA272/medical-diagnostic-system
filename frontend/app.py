from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)

https://medical-backend123-2kiw.onrender.com

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["xray"]
        response = requests.post(BACKEND_URL, files={"file": file})
        result = response.json()
        return render_template("result.html", prediction=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)