from flask import Flask, send_from_directory
from pathlib import Path


app = Flask(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@app.route("/")
def home():
    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


@app.route("/css/<path:filename>")
def css(filename):
    return send_from_directory(
        FRONTEND_DIR / "css",
        filename
    )


@app.route("/js/<path:filename>")
def js(filename):
    return send_from_directory(
        FRONTEND_DIR / "js",
        filename
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )