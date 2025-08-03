# app.py

from flask import Flask, render_template, request, jsonify
from chatbot import get_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json["message"]
    answer = get_response(user_input)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
