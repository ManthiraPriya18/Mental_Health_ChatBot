from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def chat():
    user_input = ""
    bot_response = ""

    if request.method == "POST":
        user_input = request.form["user_input"]

        try:
            # Use Ollama LLaMA3 model to respond
            result = subprocess.run(
                ["ollama", "run", "llama3:instruct", user_input],
                capture_output=True,
                text=True,
                timeout=60
            )
            bot_response = result.stdout.strip()
        except Exception as e:
            bot_response = f"Error: {str(e)}"

    return render_template("index.html", user_input=user_input, bot_response=bot_response)

if __name__ == "__main__":
    app.run(debug=True)
