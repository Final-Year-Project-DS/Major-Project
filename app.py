from flask import Flask, request, render_template, redirect, url_for, session, Response, stream_with_context, jsonify
import subprocess
import threading
import sys
import sqlite3

from datetime import datetime

# Create reports table if it doesn't exist
def init_db():
    conn = sqlite3.connect("ids_reports.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attack_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            manual_summary TEXT,
            llm_insight TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Call this once when app starts
init_db()

app = Flask(__name__)
app.secret_key = "your_super_secret_key_here"  # Change this securely in production

PYTHON_EXEC = sys.executable

# ✅ Root route redirects to login
@app.route("/")
def home():
    return redirect(url_for("login"))

# 🔁 Helper to run a script
def run_script(script_name):
    try:
        subprocess.run([PYTHON_EXEC, script_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {script_name}: {e}")
        return False

# 🔐 Login page (GET + POST)
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "password123":
            session["logged_in"] = True
            return redirect(url_for("frontend"))  # Redirect to frontend page after login
        else:
            return render_template("login.html", error="❌ Invalid credentials")

    return render_template("login.html")

# ✅ Frontend page after login
@app.route("/frontend")
def frontend():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template("frontend.html")

# ✅ Pipeline execution after login
@app.route("/start_pipeline")
def start_pipeline():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    ml_success = run_script("machine_learning_code.py")
    fl_success = run_script("optimized_code.py")

    # Launch LLM Gradio UI in background
    def launch_llm():
        run_script("llm_ids_assistant.py")

    threading.Thread(target=launch_llm).start()

    return render_template(
        "success.html",
        ml=ml_success,
        fl=fl_success,
        llm=True
    )

# 🔓 Logout route
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ✅ Full Pipeline (Streamed with SSE)
@app.route("/run_all_stream", methods=["GET"])
def run_all_stream():
    @stream_with_context
    def generate():
        # Step 1: ML Training
        yield f"data: ▶️ Starting ML Training (Random Forest & LSTM)...\n\n"
        if run_script("machine_learning_code.py"):
            yield f"data: ✅ ML Training completed.\n\n"
        else:
            yield f"data: ❌ ML Training failed.\n\n"
            return

        # Step 2: Federated + RL Training
        yield f"data: ▶️ Starting Federated + RL Training...\n\n"
        if run_script("optimized_code.py"):
            yield f"data: ✅ Federated + RL Training completed.\n\n"
        else:
            yield f"data: ❌ Federated + RL Training failed.\n\n"
            return

        # Step 3: Launch LLM Gradio UI
        yield f"data: ⚙️ Launching LLM Gradio UI...\n\n"
        thread = threading.Thread(target=lambda: subprocess.run([PYTHON_EXEC, "llm_ids_assistant.py"]))
        thread.start()

        yield f"data: 🎯 Full pipeline executed successfully. LLM UI launching...\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route("/reports", methods=["GET"])
def view_reports():
    try:
        conn = sqlite3.connect("ids_reports.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reports ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()

        report_list = [
            {
                "id": row[0],
                "attack_type": row[1],
                "timestamp": row[2],
                "manual_summary": row[3],
                "llm_insight": row[4]
            }
            for row in rows
        ]

        return jsonify(report_list)

    except Exception as e:
        return jsonify({"error": f"❌ Failed to fetch reports: {str(e)}"}), 500

# ✅ Start the app
if __name__ == "__main__":
    print(f"✅ Using Python interpreter: {sys.executable}")
    app.run(host="0.0.0.0", port=5000)


