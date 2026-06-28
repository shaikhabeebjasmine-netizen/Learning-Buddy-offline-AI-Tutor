from flask import Flask, request, jsonify, render_template_string
import sqlite3
import requests
from datetime import datetime

app = Flask(__name__)

DB = "learning_buddy.db"
MODEL_NAME = "gemma3:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"

BADGES = [
    (50, "Beginner Learner 🟢"),
    (100, "Smart Student ⭐"),
    (200, "Quiz Master 🧠"),
    (300, "Learning Champion 🏆"),
    (500, "Offline AI Expert 🚀")
]

BLOOM_LEVELS = {
    "remember": "Define, list, recall basic facts.",
    "understand": "Explain in simple words with examples.",
    "apply": "Show how to use this concept in real life or coding.",
    "analyze": "Break the concept into parts and compare.",
    "evaluate": "Judge advantages, disadvantages, and usefulness.",
    "create": "Give project ideas or creative tasks using this concept."
}

def init_db():
    con = sqlite3.connect(DB)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS progress(
        id INTEGER PRIMARY KEY,
        points INTEGER DEFAULT 0
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mode TEXT,
        user_input TEXT,
        ai_response TEXT,
        time TEXT
    )
    """)

    cur.execute("INSERT OR IGNORE INTO progress(id, points) VALUES(1, 0)")
    con.commit()
    con.close()

def get_points():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT points FROM progress WHERE id=1")
    points = cur.fetchone()[0]
    con.close()
    return points

def add_points(p):
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("UPDATE progress SET points = points + ? WHERE id=1", (p,))
    con.commit()
    con.close()

def get_badge(points):
    badge = "No badge yet"
    for limit, name in BADGES:
        if points >= limit:
            badge = name
    return badge

def save_history(mode, user_input, ai_response):
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO history(mode, user_input, ai_response, time) VALUES(?,?,?,?)",
        (mode, user_input, ai_response, datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    con.commit()
    con.close()

def ask_ai(prompt):
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        data = res.json()
        return data.get("response", "AI response error. Check Ollama model name.")

    except Exception as e:
        return f"""Ollama is not connected.

Do this:
1. Open new terminal
2. Run: ollama run gemma3:1b
3. Keep it open
4. Refresh this page

Error:
{e}
"""

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Learning Buddy</title>

<style>
body{
    margin:0;
    font-family:'Segoe UI', Arial;
    background:linear-gradient(135deg,#667eea,#764ba2);
    min-height:100vh;
}

.header{
    color:white;
    text-align:center;
    padding:35px;
}

.header h1{
    font-size:42px;
    margin-bottom:5px;
}

.container{
    width:90%;
    max-width:1100px;
    margin:auto;
}

.grid{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
    gap:25px;
}

.card{
    background:white;
    padding:24px;
    border-radius:20px;
    box-shadow:0 15px 30px rgba(0,0,0,0.2);
}

.card h2{
    color:#4f46e5;
}

input, textarea, select{
    width:100%;
    padding:14px;
    font-size:16px;
    border-radius:12px;
    border:1px solid #ccc;
    margin-top:10px;
    box-sizing:border-box;
}

button{
    background:linear-gradient(135deg,#ff7eb3,#ff758c);
    color:white;
    border:none;
    padding:14px 24px;
    border-radius:12px;
    margin-top:12px;
    cursor:pointer;
    font-size:16px;
    font-weight:bold;
}

button:hover{
    transform:scale(1.03);
}

.response{
    background:#f1f5ff;
    padding:18px;
    margin-top:15px;
    border-radius:14px;
    white-space:pre-wrap;
    line-height:1.6;
    border-left:6px solid #6366f1;
}

.badge-box{
    background:#fff7ed;
    border-left:6px solid #f97316;
}

.footer{
    color:white;
    text-align:center;
    padding:25px;
}
</style>
</head>

<body>

<div class="header">
<h1>📚 Learning Buddy</h1>
<p>Offline AI Tutor with Bloom's Taxonomy</p>
</div>

<div class="container">

<div class="grid">

<div class="card">
<h2>💬 Learn Mode</h2>
<input id="learnTopic" placeholder="Example: Deep Learning, Python, DBMS">
<button onclick="learn()">Learn</button>
<div id="learnAns" class="response"></div>
</div>

<div class="card">
<h2>🌱 Bloom's Taxonomy Mode</h2>
<input id="bloomTopic" placeholder="Enter topic: Machine Learning">
<select id="bloomLevel">
<option value="remember">Remember</option>
<option value="understand">Understand</option>
<option value="apply">Apply</option>
<option value="analyze">Analyze</option>
<option value="evaluate">Evaluate</option>
<option value="create">Create</option>
</select>
<button onclick="bloom()">Explain by Level</button>
<div id="bloomAns" class="response"></div>
</div>

<div class="card">
<h2>🧠 Quiz Mode</h2>
<input id="quizTopic" placeholder="Example: Python loops">
<button onclick="quiz()">Generate Quiz</button>
<div id="quizAns" class="response"></div>
</div>

<div class="card">
<h2>📝 Summarize Notes</h2>
<textarea id="content" rows="7" placeholder="Paste your notes here..."></textarea>
<button onclick="summarize()">Summarize</button>
<div id="summary" class="response"></div>
</div>

<div class="card">
<h2>🏆 Progress</h2>
<button onclick="progress()">Show Points & Badge</button>
<div id="progress" class="response badge-box"></div>
</div>

<div class="card">
<h2>📜 Chat History</h2>
<button onclick="history()">Show History</button>
<div id="history" class="response"></div>
</div>

</div>

</div>

<div class="footer">
<p>Made for students who learn even without internet 🚀</p>
</div>

<script>
function learn(){
    let topic = document.getElementById("learnTopic").value;
    document.getElementById("learnAns").innerText = "Thinking offline...";

    fetch("/learn",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({topic:topic})
    })
    .then(r=>r.json())
    .then(d=>{
        document.getElementById("learnAns").innerText=d.reply;
    });
}

function bloom(){
    let topic = document.getElementById("bloomTopic").value;
    let level = document.getElementById("bloomLevel").value;

    document.getElementById("bloomAns").innerText = "Learning using Bloom's level...";

    fetch("/bloom",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({topic:topic, level:level})
    })
    .then(r=>r.json())
    .then(d=>{
        document.getElementById("bloomAns").innerText=d.reply;
    });
}

function quiz(){
    let topic = document.getElementById("quizTopic").value;
    document.getElementById("quizAns").innerText = "Creating quiz offline...";

    fetch("/quiz",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({topic:topic})
    })
    .then(r=>r.json())
    .then(d=>{
        document.getElementById("quizAns").innerText=d.reply;
    });
}

function summarize(){
    let content = document.getElementById("content").value;
    document.getElementById("summary").innerText = "Summarizing offline...";

    fetch("/summarize",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({content:content})
    })
    .then(r=>r.json())
    .then(d=>{
        document.getElementById("summary").innerText=d.reply;
    });
}

function progress(){
    fetch("/progress")
    .then(r=>r.json())
    .then(d=>{
        document.getElementById("progress").innerText =
        "🎯 Points: " + d.points + "\\n🏅 Badge: " + d.badge;
    });
}

function history(){
    fetch("/history")
    .then(r=>r.json())
    .then(d=>{
        document.getElementById("history").innerText = d.history;
    });
}
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/learn", methods=["POST"])
def learn():
    topic = request.json.get("topic", "").strip()

    if topic == "":
        return jsonify({"reply": "Please enter a topic."})

    prompt = f"""
You are Learning Buddy, an offline AI tutor.

Teach this topic to a student:
{topic}

Rules:
- Use simple English.
- Explain clearly.
- Give definition.
- Give example.
- Give real-life use.
- Give important points.
- End with a small practice question.
"""

    answer = ask_ai(prompt)
    add_points(10)
    points = get_points()
    save_history("Learn Mode", topic, answer)

    return jsonify({"reply": answer + f"""

🎯 Points earned: +10
Total Points: {points}
🏅 Badge: {get_badge(points)}
"""})

@app.route("/bloom", methods=["POST"])
def bloom():
    topic = request.json.get("topic", "").strip()
    level = request.json.get("level", "").strip()

    if topic == "":
        return jsonify({"reply": "Please enter a topic."})

    meaning = BLOOM_LEVELS.get(level, "Explain clearly.")

    prompt = f"""
You are Learning Buddy.

Topic: {topic}
Bloom's Taxonomy Level: {level.upper()}
Level meaning: {meaning}

Explain the topic according to this level.

Format:
1. Level name
2. Simple explanation
3. Example
4. Student activity
5. One practice question
"""

    answer = ask_ai(prompt)
    add_points(15)
    points = get_points()
    save_history("Bloom Mode", topic + " - " + level, answer)

    return jsonify({"reply": answer + f"""

🎯 Points earned: +15
Total Points: {points}
🏅 Badge: {get_badge(points)}
"""})

@app.route("/quiz", methods=["POST"])
def quiz():
    topic = request.json.get("topic", "").strip()

    if topic == "":
        return jsonify({"reply": "Please enter quiz topic."})

    prompt = f"""
Create a student-friendly quiz on:
{topic}

Rules:
- Give 5 MCQ questions.
- Each question should have 4 options.
- Mark the correct answer.
- Give short explanation after each answer.
- Use simple English.
"""

    answer = ask_ai(prompt)
    add_points(20)
    points = get_points()
    save_history("Quiz Mode", topic, answer)

    return jsonify({"reply": answer + f"""

🎯 Points earned: +20
Total Points: {points}
🏅 Badge: {get_badge(points)}
"""})

@app.route("/summarize", methods=["POST"])
def summarize():
    content = request.json.get("content", "").strip()

    if content == "":
        return jsonify({"reply": "Please paste content first."})

    prompt = f"""
Summarize this content for a student.

Rules:
- Use simple points.
- Highlight important keywords.
- Make it easy to revise before exam.

Content:
{content}
"""

    answer = ask_ai(prompt)
    add_points(20)
    points = get_points()
    save_history("Summary Mode", content[:100], answer)

    return jsonify({"reply": answer + f"""

🎯 Points earned: +20
Total Points: {points}
🏅 Badge: {get_badge(points)}
"""})

@app.route("/progress")
def progress():
    points = get_points()
    return jsonify({
        "points": points,
        "badge": get_badge(points)
    })

@app.route("/history")
def history():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT mode, user_input, time FROM history ORDER BY id DESC LIMIT 10")
    rows = cur.fetchall()
    con.close()

    if not rows:
        return jsonify({"history": "No history yet."})

    text = ""
    for mode, user_input, time in rows:
        text += f"📌 {mode}\\nQuestion: {user_input}\\nTime: {time}\\n\\n"

    return jsonify({"history": text})

if __name__ == "__main__":
    init_db()
    app.run(debug=True)