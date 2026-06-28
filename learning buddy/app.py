import json
import math
import os
import re
import sqlite3
import traceback
import urllib.error
import urllib.request
import base64
from collections import Counter
from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), "learning_buddy.db")
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"pdf", "txt"}
MAX_FILE_SIZE = 50 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                points INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (topic_id) REFERENCES topics (id)
            )
            """
        )
        conn.execute("INSERT OR IGNORE INTO users (username, points) VALUES (?, ?)", ("learner", 0))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bloom_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                bloom_level TEXT NOT NULL,
                current_level INTEGER DEFAULT 1,
                mastered INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0,
                correct_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, topic_id, bloom_level),
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (topic_id) REFERENCES topics (id)
            )
            """
        )
        default_topics = [
            "Machine Learning",
            "Natural Language Processing",
            "Neural Networks",
            "AI Ethics",
        ]
        for topic_name in default_topics:
            conn.execute("INSERT OR IGNORE INTO topics (name) VALUES (?)", (topic_name,))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pdf_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS study_materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                material_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pdf_id) REFERENCES pdf_files (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        conn.commit()


init_db()

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "how", "i", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "this", "to", "what", "when", "where", "who",
    "will", "with", "you", "your", "can", "could", "do", "does", "did",
    "learn", "learning", "about", "into", "than", "then", "them", "use", "using"
}

TOPIC_KEYWORDS = {
    "Machine Learning": [
        "machine learning",
        "algorithm",
        "data",
        "training",
        "model",
        "prediction",
        "classification",
        "regression",
    ],
    "Natural Language Processing": [
        "natural language processing",
        "nlp",
        "text",
        "language",
        "token",
        "sentence",
        "embedding",
        "transformer",
    ],
    "Neural Networks": [
        "neural network",
        "neuron",
        "layer",
        "activation",
        "backpropagation",
        "perceptron",
        "deep learning",
    ],
    "AI Ethics": [
        "ai ethics",
        "fairness",
        "bias",
        "privacy",
        "transparency",
        "accountability",
        "responsible ai",
    ],
}


BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

BLOOM_DESCRIPTIONS = {
    "Remember": "Recall facts and basic concepts",
    "Understand": "Explain ideas or concepts",
    "Apply": "Use information in a new situation",
    "Analyze": "Draw connections between ideas",
    "Evaluate": "Justify a decision or choice",
    "Create": "Produce something new",
}



def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = re.split(r"\s+", text.strip())
    return [token for token in tokens if token and token not in STOP_WORDS]


def build_tf(tokens):
    return Counter(tokens)


def cosine_similarity(vec_a, vec_b):
    keys = set(vec_a) | set(vec_b)
    dot_product = sum(vec_a.get(key, 0) * vec_b.get(key, 0) for key in keys)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def get_topics():
    with get_db_connection() as conn:
        rows = conn.execute("SELECT id, name FROM topics ORDER BY id").fetchall()
    return [(row["id"], row["name"]) for row in rows]


def match_topic(message):
    query_tokens = preprocess_text(message)
    if not query_tokens:
        return None, None, 0.0

    query_vector = build_tf(query_tokens)
    query_text = " ".join(query_tokens)
    best_topic_id = None
    best_topic_name = None
    best_score = 0.0

    for topic_id, topic_name in get_topics():
        expanded_text = " ".join([topic_name.lower()] + TOPIC_KEYWORDS.get(topic_name, []))
        topic_vector = build_tf(preprocess_text(expanded_text))
        score = cosine_similarity(query_vector, topic_vector)
        if score > best_score:
            best_score = score
            best_topic_id = topic_id
            best_topic_name = topic_name

    keyword_matches = []
    for topic_name in TOPIC_KEYWORDS:
        topic_keywords = TOPIC_KEYWORDS[topic_name]
        if topic_name.lower() in query_text or any(keyword in query_text for keyword in topic_keywords):
            keyword_matches.append(topic_name)

    if keyword_matches and best_topic_name is None:
        best_topic_name = keyword_matches[0]
        best_score = 0.3
    elif best_topic_name and best_score < 0.1 and keyword_matches:
        best_topic_name = keyword_matches[0]
        best_score = 0.3

    return best_topic_id, best_topic_name, best_score


def build_response(message, topic_name=None, similarity=0.0):
    if topic_name and similarity >= 0.1:
        if topic_name == "Machine Learning":
            explanation = "Machine learning helps computers learn patterns from data so they can make predictions or decisions."
            example = "A spam filter learns from past emails and improves at spotting new junk messages."
            exercise = "Write one problem that could be solved by machine learning."
        elif topic_name == "Natural Language Processing":
            explanation = "Natural Language Processing teaches computers to understand and work with human language."
            example = "A chatbot reads your message and tries to respond in a helpful way."
            exercise = "List two words that a language model might need to understand in a sentence."
        elif topic_name == "Neural Networks":
            explanation = "Neural networks are layered models that can recognize patterns by passing information through many connected nodes."
            example = "A vision system can learn to tell the difference between a cat and a dog from many examples."
            exercise = "Explain what a layer in a neural network does in one sentence."
        else:
            explanation = "AI ethics focuses on making artificial intelligence fair, safe, transparent, and responsible."
            example = "A company should check whether its hiring tool treats people fairly before using it."
            exercise = "Name one way to make an AI system more fair and one way to make it more transparent."
        greeting = f"😊 Great question! Let’s explore {topic_name}."
    else:
        explanation = (
            f"📘 You asked about '{message}'. A simple way to learn it is to break the idea into three parts: "
            f"what it means, how it works, and where it appears in real life."
        )
        example = "🌍 For example, if you are learning a new concept, you can connect it to something you already do every day."
        exercise = "📝 Try explaining the idea in one sentence and giving one everyday example."
        greeting = "😊 Nice question! Let’s build the idea step by step."

    return f"{greeting}\n\n📘 {explanation}\n\n🌍 {example}\n\n📝 {exercise}"


def ollama_prompt(message):
    return (
        "You are Learning Buddy, an educational tutor for students. "
        "Respond in clear, friendly language and include all of the following sections: "
        "Friendly greeting, Explanation, Real-life example, Mini exercise, and Bloom's Taxonomy questions with Remember, Understand, Apply, Analyze, Evaluate, and Create. "
        "Do not include any system or API details. "
        f"User question: {message}"
    )


def call_ollama(message):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": ollama_prompt(message),
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    output_text = []
    with urllib.request.urlopen(req, timeout=30) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "response" in data:
                output_text.append(str(data["response"]))
            elif isinstance(data, dict) and "output" in data:
                output = data["output"]
                if isinstance(output, list):
                    output_text.extend(str(item) for item in output)
                else:
                    output_text.append(str(output))
            elif isinstance(data, dict) and "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    output_text.append(str(choice.get("text") or choice.get("message") or json.dumps(choice)))
            if isinstance(data, dict) and data.get("done"):
                break
    return "".join(output_text).strip()


def log_interaction(message, response_text, topic_id=None):
    try:
        with get_db_connection() as conn:
            user_row = conn.execute("SELECT id FROM users WHERE username = ?", ("learner",)).fetchone()
            user_id = user_row["id"] if user_row else 1
            if topic_id is None:
                topic_id = 1
            conn.execute(
                "INSERT INTO interactions (user_id, topic_id, action) VALUES (?, ?, ?)",
                (user_id, topic_id, f"user:{message} | assistant:{response_text}"),
            )
            conn.commit()
    except Exception:
        print("Exception in log_interaction:")
        traceback.print_exc()


def get_bloom_progress(user_id, topic_id):
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT bloom_level, current_level, mastered, attempts, correct_count FROM bloom_progress WHERE user_id = ? AND topic_id = ? ORDER BY ROWID",
                (user_id, topic_id),
            ).fetchall()
        if not rows:
            return None
        return [{"level": row["bloom_level"], "current": row["current_level"], "mastered": row["mastered"], "attempts": row["attempts"], "correct": row["correct_count"]} for row in rows]
    except Exception:
        print("Exception in get_bloom_progress:")
        traceback.print_exc()
        return None


def init_bloom_progress(user_id, topic_id):
    try:
        with get_db_connection() as conn:
            for level in BLOOM_LEVELS:
                conn.execute(
                    "INSERT OR IGNORE INTO bloom_progress (user_id, topic_id, bloom_level, current_level, mastered) VALUES (?, ?, ?, ?, ?)",
                    (user_id, topic_id, level, 1, 0),
                )
            conn.commit()
    except Exception:
        print("Exception in init_bloom_progress:")
        traceback.print_exc()


def generate_bloom_question(message, topic_name, bloom_level):
    try:
        prompt = (
            f"You are an educational tutor. Generate a single, focused question at the '{bloom_level}' level "
            f"of Bloom's Taxonomy for the student's question about {topic_name}: '{message}'. "
            f"\n\n'{bloom_level}' level means: {BLOOM_DESCRIPTIONS.get(bloom_level, bloom_level)}. "
            f"\n\nRespond with ONLY the question, no explanation."
        )
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
        }
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        output_text = []
        with urllib.request.urlopen(req, timeout=30) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and "response" in data:
                    output_text.append(str(data["response"]))
                if isinstance(data, dict) and data.get("done"):
                    break
        return "".join(output_text).strip()
    except Exception:
        print("Exception in generate_bloom_question:")
        traceback.print_exc()
        return f"What is the {bloom_level.lower()} understanding of {message}?"


def evaluate_answer(user_answer, question, topic_name):
    try:
        prompt = (
            f"You are an educational evaluator. The student was asked: '{question}' "
            f"about {topic_name}. They answered: '{user_answer}'. "
            f"\n\nRespond with ONLY 'CORRECT' if the answer is accurate and shows understanding, "
            f"or 'INCORRECT' if it is wrong or shows misunderstanding. No explanation."
        )
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.3,
        }
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        output_text = []
        with urllib.request.urlopen(req, timeout=30) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and "response" in data:
                    output_text.append(str(data["response"]))
                if isinstance(data, dict) and data.get("done"):
                    break
        result = "".join(output_text).strip().upper()
        return "CORRECT" in result
    except Exception:
        print("Exception in evaluate_answer:")
        traceback.print_exc()
        return False


def update_bloom_progress(user_id, topic_id, bloom_level, is_correct):
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT current_level, attempts, correct_count, mastered FROM bloom_progress WHERE user_id = ? AND topic_id = ? AND bloom_level = ?",
                (user_id, topic_id, bloom_level),
            ).fetchone()
            if not row:
                init_bloom_progress(user_id, topic_id)
                row = conn.execute(
                    "SELECT current_level, attempts, correct_count, mastered FROM bloom_progress WHERE user_id = ? AND topic_id = ? AND bloom_level = ?",
                    (user_id, topic_id, bloom_level),
                ).fetchone()
            attempts = (row["attempts"] or 0) + 1
            correct_count = (row["correct_count"] or 0) + (1 if is_correct else 0)
            mastered = 1 if correct_count >= 2 else 0
            conn.execute(
                "UPDATE bloom_progress SET attempts = ?, correct_count = ?, mastered = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND topic_id = ? AND bloom_level = ?",
                (attempts, correct_count, mastered, user_id, topic_id, bloom_level),
            )
            conn.commit()
            return mastered
    except Exception:
        print("Exception in update_bloom_progress:")
        traceback.print_exc()
        return 0


def extract_pdf_text(file_path):
    """Extract text from PDF or TXT file"""
    try:
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text
            except ImportError:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
    except Exception:
        print("Exception in extract_pdf_text:")
        traceback.print_exc()
        return ""


def generate_study_material(pdf_content, material_type, topic="Study Material"):
    """Generate study material using Ollama"""
    try:
        prompts = {
            "summary": f"Provide a concise chapter-wise summary of the following content in 3-4 bullet points per chapter:\n\n{pdf_content}\n\nFormat: Chapter Name: • Point 1 • Point 2 • Point 3",
            "important_points": f"Extract the 10 most important points from the following content:\n\n{pdf_content}\n\nFormat: 1. Point\n2. Point\n... (numbered list)",
            "short_notes": f"Create short study notes (200-300 words) from the following content:\n\n{pdf_content}\n\nBe concise and focus on key concepts.",
            "long_notes": f"Create detailed study notes (800-1000 words) from the following content:\n\n{pdf_content}\n\nInclude explanations and examples.",
            "2mark_questions": f"Generate 5 questions answerable in 2-3 lines from the following content:\n\n{pdf_content}\n\nFormat: Q1: Question?\nA1: Answer\n...",
            "5mark_questions": f"Generate 5 questions answerable in about 5 lines from the following content:\n\n{pdf_content}\n\nFormat: Q1: Question?\nA1: Answer\n...",
            "10mark_questions": f"Generate 3 questions answerable in about 10 lines from the following content:\n\n{pdf_content}\n\nFormat: Q1: Question?\nA1: Answer\n...",
            "16mark_questions": f"Generate 2 questions answerable in about 16 lines (detailed) from the following content:\n\n{pdf_content}\n\nFormat: Q1: Question?\nA1: Detailed Answer\n...",
            "mcq": f"Generate 10 multiple choice questions with answers from the following content:\n\n{pdf_content}\n\nFormat: Q1: Question?\nA) Option1\nB) Option2\nC) Option3\nD) Option4\nAnswer: A",
            "flashcards": f"Generate 20 flashcard Q&A pairs from the following content:\n\n{pdf_content}\n\nFormat: Q: Question?\nA: Answer\n---\nQ: Next Question?\nA: Next Answer",
        }
        
        prompt = prompts.get(material_type, prompts["summary"])
        max_tokens_map = {
            "summary": 500,
            "important_points": 400,
            "short_notes": 400,
            "long_notes": 1200,
            "2mark_questions": 600,
            "5mark_questions": 800,
            "10mark_questions": 1000,
            "16mark_questions": 1200,
            "mcq": 1000,
            "flashcards": 1500,
        }
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens_map.get(material_type, 500),
            "temperature": 0.7,
        }
        
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        output_text = []
        with urllib.request.urlopen(req, timeout=60) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and "response" in data:
                    output_text.append(str(data["response"]))
                if isinstance(data, dict) and data.get("done"):
                    break
        
        return "".join(output_text).strip()
    except Exception:
        print(f"Exception in generate_study_material for {material_type}:")
        traceback.print_exc()
        return f"Unable to generate {material_type}. Please try again."


def answer_pdf_question(pdf_content, question):
    """Answer a question based on PDF content"""
    try:
        prompt = f"Based on the following document, answer this question:\n\nDocument:\n{pdf_content}\n\nQuestion: {question}\n\nAnswer:"
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.7,
        }
        
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        output_text = []
        with urllib.request.urlopen(req, timeout=30) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and "response" in data:
                    output_text.append(str(data["response"]))
                if isinstance(data, dict) and data.get("done"):
                    break
        
        return "".join(output_text).strip()
    except Exception:
        print("Exception in answer_pdf_question:")
        traceback.print_exc()
        return "I cannot answer this question from the document."



HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Learning Buddy</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07111f;
      --panel: #0f172a;
      --panel-2: #111c31;
      --line: rgba(255,255,255,0.08);
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #7c3aed;
      --accent-2: #22d3ee;
      --user: linear-gradient(135deg, #7c3aed, #4f46e5);
      --assistant: #14213d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
      background: radial-gradient(circle at top left, #12243f 0%, var(--bg) 40%, #040816 100%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .shell {
      width: min(980px, 100%);
      height: min(85vh, 860px);
      background: rgba(7, 17, 31, 0.86);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
      overflow: hidden;
      backdrop-filter: blur(18px);
      display: flex;
      flex-direction: column;
    }
    .topbar {
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(255,255,255,0.03);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 700;
      letter-spacing: 0.01em;
    }
    .brand-badge {
      width: 40px;
      height: 40px;
      border-radius: 12px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      font-weight: 800;
    }
    .status {
      color: var(--muted);
      font-size: 0.92rem;
    }
    .status-right {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .bloom-progress {
      color: var(--accent-2);
      font-size: 0.88rem;
      font-weight: 600;
      padding: 4px 12px;
      border-radius: 12px;
      background: rgba(34, 211, 238, 0.1);
    }
    .chat-area {
      flex: 1;
      padding: 20px 20px 10px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
      scroll-behavior: smooth;
    }
    .bubble {
      max-width: 78%;
      padding: 12px 14px;
      border-radius: 16px;
      line-height: 1.5;
      font-size: 0.97rem;
      word-wrap: break-word;
      white-space: pre-line;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .bubble.user {
      align-self: flex-end;
      background: var(--user);
      color: white;
      border-bottom-right-radius: 4px;
    }
    .bubble.assistant {
      align-self: flex-start;
      background: var(--assistant);
      color: var(--text);
      border-bottom-left-radius: 4px;
    }
    .typing {
      display: flex;
      gap: 6px;
      padding: 12px 14px;
      width: fit-content;
      background: var(--assistant);
      border-radius: 16px;
      border-bottom-left-radius: 4px;
      align-self: flex-start;
    }
    .typing span {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent-2);
      animation: pulse 1.2s infinite ease-in-out;
    }
    .typing span:nth-child(2) { animation-delay: 0.2s; }
    .typing span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes pulse {
      0%, 80%, 100% { transform: scale(0.7); opacity: 0.45; }
      40% { transform: scale(1); opacity: 1; }
    }
    .composer {
      padding: 14px 16px 18px;
      border-top: 1px solid var(--line);
      background: rgba(255,255,255,0.02);
    }
    form {
      display: flex;
      gap: 10px;
      align-items: center;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 8px 8px 16px;
    }
    input {
      flex: 1;
      border: none;
      outline: none;
      background: transparent;
      color: var(--text);
      font-size: 0.98rem;
    }
    button {
      border: none;
      border-radius: 999px;
      padding: 10px 14px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: white;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { filter: brightness(1.05); }
    .tabs-container {
      display: flex;
      gap: 4px;
      padding: 12px;
      background: rgba(255,255,255,0.02);
      border-bottom: 1px solid var(--line);
      overflow-x: auto;
    }
    .tab-btn {
      padding: 8px 14px;
      border-radius: 8px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.1);
      color: var(--text);
      cursor: pointer;
      font-size: 0.9rem;
      white-space: nowrap;
      transition: all 0.2s;
    }
    .tab-btn.active {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    .tab-content {
      display: none;
      padding: 16px;
      flex: 1;
      overflow-y: auto;
    }
    .tab-content.active {
      display: block;
    }
    .upload-area {
      border: 2px dashed rgba(34, 211, 238, 0.4);
      border-radius: 12px;
      padding: 24px;
      text-align: center;
      background: rgba(34, 211, 238, 0.05);
      cursor: pointer;
      transition: all 0.2s;
    }
    .upload-area:hover {
      border-color: rgba(34, 211, 238, 0.8);
      background: rgba(34, 211, 238, 0.1);
    }
    .upload-area.dragging {
      border-color: var(--accent-2);
      background: rgba(34, 211, 238, 0.2);
    }
    .material-item {
      padding: 12px;
      background: rgba(255,255,255,0.05);
      border-radius: 8px;
      margin-bottom: 12px;
      border-left: 3px solid var(--accent);
    }
    .material-item h4 {
      margin: 0 0 8px 0;
      color: var(--accent-2);
      font-size: 0.95rem;
    }
    .material-item p {
      margin: 0;
      color: var(--text);
      font-size: 0.9rem;
      line-height: 1.5;
    }
    .pdf-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .pdf-item {
      background: rgba(255,255,255,0.08);
      padding: 12px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
      border: 1px solid rgba(255,255,255,0.1);
    }
    .pdf-item:hover {
      background: rgba(255,255,255,0.12);
      border-color: var(--accent);
    }
    .pdf-item-name {
      font-weight: 600;
      color: var(--accent-2);
      margin-bottom: 4px;
    }
    .pdf-item-actions {
      display: flex;
      gap: 8px;
      margin-top: 8px;
    }
    .pdf-item-actions button {
      padding: 6px 12px;
      font-size: 0.85rem;
      flex: 1;
    }
    @media (max-width: 700px) {
      body { padding: 10px; }
      .shell { height: 95vh; border-radius: 16px; }
      .bubble { max-width: 90%; }
      .topbar { padding: 14px 16px; }
      .composer { padding: 12px 12px 14px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div class="brand">
        <div class="brand-badge">LB</div>
        <div>
          <div>Learning Buddy</div>
          <div class="status">AI study partner • always ready</div>
        </div>
      </div>
      <div class="status-right">
        <div id="bloomProgress" class="bloom-progress">Loading...</div>
        <div class="status">Online</div>
      </div>
    </div>

    <div class="chat-area" id="chatArea">
      <div class="bubble assistant">😊 Hello! I’m your offline Learning Buddy.\n\n📘 I can help you study by matching your question to a topic and explaining it step by step.\n\n🌍 For example, you can ask me about machine learning, NLP, neural networks, or AI ethics.\n\n📝 Try asking: “What is machine learning?”</div>
    </div>

    <div class="composer">
      <form id="chatForm">
        <input id="messageInput" type="text" placeholder="Ask anything about AI, ML, or study tips..." autocomplete="off" />
        <button type="submit">Send</button>
      </form>
    </div>
  </div>

  <script>
    const chatArea = document.getElementById('chatArea');
    const form = document.getElementById('chatForm');
    const input = document.getElementById('messageInput');

    function scrollToBottom() {
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    function addMessage(text, sender) {
      const bubble = document.createElement('div');
      bubble.className = `bubble ${sender}`;
      bubble.textContent = text;
      bubble.style.whiteSpace = 'pre-line';
      chatArea.appendChild(bubble);
      scrollToBottom();
    }

    function showTyping() {
      const typing = document.createElement('div');
      typing.className = 'typing';
      typing.id = 'typingIndicator';
      typing.innerHTML = '<span></span><span></span><span></span>';
      chatArea.appendChild(typing);
      scrollToBottom();
    }

    function hideTyping() {
      const typing = document.getElementById('typingIndicator');
      if (typing) typing.remove();
    }

    async function loadBloomProgress() {
      try {
        const resp = await fetch('/api/bloom/progress');
        const data = await resp.json();
        const progress = data.progress || [];
        const masteredCount = progress.filter(p => p.mastered === 1).length;
        const progressBar = `📚 Mastered: ${masteredCount}/6 Levels`;
        document.getElementById('bloomProgress').textContent = progressBar;
      } catch (e) {
        console.error('Error loading Bloom progress:', e);
      }
    }

    async function showBloomsQuestion(message, topicName, levelIdx = 0) {
      try {
        const resp = await fetch('/api/bloom/question', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, topic: topicName, level: levelIdx })
        });
        const data = await resp.json();
        const levelLabel = `[${data.level}]`;
        const questionText = `${levelLabel} ${data.question}`;
        addMessage(questionText, 'assistant');
        
        const answerForm = document.createElement('div');
        answerForm.id = 'bloomAnswerForm';
        answerForm.style.cssText = 'margin: 12px 0; padding: 12px; background: rgba(34, 211, 238, 0.1); border-radius: 12px; display: flex; gap: 8px;';
        answerForm.innerHTML = `
          <input id="bloomInput" type="text" placeholder="Your answer..." style="flex: 1; padding: 8px; border-radius: 8px; border: 1px solid rgba(34, 211, 238, 0.3); background: rgba(15, 23, 42, 0.8); color: #f8fafc; font-size: 0.9rem;" />
          <button id="submitBloom" style="padding: 8px 16px; border-radius: 8px; border: none; background: linear-gradient(135deg, #7c3aed, #22d3ee); color: white; cursor: pointer; font-weight: 600;">Submit</button>
        `;
        chatArea.appendChild(answerForm);
        scrollToBottom();
        
        document.getElementById('submitBloom').addEventListener('click', async () => {
          const answer = document.getElementById('bloomInput').value.trim();
          if (!answer) return;
          
          addMessage(answer, 'user');
          answerForm.remove();
          
          try {
            const evalResp = await fetch('/api/bloom/evaluate', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ answer, question: data.question, topic: topicName, level: data.level })
            });
            const evalData = await evalResp.json();
            addMessage(evalData.feedback, 'assistant');
            
            await loadBloomProgress();
            
            if (evalData.correct && evalData.next_level < 6) {
              setTimeout(() => showBloomsQuestion(message, topicName, evalData.next_level), 1000);
            }
          } catch (e) {
            addMessage('Could not evaluate answer. Try again.', 'assistant');
          }
        });
      } catch (e) {
        console.error('Error loading Bloom question:', e);
      }
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const message = input.value.trim();
      if (!message) return;

      addMessage(message, 'user');
      input.value = '';
      showTyping();

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message })
        });
        const data = await response.json();
        hideTyping();
        addMessage(data.reply || 'I am here to help.', 'assistant');
        
        setTimeout(() => showBloomsQuestion(message, 'Learning', 0), 1500);
      } catch (error) {
        hideTyping();
        addMessage('Sorry, I could not respond right now.', 'assistant');
      }
    });

    loadBloomProgress();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()

        if not message:
            return jsonify({"reply": "😊 Hello!\n\n📘 Ask me anything about AI or study topics and I will explain it step by step.\n\n🌍 For example, try asking about machine learning or neural networks.\n\n📝 What would you like to learn today?"})

        topic_id, topic_name, similarity = match_topic(message)

        try:
          reply = call_ollama(message)
        except (urllib.error.URLError, ConnectionError, TimeoutError):
          print("Ollama is not reachable at", OLLAMA_API_URL)
          return jsonify({
            "reply": "😊 Ollama is not running right now. Please start your local Ollama server at http://localhost:11434 and try again.\n\n📘 I rely on the qwen2.5:1.5b model to generate your answer.\n\n🌍 Once the server is available, I can continue helping you learn.\n\n📝 Keep the question ready and ask again."
          })
        except Exception:
          print("Exception while calling Ollama:")
          traceback.print_exc()
          return jsonify({
            "reply": "😊 I hit a problem reaching Ollama. Please check that the local server is running and try again.\n\n📘 I need the offline qwen2.5:1.5b model to answer your question.\n\n🌍 Once the server is available, I can continue tutoring you.\n\n📝 Try again in a moment."
          })

        log_interaction(message, reply, topic_id)
        return jsonify({"reply": reply})
    except Exception:
        print("Exception in /api/chat:")
        traceback.print_exc()
        return jsonify({"reply": "😊 I hit a small issue while preparing your answer.\n\n📘 Please try again with a simple question such as: What is Machine Learning?\n\n🌍 I am still here to help you learn offline.\n\n📝 Try asking again."})


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    print("Unhandled exception:")
    traceback.print_exc()
    return jsonify({"reply": "😊 I hit a small issue while preparing your answer.\n\n📘 Please try again with a simple question such as: What is Machine Learning?\n\n🌍 I am still here to help you learn offline.\n\n📝 Try asking again."})


@app.route("/api/bloom/progress", methods=["GET"])
def get_progress():
    try:
        with get_db_connection() as conn:
            user = conn.execute("SELECT id FROM users WHERE username = ?", ("learner",)).fetchone()
            user_id = user["id"] if user else 1
            topic_rows = conn.execute("SELECT id FROM topics LIMIT 1").fetchone()
            topic_id = topic_rows["id"] if topic_rows else 1
        progress = get_bloom_progress(user_id, topic_id)
        if not progress:
            init_bloom_progress(user_id, topic_id)
            progress = get_bloom_progress(user_id, topic_id)
        return jsonify({"progress": progress or []})
    except Exception:
        print("Exception in /api/bloom/progress:")
        traceback.print_exc()
        return jsonify({"progress": []})


@app.route("/api/bloom/question", methods=["POST"])
def bloom_question():
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        topic_name = (data.get("topic") or "General").strip()
        bloom_level_idx = data.get("level", 0)
        
        if bloom_level_idx < 0 or bloom_level_idx >= len(BLOOM_LEVELS):
            bloom_level_idx = 0
        bloom_level = BLOOM_LEVELS[bloom_level_idx]
        
        question = generate_bloom_question(user_message, topic_name, bloom_level)
        return jsonify({"level": bloom_level, "question": question, "level_idx": bloom_level_idx})
    except Exception:
        print("Exception in /api/bloom/question:")
        traceback.print_exc()
        return jsonify({"level": "Remember", "question": "What are the key ideas from your previous answer?", "level_idx": 0})


@app.route("/api/bloom/evaluate", methods=["POST"])
def bloom_evaluate():
    try:
        data = request.get_json(silent=True) or {}
        user_answer = (data.get("answer") or "").strip()
        question = (data.get("question") or "").strip()
        topic_name = (data.get("topic") or "General").strip()
        bloom_level = (data.get("level") or "Remember").strip()
        
        with get_db_connection() as conn:
            user = conn.execute("SELECT id FROM users WHERE username = ?", ("learner",)).fetchone()
            user_id = user["id"] if user else 1
            topic_rows = conn.execute("SELECT id FROM topics WHERE name = ?", (topic_name,)).fetchone()
            topic_id = topic_rows["id"] if topic_rows else 1
        
        is_correct = evaluate_answer(user_answer, question, topic_name)
        mastered = update_bloom_progress(user_id, topic_id, bloom_level, is_correct)
        
        feedback = "✓ Excellent answer! You've shown good understanding." if is_correct else "✗ Not quite. Let's review the concept and try again."
        next_level_idx = min(BLOOM_LEVELS.index(bloom_level) + 1, len(BLOOM_LEVELS) - 1) if is_correct else BLOOM_LEVELS.index(bloom_level)
        
        return jsonify({"correct": is_correct, "feedback": feedback, "mastered": mastered, "next_level": next_level_idx})
    except Exception:
        print("Exception in /api/bloom/evaluate:")
        traceback.print_exc()
        return jsonify({"correct": False, "feedback": "Unable to evaluate. Please try again.", "mastered": 0, "next_level": 0})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
