import os
import json
import sqlite3
import subprocess
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter

# ---------- Configuration ----------
DEPARTMENT_VECTORS_FILE = "category_individual_vectors.json"
LEVEL_VECTORS_FILE = "level_individual_vectors.json"
OLLAMA_MODEL = "mistral"
SEMANTIC_CONFIDENCE_THRESHOLD = 0.75

# ---------- Load embedding model ----------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Database setup ----------
conn = sqlite3.connect("tickets.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS ticket_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT,
        body TEXT,
        department TEXT,
        it_level TEXT,
        confidence REAL,
        fallback_used INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# ---------- Department examples ----------
category_examples = {
    "IT Support": [
        "Cannot connect to Wi-Fi on laptop",
        "VPN login fails when working remotely",
        "Outlook not syncing with mail server",
        "Printer not responding in department office",
        "Forgot corporate Windows password",
        "Laptop battery draining quickly",
        "Unable to access internal HR portal from desktop",
        "Peripheral device (webcam/microphone) not working",
        "Cannot map network drive",
        "Unable to print to network printer"
    ],
    "Technical Support": [
        "Jenkins build job failing during deployment",
        "Error: module not found in production logs",
        "API throwing 500 server error after patch release",
        "React build failing due to dependency mismatch",
        "Python backend crashes on user login request",
        "Docker container does not start in staging environment",
        "CI pipeline stuck on database migration step",
        "Integration issue with Microsoft Office 365 SaaS platform",
        "Security configuration issue in cloud SaaS deployment",
        "Need help setting up CI/CD pipeline with secure tenant separation",
        "Azure Active Directory SSO not integrating with Office 2021"
    ],
    "Lab Support": ["Need assistance with lab equipment", "Test tubes missing in lab 3", "Calibration issue with centrifuge"],
    "Finance": ["Reimbursement form not processed", "Issue with payslip", "Incorrect deductions in salary"],
    "HR": ["How to apply for leave?", "Onboarding documents not received", "Update personal information in HR portal"],
    "WPR Support": ["Air conditioning not working in floor 5", "Request for workstation relocation", "Broken chair in conference room"],
    "Training Support": ["Unable to access LMS", "Schedule training session for new team", "Feedback for online module"]
}

# ---------- L1/L2/L3 example bank ----------
example_levels = {
    "L1": [
        "Password reset issue",
        "Cannot connect to Wi-Fi",
        "Outlook email not loading properly",
        "Printer not printing",
        "VPN not working for remote access",
        "Need help accessing shared folder",
        "Audio device not detected on laptop",
        "Unable to login after password change",
        "Corporate email sync problem"
    ],
    "L2": [
        "Software installation fails repeatedly",
        "Configuration issue with internal system",
        "Persistent network dropouts across users",
        "System boot failure after update",
        "Login error even after cache cleared",
        "Operating system performance degradation",
        "Outlook plugin not syncing calendar entries",
        "Application throws exception during startup"
    ],
    "L3": [
        "Database corruption detected in logs",
        "Critical server crash affecting multiple users",
        "Security vulnerability discovered in application",
        "Code error causing backend service failure",
        "Kernel panic after patch update",
        "Hardware fault in data center rack",
        "Email service down due to DNS misconfiguration",
        "Data loss after replication failure"
    ]
}

# ---------- Embedding cache ----------
def cache_individual_vectors(examples_dict, path):
    if os.path.exists(path):
        with open(path, "r") as f:
            raw = json.load(f)
            return {k: [(text, np.array(vec)) for text, vec in v] for k, v in raw.items()}

    out = {}
    for k, examples in examples_dict.items():
        embs = model.encode(examples)
        out[k] = list(zip(examples, embs.tolist()))
    with open(path, "w") as f:
        json.dump(out, f)
    return {k: [(text, np.array(vec)) for text, vec in v] for k, v in out.items()}

embedded_examples = cache_individual_vectors(category_examples, DEPARTMENT_VECTORS_FILE)
level_embedded_examples = cache_individual_vectors(example_levels, LEVEL_VECTORS_FILE)

# ---------- Semantic helpers ----------
def semantic_department_classify(ticket_vector, top_k=5):
    scores = []
    for category, examples in embedded_examples.items():
        for text, vec in examples:
            sim = float(cosine_similarity([ticket_vector], [vec])[0][0])
            scores.append((category, text, sim))
    scores.sort(key=lambda x: x[2], reverse=True)
    top = scores[:top_k]
    vote = Counter()
    for cat, _, s in top:
        vote[cat] += s
    best, best_score = vote.most_common(1)[0]
    raw_max_sim = top[0][2] if top else 0.0
    return best, float(best_score), float(raw_max_sim), top

def semantic_level_classify(ticket_vector):
    sim_scores = {}
    for lvl, examples in level_embedded_examples.items():
        sims = [float(cosine_similarity([ticket_vector], [vec])[0][0]) for _, vec in examples]
        sim_scores[lvl] = float(np.mean(sims))
    best_lvl = max(sim_scores, key=sim_scores.get)
    return best_lvl, sim_scores[best_lvl]

# ---------- Ollama ----------
def call_ollama(prompt, timeout=15):
    try:
        proc = subprocess.run(["ollama", "run", OLLAMA_MODEL],
                              input=prompt.encode("utf-8"),
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              timeout=timeout)
        out = proc.stdout.decode("utf-8").strip()
        if not out:
            raise RuntimeError(f"Empty response from Ollama. Stderr: {proc.stderr.decode('utf-8')}")
        return out
    except FileNotFoundError:
        raise RuntimeError("Ollama CLI not found. Install it from ollama.ai and pull a model (e.g. 'ollama pull mistral').")

def parse_level_from_text(text):
    text_upper = text.upper()
    for lvl in ["L1", "L2", "L3"]:
        if lvl in text_upper:
            return lvl
    if "LEVEL 1" in text_upper or "LEVEL1" in text_upper:
        return "L1"
    if "LEVEL 2" in text_upper or "LEVEL2" in text_upper:
        return "L2"
    if "LEVEL 3" in text_upper or "LEVEL3" in text_upper:
        return "L3"
    if any(w in text_upper for w in ["PASSWORD","VPN","PRINTER","OUTLOOK"]):
        return "L1"
    if any(w in text_upper for w in ["ERROR","INSTALL","CRASH","CONFIGURATION","EXCEPTION"]):
        return "L2"
    return "Uncertain"

# ---------- Main flow ----------
subject = input("Enter the ticket subject: ")
body = input("Enter the ticket body: ")
combined_text = f"Subject: {subject}\nBody: {body}\nCategory context: IT, support, incident ticket."

ticket_vector = model.encode([combined_text])[0]
predicted_dept, dept_agg_score, dept_raw_max_sim, dept_top = semantic_department_classify(ticket_vector)

# Fix for low-confidence or misclassifications
if dept_raw_max_sim < 0.45 or predicted_dept in ["HR", "Finance"]:
    if any(k in combined_text.lower() for k in ["integration", "pipeline", "ci/cd", "azure", "microsoft", "saas", "deployment", "devops", "security", "tenant"]):
        predicted_dept = "Technical Support"

# ---------- Display department ----------
print("\nTop similar examples for department:")
for cat, text, score in dept_top[:5]:
    print(f"[{cat}] ({score:.3f}): {text}")

print(f"\nPredicted Department: {predicted_dept} (aggregated score: {dept_agg_score:.3f}, top_sim: {dept_raw_max_sim:.3f})")

# ---------- Only classify level if IT-related ----------
predicted_level = "N/A"
level_confidence = 0.0
fallback_used = 0

if predicted_dept in ["IT Support", "Technical Support"]:
    predicted_level, level_confidence = semantic_level_classify(ticket_vector)

    if level_confidence < SEMANTIC_CONFIDENCE_THRESHOLD:
        print(f"\nLow semantic confidence for level ({level_confidence:.3f})")
        print("Falling back to local LLM (Ollama)...")
        prompt = f"""Classify the following IT ticket as L1, L2, or L3.

Ticket:
{combined_text}

Definitions:
- L1: Basic issues like passwords, Wi-Fi, VPN, or simple app fixes.
- L2: Intermediate issues like installations, configuration, or system errors.
- L3: Advanced issues like database corruption, outages, or code-level bugs.

Answer only with: L1, L2, or L3."""
        try:
            ollama_out = call_ollama(prompt)
            parsed = parse_level_from_text(ollama_out)
            if parsed != "Uncertain":
                predicted_level = parsed
            fallback_used = 1
        except Exception as e:
            print("Ollama fallback failed:", e)

    print(f"Predicted IT Support Level: {predicted_level} (semantic_conf: {level_confidence:.3f})")
    if fallback_used:
        print("(Used Ollama fallback)")
else:
    print("\nLevel classification skipped (non-IT department).")

# ---------- Log result ----------
cursor.execute('''
    INSERT INTO ticket_log (subject, body, department, it_level, confidence, fallback_used)
    VALUES (?, ?, ?, ?, ?, ?)
''', (subject, body, predicted_dept, predicted_level, float(level_confidence), int(fallback_used)))
conn.commit()
conn.close()
