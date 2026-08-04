# 🎫 TicketClassifier

Semantic support‑ticket router. Given a ticket's subject and body, it predicts the
right **department** (IT Support, Technical Support, HR, Finance, Lab, …) and the
**escalation level** (L1 / L2 / L3), logs every decision to SQLite, and falls back
to a local LLM when it isn't confident.

---

## 🧠 How it works
1. Each ticket is embedded with `all-MiniLM-L6-v2` (sentence‑transformers).
2. It's compared by **cosine similarity** against a bank of labelled example tickets
   for every department and level.
3. If the top match clears the confidence threshold (`0.75`), that label is used.
4. If not, the ticket is routed to a **local Ollama model (`mistral`)** as a fallback,
   and the decision is flagged so you can see where the semantic layer was unsure.
5. Every classification — labels, confidence, whether the fallback fired — is written
   to `tickets.db` for auditing.

Precomputed example embeddings are cached to JSON so startup stays fast after the
first run.

---

## ⚙️ Setup

```bash
git clone https://github.com/dhxvxn/TicketClassifier.git
cd TicketClassifier
pip install -r requirements.txt   # sentence-transformers, scikit-learn, numpy
```

For the LLM fallback, install [Ollama](https://ollama.com) and pull the model:
```bash
ollama pull mistral
```

Run it:
```bash
python tc.py
```

---

## 🎯 Why it's built this way
Pure keyword routing misclassifies anything phrased unusually; a pure‑LLM router is
slow and costs a call per ticket. This uses fast embeddings for the confident majority
and only spends an LLM call on the genuinely ambiguous ones — with the confidence score
and fallback flag logged so the routing stays explainable.

## 🛠 Tech Stack
Python · sentence‑transformers (MiniLM) · scikit‑learn · Ollama (mistral) · SQLite

## 👨‍💻 Author
**Dhavan** — CSE student
