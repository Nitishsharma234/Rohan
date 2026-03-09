"""
MediScan AI — Flask backend
Endpoints:
  POST /api/chat          — text chat
  POST /api/image         — image medicine detection
  GET  /api/history       — list all sessions
  GET  /api/history/<sid> — get one session
  DEL  /api/history/<sid> — delete session
  GET  /api/health        — system status
  GET  /                  — serve index.html
"""
import json, os, re, uuid, base64, io, time, glob
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context

# ── sibling modules ───────────────────────────────────────────────────────
from rag_engine       import search_rag
from internet_search  import search_web, web_context_text, DDGS_OK
from image_analyzer   import detect_medicine, OCR_OK

# ── optional ollama ───────────────────────────────────────────────────────
OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_OK    = False
try:
    import requests as _req
    r = _req.get(f"{OLLAMA_URL}/api/tags", timeout=2)
    OLLAMA_OK = r.status_code == 200
    print(f"  [ollama] Connected — model: {OLLAMA_MODEL}")
except:
    print(f"  [ollama] Not reachable — will use RAG-only mode")

# ── CSV medicine database ─────────────────────────────────────────────────
import csv
MEDICINE_DB = []
CSV_PATH    = os.path.join(os.path.dirname(__file__), "data", "medicines.csv")
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        MEDICINE_DB = [row for row in reader]
    print(f"  [csv] Loaded {len(MEDICINE_DB)} medicines from medicines.csv")
else:
    print(f"  [csv] WARNING: data/medicines.csv not found")

# ── history store ─────────────────────────────────────────────────────────
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

def _hist_path(sid): return os.path.join(HISTORY_DIR, f"{sid}.json")

def load_session(sid):
    try:
        with open(_hist_path(sid), encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"id": sid, "title": "New Chat", "created": _now(), "messages": []}

def save_session(session):
    with open(_hist_path(session["id"]), "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)

def list_sessions():
    sessions = []
    for fpath in sorted(glob.glob(os.path.join(HISTORY_DIR, "*.json")), reverse=True):
        try:
            with open(fpath, encoding="utf-8") as f:
                s = json.load(f)
            sessions.append({
                "id": s["id"], "title": s.get("title","Chat"),
                "created": s.get("created",""), "msg_count": len(s.get("messages",[]))
            })
        except: pass
    return sessions

def _now(): return datetime.now().isoformat()

# ── medicine CSV lookup (exact + substring) ───────────────────────────────
def csv_lookup(query: str):
    q = query.lower()
    for row in MEDICINE_DB:
        if row["Name"].lower() in q or q in row["Name"].lower():
            return row
    return None

# ── system prompt builder ─────────────────────────────────────────────────
_GREETINGS = {"hi","hello","hey","sup","yo","hola","thanks","thank","bye","ok","okay","great","sure"}

def is_greeting(text):
    words = set(re.sub(r"[^a-z ]", "", text.lower()).split())
    return bool(words & _GREETINGS) and len(words) <= 4

def build_system(rag_ctx="", csv_row=None, web_ctx=""):
    csv_block = ""
    if csv_row:
        csv_block = f"""
Medicine Database Match:
  Name        : {csv_row.get('Name','')}
  Category    : {csv_row.get('Category','')}
  Strength    : {csv_row.get('Strength','')}
  Form        : {csv_row.get('Dosage Form','')}
  Manufacturer: {csv_row.get('Manufacturer','')}
  Indication  : {csv_row.get('Indication','')}
  Class       : {csv_row.get('Classification','')}
"""
    rag_section = ("--- RAG Knowledge ---\n" + rag_ctx) if rag_ctx else ""
    web_section = ("--- Web Search ---\n" + web_ctx) if web_ctx else ""
    return f"""You are MediScan AI — a highly knowledgeable and detailed medical assistant.

RESPONSE STYLE — follow for ALL medical questions:
- Write DETAILED, thorough answers. Never give one-line replies to medical questions.
- Structure your answer with bold headers: **What is it?** **Uses** **How it works** **Dosage** **Side Effects** **Precautions**
- Use bullet points inside each section.
- Explain medical terms in simple language.
- Include specific dosage numbers (mg, frequency, max dose) when available.
- Minimum 150 words for any medicine or disease question.

RULES:
1. Greetings -> reply naturally and briefly. No medical headers.
2. Use ALL context provided below — do not skip any of it.
3. If context is missing details, fill gaps with your medical knowledge.
4. Always end with: "Always consult a healthcare professional before starting any medication."
5. Never say "based on the context" — just answer directly and confidently.

{rag_section}
{csv_block}
{web_section}
""".strip()

# ── Ollama call ───────────────────────────────────────────────────────────
import requests as _req

def call_ollama(messages, system, stream=False):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role":"system","content":system}] + messages,
        "stream": stream,
        "options": {"temperature": 0.5, "num_predict": 800}
    }
    if stream:
        resp = _req.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=60)
        def gen():
            for line in resp.iter_lines():
                if not line: continue
                try:
                    chunk = json.loads(line)
                    tok = chunk.get("message",{}).get("content","")
                    if tok:
                        yield f"data: {json.dumps({'token': tok})}\n\n"
                    if chunk.get("done"):
                        yield "data: [DONE]\n\n"
                except: pass
        return gen()
    else:
        resp = _req.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
        data = resp.json()
        return data.get("message",{}).get("content","I could not generate a response.")

def rag_only_answer(query, rag_chunks, csv_row, web_results):
    """Structured fallback when Ollama is offline."""
    parts = []
    if csv_row:
        parts.append(f"**{csv_row['Name']}** ({csv_row.get('Category','')})\n"
                     f"- Strength: {csv_row.get('Strength','')}\n"
                     f"- Use: {csv_row.get('Indication','')}\n"
                     f"- Form: {csv_row.get('Dosage Form','')}\n"
                     f"- Class: {csv_row.get('Classification','')}")
    for chunk in rag_chunks[:2]:
        parts.append(chunk["text"][:500])
    if web_results:
        parts.append("**Web Sources:**")
        for r in web_results[:2]:
            parts.append(f"- [{r['title']}]({r['url']}): {r['body'][:200]}")
    if not parts:
        return "I don't have enough information. Please consult a healthcare professional."
    return "\n\n".join(parts) + "\n\n⚠️ Consult a healthcare professional before taking any medication."

# ── Flask app ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "ollama": OLLAMA_OK,
        "model":  OLLAMA_MODEL,
        "ocr":    OCR_OK,
        "web":    DDGS_OK,
        "csv":    len(MEDICINE_DB),
        "status": "ok"
    })

# ── CHAT ─────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    body    = request.get_json(force=True)
    query   = body.get("message","").strip()
    sid     = body.get("session_id") or str(uuid.uuid4())
    do_stream = body.get("stream", False)

    if not query:
        return jsonify({"error": "empty message"}), 400

    session  = load_session(sid)
    greeting = is_greeting(query)

    # build context
    rag_chunks  = [] if greeting else search_rag(query, k=6)
    csv_row     = None if greeting else csv_lookup(query)
    web_results = []
    low_confidence = rag_chunks and all(c["score"] < 0.15 for c in rag_chunks)
    if not greeting and (not rag_chunks or low_confidence) and not csv_row:
        web_results = search_web(query)

    rag_ctx = "\n\n".join(c["text"][:1200] for c in rag_chunks)
    web_ctx = web_context_text(web_results)
    system  = build_system(rag_ctx, csv_row, web_ctx)

    # build ollama message history (last 10 turns)
    history_msgs = session["messages"][-20:]
    ollama_msgs  = [{"role": m["role"], "content": m["content"]} for m in history_msgs]
    ollama_msgs.append({"role": "user", "content": query})

    # save user message
    session["messages"].append({"role":"user","content":query,"ts":_now()})
    if len(session["messages"]) == 1:
        session["title"] = query[:60]

    # SSE streaming
    if do_stream and OLLAMA_OK:
        gen = call_ollama(ollama_msgs, system, stream=True)
        def stream_and_save():
            full = ""
            for chunk in gen:
                yield chunk
                if chunk.startswith("data:") and "[DONE]" not in chunk:
                    try:
                        full += json.loads(chunk[5:].strip()).get("token","")
                    except: pass
            session["messages"].append({"role":"assistant","content":full,"ts":_now()})
            save_session(session)
        return Response(stream_with_context(stream_and_save()),
                        mimetype="text/event-stream",
                        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    # non-streaming
    if OLLAMA_OK:
        answer = call_ollama(ollama_msgs, system, stream=False)
    else:
        answer = rag_only_answer(query, rag_chunks, csv_row, web_results)

    session["messages"].append({"role":"assistant","content":answer,"ts":_now()})
    save_session(session)

    return jsonify({
        "answer":     answer,
        "session_id": sid,
        "sources": {
            "rag":  [{"source":c["source"],"score":c["score"]} for c in rag_chunks],
            "csv":  bool(csv_row),
            "web":  len(web_results),
        }
    })

# ── IMAGE ─────────────────────────────────────────────────────────────────
@app.route("/api/image", methods=["POST"])
def analyze_image():
    body = request.get_json(force=True)
    sid  = body.get("session_id") or str(uuid.uuid4())
    b64  = body.get("image","")
    if not b64:
        return jsonify({"error": "no image"}), 400

    # strip data-url prefix if present
    if "," in b64: b64 = b64.split(",",1)[1]
    img_bytes = base64.b64decode(b64)

    result = detect_medicine(img_bytes, MEDICINE_DB)

    top_name   = result["top_name"]
    confidence = result["confidence"]
    matches    = result["matches"]
    tokens     = result["tokens"]

    # build answer
    if matches:
        top = matches[0]["medicine"]
        answer = (f"📦 **Detected: {top.get('Name','?')}** ({int(confidence*100)}% confidence)\n\n"
                  f"- **Category:** {top.get('Category','')}\n"
                  f"- **Strength:** {top.get('Strength','')}\n"
                  f"- **Form:** {top.get('Dosage Form','')}\n"
                  f"- **Use:** {top.get('Indication','')}\n"
                  f"- **Class:** {top.get('Classification','')}\n\n")
        if len(matches) > 1:
            answer += "**Other possible matches:** " + ", ".join(
                f"{m['medicine'].get('Name','?')} ({int(m['score']*100)}%)" for m in matches[1:4])
        answer += "\n\n⚠️ Consult a healthcare professional before taking any medication."
    elif tokens:
        answer = f"🔍 OCR extracted: `{'`, `'.join(tokens[:5])}`\n\nNo exact medicine match found. Try asking about it directly."
    else:
        answer = "❌ Could not extract text from image. Ensure the label is clearly visible and well-lit."

    # store in history
    session = load_session(sid)
    session["messages"].append({"role":"user","content":"[Image uploaded for analysis]","ts":_now()})
    session["messages"].append({"role":"assistant","content":answer,"ts":_now()})
    if not session.get("title") or session["title"] == "New Chat":
        session["title"] = f"Image: {top_name}"
    save_session(session)

    return jsonify({
        "answer":      answer,
        "session_id":  sid,
        "top_name":    top_name,
        "confidence":  confidence,
        "tokens":      tokens,
        "matches":     [{"name": m["medicine"].get("Name",""), "score": m["score"]} for m in matches],
    })

# ── HISTORY ──────────────────────────────────────────────────────────────
@app.route("/api/history")
def get_history():
    return jsonify(list_sessions())

@app.route("/api/history/<sid>")
def get_session(sid):
    return jsonify(load_session(sid))

@app.route("/api/history/<sid>", methods=["DELETE"])
def delete_session(sid):
    path = _hist_path(sid)
    if os.path.exists(path): os.remove(path)
    return jsonify({"deleted": sid})

@app.route("/api/history/<sid>/title", methods=["POST"])
def rename_session(sid):
    session = load_session(sid)
    session["title"] = request.get_json(force=True).get("title", session["title"])
    save_session(session)
    return jsonify({"ok": True})

if __name__ == "__main__":
    import sys
    print("\n🩺 MediScan AI starting…\n")
    # exclude torch/site-packages from watchdog to prevent spurious restarts
    site_pkgs = next((p for p in sys.path if "site-packages" in p), None)
    exclude   = [site_pkgs] if site_pkgs else []
    app.run(debug=True, port=5000,
            exclude_patterns=exclude,
            reloader_type="stat")          # stat reloader ignores binary changes