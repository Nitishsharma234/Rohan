# MediScan AI 🩺

A professional medical chatbot with image OCR, RAG knowledge base, CSV medicine lookup, and DuckDuckGo web search.

## Project Structure

```
mediscan/
├── app.py              ← Flask server (main entry point)
├── rag_engine.py       ← TF-IDF RAG over .txt files
├── image_analyzer.py   ← OCR + fuzzy medicine matching
├── internet_search.py  ← DuckDuckGo web search
├── requirements.txt
├── data/
│   ├── medicines.csv       ← Medicine database (add your own rows)
│   ├── diseases.txt        ← Disease knowledge base
│   ├── medicines_info.txt  ← Detailed medicine info
│   └── general_info.txt    ← General medical info
├── templates/
│   └── index.html          ← Full single-page UI
└── history/                ← Auto-created; stores chat sessions as JSON
```

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Tesseract OCR engine
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Linux:   sudo apt install tesseract-ocr
# macOS:   brew install tesseract

# 3. (Optional) Start Ollama + Gemma3 for AI responses
ollama pull gemma3:4b
ollama serve

# 4. Run the app
python app.py
# → Open http://localhost:5000
```

## Features

- 💬 **Chat** — Ask about any medicine or disease
- 📷 **Image OCR** — Upload tablet/strip photo → fuzzy match → medicine info
- 📚 **RAG** — Searches your .txt knowledge files via TF-IDF
- 💊 **CSV Lookup** — Exact + substring match on medicines.csv
- 🌐 **Web Search** — DuckDuckGo fallback when local data is missing
- 🕒 **History** — All chats saved as JSON, browsable in sidebar
- ⬇ **Export** — Download any conversation as .txt

## Adding More Medicines

Edit `data/medicines.csv` — add rows following the header format:
```
Name,Category,Dosage Form,Strength,Manufacturer,Indication,Classification
```

## Adding Knowledge

Drop any `.txt` file into `data/` (subdirectories work too).
Delete `data/rag_cache.pkl` and restart to rebuild the index.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| OLLAMA_URL | http://localhost:11434 | Ollama server URL |
| OLLAMA_MODEL | gemma3:4b | Model to use |
