"""Ask My Docs - a simple RAG (Retrieval-Augmented Generation) chatbot.

How it works:
  1. LOAD    read your .txt / .md / .pdf files from the docs/ folder
  2. CHUNK   cut each file into small overlapping pieces
  3. INDEX   turn every chunk into a TF-IDF vector (simple keyword-based search)
  4. SEARCH  find the chunks most similar to your question
  5. ASK     send those chunks + your question to an AI model
  6. ANSWER  the model answers using ONLY those chunks, and we show the sources

Providers:  mock (no AI) | ollama (free, local) | anthropic (Claude API)
"""
import argparse
import os
import sys
from pathlib import Path

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 600   # characters per chunk
OVERLAP = 100      # characters shared between neighbouring chunks
TOP_K = 3          # how many chunks to send to the AI

PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find that in the documents."
Keep the answer short and clear.

CONTEXT:
{context}

QUESTION:
{question}
"""


# ---------- 1. LOAD ----------
def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(folder: Path) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() in {".txt", ".md", ".pdf"}:
            text = read_file(path).strip()
            if text:
                docs.append((path.name, text))
    return docs


# ---------- 2. CHUNK ----------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


# ---------- 3. INDEX ----------
def build_index(docs):
    chunks = []
    for name, text in docs:
        for piece in chunk_text(text):
            chunks.append({"source": name, "text": piece})
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return chunks, vectorizer, matrix


# ---------- 4. SEARCH ----------
def retrieve(question, chunks, vectorizer, matrix, k=TOP_K):
    scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
    best = scores.argsort()[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in best if scores[i] > 0]


# ---------- 5. ASK (AI providers) ----------
def call_mock(prompt: str) -> str:
    context = prompt.split("CONTEXT:")[1].split("QUESTION:")[0].strip()
    return "[MOCK MODE - no AI used] Best matching text I found:\n" + context[:400]


def call_ollama(prompt: str) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    resp = requests.post(url, json={"model": model, "stream": False,
                         "messages": [{"role": "user", "content": prompt}]}, timeout=600)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def call_anthropic(prompt: str) -> str:
    from anthropic import Anthropic
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    msg = Anthropic().messages.create(model=model, max_tokens=800,
                                      messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


PROVIDERS = {"mock": call_mock, "ollama": call_ollama, "anthropic": call_anthropic}


# ---------- 6. ANSWER ----------
def answer(question, index, provider):
    chunks, vectorizer, matrix = index
    hits = retrieve(question, chunks, vectorizer, matrix)
    if not hits:   # guardrail: do not call the AI when nothing relevant was found
        return "I could not find that in the documents.", []
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h, _ in hits)
    reply = PROVIDERS[provider](PROMPT.format(context=context, question=question))
    sources = sorted({h["source"] for h, _ in hits})
    return reply, sources


def main():
    ap = argparse.ArgumentParser(description="Chat with your own documents.")
    ap.add_argument("--provider", choices=PROVIDERS, default="mock")
    ap.add_argument("--docs", default="docs", help="Folder with .txt/.md/.pdf files")
    ap.add_argument("--question", help="Ask one question and exit")
    args = ap.parse_args()

    docs = load_documents(Path(args.docs))
    if not docs:
        sys.exit(f"No .txt/.md/.pdf files found in '{args.docs}'. Add some and try again.")
    index = build_index(docs)
    print(f"Loaded {len(docs)} file(s), {len(index[0])} chunks. Provider: {args.provider}")

    def ask(q):
        reply, sources = answer(q, index, args.provider)
        print(f"\n{reply}")
        if sources:
            print(f"\nSources: {', '.join(sources)}")

    if args.question:
        ask(args.question)
        return
    print("Type your question (or 'exit' to quit).")
    while True:
        q = input("\nAsk> ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        ask(q)


if __name__ == "__main__":
    main()
