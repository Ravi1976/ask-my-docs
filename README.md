# Ask My Docs (simple RAG chatbot)

Put your .txt / .md / .pdf files in `docs/`, then ask questions about them.

## Setup
    python -m venv venv
    venv\Scripts\activate          (Windows)  |  source venv/bin/activate  (Mac/Linux)
    pip install -r requirements.txt

## Run
    python app.py --provider mock                      # no AI, tests retrieval
    python app.py --provider ollama                    # free local model
    python app.py --provider anthropic                 # Claude API (set ANTHROPIC_API_KEY)
    python app.py --provider mock --question "How many sick leave days?"
