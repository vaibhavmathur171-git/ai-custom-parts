# AI Custom Parts

A Streamlit web app for designing custom 3D-printable parts (brackets, holders, hooks) by conversation. The user describes a problem in plain English, the agent routes them to the right parametric template, gathers measurements, and produces an STL plus STEP file with manufacturing checks tuned to how the part will be made.

## Run

```bash
# 1. Install (creates and uses a venv)
uv pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# edit .env and put your ANTHROPIC_API_KEY in it

# 3. Start the app
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

## Tests

```bash
pytest tests/
```

The full suite includes live-API e2e flows — those skip themselves automatically if `ANTHROPIC_API_KEY` is unset.

## Environment variables

- `ANTHROPIC_API_KEY` — required for the conversation agent. Loaded from `.env` via `python-dotenv`.
- `HIDE_MODEL` — controls the per-turn model-routing display (the small `🧠 opus|sonnet|haiku` pill under each assistant message and the running counter in the sidebar). **Visible by default** — they're how you can see the routing actually happening. Set `HIDE_MODEL=1` for a cleaner UI.

## Architecture (one-line)

LLM does intake (routing → production context → parameters), never generates geometry. Hand-coded parametric templates own the geometry. Per-turn model routing picks Opus / Sonnet / Haiku based on what the agent is doing on that turn.
