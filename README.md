# AI Cover Letter Generator

A full-stack Generative AI application that creates tailored, ATS-aligned cover letters from a candidate's resume and a target job description using a **3-stage hybrid NLP + LLM pipeline**—implemented purely with the **Python 3 standard library** and a modern HTML/CSS/JS Web Studio UI.

## Features
- **Stage 1 — ATS Keyword & Alignment Analysis (`nlp_utils.py`)**: Extracts multi-word skills (e.g., `retrieval augmented generation`, `prompt engineering`) and technical terms from the job description, matches them (with alias support) against the resume, computes an ATS match score (%), and extracts verified resume metrics.
- **Stage 2 — Structured Prompt Engineering (`prompts.py`)**: Separates a role-based `SYSTEM_PROMPT` (with strict anti-hallucination rules) from a context-enriched user prompt containing ATS overlap, tone (`professional`, `enthusiastic`, `technical`, `concise`), and length (`short`, `medium`, `detailed`) constraints.
- **Stage 3 — Multi-Provider Generation & Refinement (`llm_client.py`, `app.py`)**: Supports **Google Gemini**, **OpenAI**, **Groq**, **Local Ollama**, and an **Offline Smart Demo** fallback (zero API key required), plus human-in-the-loop refinement (`/api/refine`).
- **Web Studio UI (`index.html`, `styles.css`, `app.js`)**: `.txt`/`.md` upload and export, print-to-PDF support, live ATS keyword pills, temperature slider, and a Prompt Inspector showing exact system/user prompts.
- **CLI (`cli.py`)**: Full pipeline access from the terminal with JSON, prompt inspection, and file export options.

## Quick Start

### 1. Run the Web Studio
No third-party dependencies are required (`Python 3.9+` standard library only):

```bash
python3 app.py --port 8000
```

Open `http://127.0.0.1:8000` in your browser, click **Load Sample Inputs**, and click **Generate Cover Letter**.

### 2. Run via CLI
```bash
python3 cli.py \
  --resume samples/sample_resume.md \
  --jd samples/sample_jd.md \
  --tone technical \
  --length medium \
  --show-prompt
```

### 3. Optional Live LLM Providers
Set an environment variable (or supply a session key in the Web Studio's advanced panel):
- **Google Gemini**: `export GEMINI_API_KEY="your-key"` (`--provider gemini`)
- **OpenAI**: `export OPENAI_API_KEY="your-key"` (`--provider openai`)
- **Groq**: `export GROQ_API_KEY="your-key"` (`--provider groq`)
- **Local Ollama**: Run Ollama locally on `http://localhost:11434` (`--provider ollama`)

If a live provider is unconfigured or unreachable, the app safely falls back to the **Offline Smart Demo** engine.

### 4. Run Unit Tests
```bash
python3 -m unittest test_generator.py -v
```
