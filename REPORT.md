# Project Report: AI Cover Letter Generator

## 1. Problem Statement & Motivation
Generic LLM prompts for cover letters frequently produce vague prose, invent skills not present on the candidate's resume, and miss critical Applicant Tracking System (ATS) terminology from the job posting. This project solves those failure modes through a deterministic **3-stage hybrid NLP + LLM pipeline** that grounds generation in verified resume facts while giving users interactive control over tone, length, sampling temperature, and iterative refinement.

## 2. System Architecture (3-Stage Pipeline)

### Stage 1: ATS Keyword & Alignment Analysis (`nlp_utils.py`)
- Normalizes text and matches curated multi-word phrases (such as `retrieval augmented generation`, `prompt engineering`, `large language models`, `unit testing`) before single-token technical skills and frequency-ranked domain terms.
- Applies alias normalization (e.g., `rag` ↔ `retrieval augmented generation`, `llm` ↔ `large language models`, `js` ↔ `javascript`).
- Computes `match_score` (%), `matched_keywords`, `missing_keywords`, and `resume_highlights` (quantified achievement lines).

### Stage 2: Structured Prompt Engineering (`prompts.py`)
- Separates `SYSTEM_PROMPT` (role framing + strict anti-hallucination rules prohibiting fabricated skills, employers, or metrics) from the dynamic user prompt.
- Injects ATS alignment context (`matched_keywords` to weave in naturally vs. `missing_keywords` explicitly marked **DO NOT claim or fabricate**), verified resume highlights, tone rules (`professional`, `enthusiastic`, `technical`, `concise`), and length targets (`short`, `medium`, `detailed`).
- Supports `build_refinement_prompt()` for iterative human-in-the-loop revisions.

### Stage 3: Multi-Provider Generation & Refinement (`llm_client.py`, `app.py`)
- Uses Python's standard library (`urllib.request`, `http.server`) with zero mandatory third-party packages.
- Integrates Google Gemini (`generateContent`), OpenAI (`/v1/chat/completions`), local Ollama (`/api/chat`), and an **Offline Smart Demo** engine so the full application works out of the box without API credentials.
- Exposes `/api/analyze`, `/api/generate`, `/api/refine`, `/api/samples`, and `/api/health`.

## 3. Web Studio & CLI Deliverables
- **Web Studio (`index.html`, `styles.css`, `app.js`)**: Upload `.txt`/`.md` resumes and JDs, inspect ATS matched/missing keyword pills, adjust temperature, inspect exact prompts in the **Prompt Inspector**, iterate via natural-language feedback, and export `.txt`, `.md`, or Print-to-PDF.
- **CLI (`cli.py`)**: Scriptable terminal interface supporting all providers, tones, lengths, refinement, and JSON output.

## 4. Verification
Automated unit tests in `test_generator.py` validate keyword extraction and ATS scoring, prompt grounding rules, offline generation across tone/length modes, human-in-the-loop refinement, and end-to-end HTTP API endpoints.
