"""Multi-provider LLM client (Gemini, OpenAI, Groq, Ollama, and Offline Smart Demo)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from nlp_utils import ATSAnalysis
from prompts import SYSTEM_PROMPT, validate_length, validate_tone

SUPPORTED_PROVIDERS = ("demo", "gemini", "openai", "groq", "ollama")


def load_dotenv(env_path: Optional[Path] = None) -> None:
    """Load key=value pairs from a local .env file without overriding existing env vars."""
    path = env_path or (Path(__file__).resolve().parent / ".env")
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        os.environ.setdefault(key, val)


load_dotenv()


@dataclass
class GenerationResult:
    """Response returned by an LLM provider or offline fallback."""

    cover_letter: str
    provider: str
    model: str
    system_prompt: str
    user_prompt: str
    fallback_used: bool = False
    notice: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cover_letter": self.cover_letter,
            "provider": self.provider,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "fallback_used": self.fallback_used,
            "notice": self.notice,
        }


def _http_post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: int = 45) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[str, str]:
    key = (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise ValueError("GEMINI_API_KEY is not set.")
    chosen_model = (model or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{chosen_model}:generateContent?key={key}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": float(temperature)},
    }
    resp = _http_post_json(url, payload)
    candidates = resp.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {resp}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text, chosen_model


def _call_openai_compatible(
    provider_label: str,
    base_url: str,
    chosen_model: str,
    key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> tuple[str, str]:
    """Call an OpenAI-compatible /chat/completions endpoint (OpenAI, Groq, etc.)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": chosen_model,
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = _http_post_json(url, payload, headers={"Authorization": f"Bearer {key}"})
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"{provider_label} returned no choices: {resp}")
    text = (choices[0].get("message", {}).get("content") or "").strip()
    if not text:
        raise RuntimeError(f"{provider_label} returned an empty response.")
    return text, chosen_model


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[str, str]:
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY is not set.")
    chosen_model = (model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return _call_openai_compatible("OpenAI", base_url, chosen_model, key, system_prompt, user_prompt, temperature)


def _call_groq(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[str, str]:
    key = (api_key or os.environ.get("GROQ_API_KEY") or "").strip()
    if not key:
        raise ValueError("GROQ_API_KEY is not set.")
    chosen_model = (model or os.environ.get("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
    base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    return _call_openai_compatible("Groq", base_url, chosen_model, key, system_prompt, user_prompt, temperature)


def _call_ollama(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    model: Optional[str] = None,
) -> tuple[str, str]:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    chosen_model = (model or os.environ.get("OLLAMA_MODEL") or "llama3.2").strip()
    url = f"{host}/api/chat"
    payload = {
        "model": chosen_model,
        "stream": False,
        "options": {"temperature": float(temperature)},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = _http_post_json(url, payload, timeout=90)
    text = (resp.get("message", {}).get("content") or "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    return text, chosen_model


def _format_keyword_list(keywords: List[str], limit: int = 5) -> str:
    chosen = keywords[:limit]
    if not chosen:
        return "software engineering and cross-functional delivery"
    if len(chosen) == 1:
        return chosen[0]
    if len(chosen) == 2:
        return f"{chosen[0]} and {chosen[1]}"
    return ", ".join(chosen[:-1]) + f", and {chosen[-1]}"


def _extract_role_and_company(job_description: str, company_name: str, role_title: str) -> tuple[str, str]:
    role = role_title.strip()
    company = company_name.strip()
    if not role:
        m = re.search(r"(?:role|position|title)\s*:\s*(.+)", job_description, re.I)
        if m:
            role = m.group(1).strip().splitlines()[0][:80]
        else:
            first_line = next((ln.strip() for ln in job_description.splitlines() if ln.strip()), "")
            role = first_line[:70] if 3 <= len(first_line) <= 70 else "this position"
    if not company:
        m = re.search(r"(?:company|organization|about)\s*:\s*(.+)", job_description, re.I)
        if m:
            company = m.group(1).strip().splitlines()[0][:60]
        else:
            company = "your team"
    return role, company


def _offline_demo_generate(
    resume_text: str,
    job_description: str,
    ats_analysis: ATSAnalysis,
    tone: str = "professional",
    length: str = "medium",
    company_name: str = "",
    role_title: str = "",
) -> str:
    """Deterministic, resume-grounded Offline Smart Demo cover letter generator."""
    tone_key = validate_tone(tone)
    length_key = validate_length(length)
    role, company = _extract_role_and_company(job_description, company_name, role_title)
    candidate = ats_analysis.candidate_name or "Candidate"
    matched_str = _format_keyword_list(ats_analysis.matched_keywords, limit=5)
    secondary_str = _format_keyword_list(ats_analysis.matched_keywords[5:], limit=4)
    highlights = ats_analysis.resume_highlights

    openings = {
        "professional": (
            f"I am writing to express my strong interest in the {role} opportunity at {company}. "
            f"With hands-on experience in {matched_str}, I am confident in my ability to contribute "
            f"immediately to your engineering and delivery goals."
        ),
        "enthusiastic": (
            f"I am thrilled to apply for the {role} role at {company}! "
            f"Your focus on {matched_str} aligns closely with the work I am most passionate about, "
            f"and I would love to bring that energy and execution track record to your team."
        ),
        "technical": (
            f"I am applying for the {role} position at {company}, bringing production experience across "
            f"{matched_str}. My background centers on designing reliable, measurable systems that translate "
            f"requirements directly into maintainable software."
        ),
        "concise": (
            f"I am applying for the {role} role at {company}. My verified experience in {matched_str} "
            f"maps directly to your core requirements."
        ),
    }

    evidence_lines: List[str] = []
    if highlights:
        if length_key == "short":
            evidence_lines.append(f"In my previous work, I {highlights[0][0].lower() + highlights[0][1:].rstrip('.')}.")
        else:
            lead_in = (
                "Specifically, my resume highlights concrete impact directly relevant to this role:"
                if tone_key != "concise"
                else "Key verified accomplishments include:"
            )
            selected = highlights[: (2 if length_key == "medium" else 3)]
            formatted = " ".join(
                f"{item.rstrip('.')}." for item in selected
            )
            evidence_lines.append(f"{lead_in} {formatted}")
    else:
        evidence_lines.append(
            f"Across my recent projects, I have applied {matched_str} to build dependable solutions "
            f"and collaborate effectively with cross-functional partners."
        )

    paragraphs: List[str] = ["Dear Hiring Manager,", openings[tone_key]]
    paragraphs.extend(evidence_lines)

    if length_key in ("medium", "detailed"):
        if ats_analysis.matched_keywords[5:]:
            paragraphs.append(
                f"Beyond core deliverables, I bring practical depth in {secondary_str}, enabling me to "
                f"ramp up quickly, maintain high code and quality standards, and partner smoothly across product and engineering."
            )
        elif tone_key == "technical":
            paragraphs.append(
                "I prioritize clean interfaces, thorough validation, and observable behavior in production so systems remain adaptable as requirements evolve."
            )
        else:
            paragraphs.append(
                f"I value clear communication, iterative problem-solving, and taking full ownership from initial design through final delivery at {company}."
            )

    if length_key == "detailed":
        if ats_analysis.missing_keywords:
            gap_preview = _format_keyword_list(ats_analysis.missing_keywords, limit=2)
            paragraphs.append(
                f"While my resume focuses on {matched_str} rather than {gap_preview}, my strong foundation in "
                f"rapidly adopting new tools allows me to bridge into adjacent workflows with minimal ramp-up time."
            )
        else:
            paragraphs.append(
                f"Given my {ats_analysis.match_score}% alignment with your posted requirements, I am eager to help {company} accelerate upcoming milestones."
            )

    closings = {
        "professional": f"Thank you for your time and consideration. I welcome the opportunity to discuss how my background can support {company}.",
        "enthusiastic": f"Thank you for considering my application—I would be excited to speak with you about how I can contribute to {company}!",
        "technical": f"Thank you for reviewing my application. I look forward to discussing architecture, past results, and how I can support {company}.",
        "concise": f"Thank you for your consideration. I welcome a conversation regarding the {role} role.",
    }
    paragraphs.append(closings[tone_key])
    paragraphs.append(f"Sincerely,\n{candidate}")
    return "\n\n".join(paragraphs)


def _offline_demo_refine(
    current_draft: str,
    feedback: str,
    resume_text: str,
    job_description: str,
    ats_analysis: ATSAnalysis,
    tone: str,
    length: str,
) -> str:
    """Apply natural-language refinement feedback deterministically in Offline Demo mode."""
    fb = feedback.lower()
    target_tone = validate_tone(tone)
    target_length = validate_length(length)

    if any(w in fb for w in ("shorter", "short", "brief", "concise", "trim", "tighter")):
        target_length = "short"
        if "concise" in fb:
            target_tone = "concise"
    elif any(w in fb for w in ("longer", "detailed", "expand", "more detail", "elaborate")):
        target_length = "detailed"

    if any(w in fb for w in ("enthusiastic", "excited", "warm", "energetic", "passionate")):
        target_tone = "enthusiastic"
    elif any(w in fb for w in ("technical", "engineering", "architecture", "metrics")):
        target_tone = "technical"
    elif any(w in fb for w in ("formal", "professional", "executive")):
        target_tone = "professional"

    regenerated = _offline_demo_generate(
        resume_text=resume_text,
        job_description=job_description,
        ats_analysis=ats_analysis,
        tone=target_tone,
        length=target_length,
    )

    # Check if user explicitly asked to emphasize one of the verified matched keywords.
    emphasized = [
        kw for kw in ats_analysis.matched_keywords
        if kw.lower() in fb
    ]
    if emphasized:
        emph_str = _format_keyword_list(emphasized, limit=3)
        parts = regenerated.split("\n\n")
        if len(parts) >= 3:
            parts.insert(
                2,
                f"Per your focus areas, I want to underscore my direct, verified experience with {emph_str}, "
                f"which I have applied in real-world deliverables documented in my resume.",
            )
            regenerated = "\n\n".join(parts)

    if "bullet" in fb and ats_analysis.resume_highlights:
        parts = regenerated.split("\n\n")
        bullets = "\n".join(f"- {h.rstrip('.')}" for h in ats_analysis.resume_highlights[:3])
        if len(parts) >= 3:
            parts[2] = f"Selected verified highlights from my background:\n{bullets}"
            regenerated = "\n\n".join(parts)

    return regenerated


def detect_available_providers() -> Dict[str, Any]:
    """Report configuration status for each supported LLM provider."""
    gemini_ready = bool((os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip())
    openai_ready = bool((os.environ.get("OPENAI_API_KEY") or "").strip())
    groq_ready = bool((os.environ.get("GROQ_API_KEY") or "").strip())
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return {
        "demo": {"configured": True, "default_model": "offline-smart-demo", "label": "Offline Smart Demo"},
        "gemini": {"configured": gemini_ready, "default_model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), "label": "Google Gemini"},
        "openai": {"configured": openai_ready, "default_model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "label": "OpenAI"},
        "groq": {"configured": groq_ready, "default_model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"), "label": "Groq"},
        "ollama": {"configured": True, "default_model": os.environ.get("OLLAMA_MODEL", "llama3.2"), "host": ollama_host, "label": "Local Ollama"},
    }


def generate_text(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    resume_text: str,
    job_description: str,
    ats_analysis: ATSAnalysis,
    tone: str = "professional",
    length: str = "medium",
    temperature: float = 0.4,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    company_name: str = "",
    role_title: str = "",
    current_draft: Optional[str] = None,
    feedback: Optional[str] = None,
    allow_fallback: bool = True,
) -> GenerationResult:
    """Execute generation or refinement against the selected provider with safe fallback."""
    chosen = (provider or "demo").strip().lower()
    if chosen not in SUPPORTED_PROVIDERS:
        chosen = "demo"

    if chosen == "demo":
        if current_draft is not None and feedback is not None:
            letter = _offline_demo_refine(
                current_draft=current_draft,
                feedback=feedback,
                resume_text=resume_text,
                job_description=job_description,
                ats_analysis=ats_analysis,
                tone=tone,
                length=length,
            )
        else:
            letter = _offline_demo_generate(
                resume_text=resume_text,
                job_description=job_description,
                ats_analysis=ats_analysis,
                tone=tone,
                length=length,
                company_name=company_name,
                role_title=role_title,
            )
        return GenerationResult(
            cover_letter=letter,
            provider="demo",
            model="offline-smart-demo",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_used=False,
            notice="Generated using Offline Smart Demo (no API key required).",
        )

    try:
        if chosen == "gemini":
            text, used_model = _call_gemini(system_prompt, user_prompt, temperature, model=model, api_key=api_key)
        elif chosen == "openai":
            text, used_model = _call_openai(system_prompt, user_prompt, temperature, model=model, api_key=api_key)
        elif chosen == "groq":
            text, used_model = _call_groq(system_prompt, user_prompt, temperature, model=model, api_key=api_key)
        else:
            text, used_model = _call_ollama(system_prompt, user_prompt, temperature, model=model)
        return GenerationResult(
            cover_letter=text,
            provider=chosen,
            model=used_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_used=False,
        )
    except (ValueError, RuntimeError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        if not allow_fallback:
            raise
        if current_draft is not None and feedback is not None:
            letter = _offline_demo_refine(
                current_draft=current_draft,
                feedback=feedback,
                resume_text=resume_text,
                job_description=job_description,
                ats_analysis=ats_analysis,
                tone=tone,
                length=length,
            )
        else:
            letter = _offline_demo_generate(
                resume_text=resume_text,
                job_description=job_description,
                ats_analysis=ats_analysis,
                tone=tone,
                length=length,
                company_name=company_name,
                role_title=role_title,
            )
        return GenerationResult(
            cover_letter=letter,
            provider="demo",
            model="offline-smart-demo",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_used=True,
            notice=f"{chosen} unavailable ({exc}); fell back to Offline Smart Demo.",
        )
