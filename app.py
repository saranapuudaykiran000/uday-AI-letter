"""Python 3 standard-library web server and API for AI Cover Letter Generator."""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from llm_client import detect_available_providers, generate_text
from nlp_utils import analyze_ats_alignment
from prompts import SYSTEM_PROMPT, build_generation_prompt, build_refinement_prompt, validate_length, validate_tone

BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "samples"

STATIC_FILES: Dict[str, Tuple[Path, str]] = {
    "/": (BASE_DIR / "index.html", "text/html; charset=utf-8"),
    "/index.html": (BASE_DIR / "index.html", "text/html; charset=utf-8"),
    "/styles.css": (BASE_DIR / "styles.css", "text/css; charset=utf-8"),
    "/app.js": (BASE_DIR / "app.js", "application/javascript; charset=utf-8"),
}


def _clamp_temperature(value: Any, default: float = 0.4) -> float:
    try:
        temp = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.5, temp))


def process_analyze(payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    resume_text = str(payload.get("resume_text") or "").strip()
    job_description = str(payload.get("job_description") or "").strip()
    if not resume_text or not job_description:
        return HTTPStatus.BAD_REQUEST, {"error": "Both resume_text and job_description are required."}
    ats = analyze_ats_alignment(resume_text, job_description)
    return HTTPStatus.OK, {"ats": ats.to_dict()}


def process_generate(payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    resume_text = str(payload.get("resume_text") or "").strip()
    job_description = str(payload.get("job_description") or "").strip()
    if not resume_text or not job_description:
        return HTTPStatus.BAD_REQUEST, {"error": "Both resume_text and job_description are required."}

    tone = validate_tone(str(payload.get("tone") or "professional"))
    length = validate_length(str(payload.get("length") or "medium"))
    temperature = _clamp_temperature(payload.get("temperature"), default=0.4)
    provider = str(payload.get("provider") or "demo").strip().lower()
    model = str(payload.get("model") or "").strip() or None
    api_key = str(payload.get("api_key") or "").strip() or None
    company_name = str(payload.get("company_name") or "").strip()
    role_title = str(payload.get("role_title") or "").strip()
    allow_fallback = bool(payload.get("allow_fallback", True))

    ats = analyze_ats_alignment(resume_text, job_description)
    user_prompt = build_generation_prompt(
        resume_text=resume_text,
        job_description=job_description,
        ats_analysis=ats,
        tone=tone,
        length=length,
        company_name=company_name,
        role_title=role_title,
    )

    try:
        result = generate_text(
            provider=provider,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            resume_text=resume_text,
            job_description=job_description,
            ats_analysis=ats,
            tone=tone,
            length=length,
            temperature=temperature,
            model=model,
            api_key=api_key,
            company_name=company_name,
            role_title=role_title,
            allow_fallback=allow_fallback,
        )
    except Exception as exc:
        return HTTPStatus.BAD_GATEWAY, {"error": f"Generation failed: {exc}"}

    return HTTPStatus.OK, {
        "ats": ats.to_dict(),
        **result.to_dict(),
    }


def process_refine(payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    current_draft = str(payload.get("current_draft") or "").strip()
    feedback = str(payload.get("feedback") or "").strip()
    resume_text = str(payload.get("resume_text") or "").strip()
    job_description = str(payload.get("job_description") or "").strip()
    if not current_draft or not feedback:
        return HTTPStatus.BAD_REQUEST, {"error": "Both current_draft and feedback are required."}
    if not resume_text or not job_description:
        return HTTPStatus.BAD_REQUEST, {"error": "Both resume_text and job_description are required for grounded refinement."}

    tone = validate_tone(str(payload.get("tone") or "professional"))
    length = validate_length(str(payload.get("length") or "medium"))
    temperature = _clamp_temperature(payload.get("temperature"), default=0.35)
    provider = str(payload.get("provider") or "demo").strip().lower()
    model = str(payload.get("model") or "").strip() or None
    api_key = str(payload.get("api_key") or "").strip() or None
    allow_fallback = bool(payload.get("allow_fallback", True))

    ats = analyze_ats_alignment(resume_text, job_description)
    user_prompt = build_refinement_prompt(
        current_draft=current_draft,
        feedback=feedback,
        resume_text=resume_text,
        job_description=job_description,
        ats_analysis=ats,
        tone=tone,
        length=length,
    )

    try:
        result = generate_text(
            provider=provider,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            resume_text=resume_text,
            job_description=job_description,
            ats_analysis=ats,
            tone=tone,
            length=length,
            temperature=temperature,
            model=model,
            api_key=api_key,
            current_draft=current_draft,
            feedback=feedback,
            allow_fallback=allow_fallback,
        )
    except Exception as exc:
        return HTTPStatus.BAD_GATEWAY, {"error": f"Refinement failed: {exc}"}

    return HTTPStatus.OK, {
        "ats": ats.to_dict(),
        **result.to_dict(),
    }


def load_samples() -> Dict[str, str]:
    resume_path = SAMPLES_DIR / "sample_resume.md"
    jd_path = SAMPLES_DIR / "sample_jd.md"
    return {
        "resume_text": resume_path.read_text(encoding="utf-8") if resume_path.exists() else "",
        "job_description": jd_path.read_text(encoding="utf-8") if jd_path.exists() else "",
    }


class CoverLetterHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving static studio assets and JSON API routes."""

    server_version = "AICoverLetterStudio/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Keep server output clean and concise.
        return

    def _send_json(self, status: int, data: Dict[str, Any]) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length <= 0 or length > 2_000_000:
            return None
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok", "providers": detect_available_providers()})
            return

        if path == "/api/samples":
            self._send_json(HTTPStatus.OK, load_samples())
            return

        if path in STATIC_FILES:
            file_path, content_type = STATIC_FILES[path]
            if not file_path.exists():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Missing asset: {file_path.name}"})
                return
            content = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        payload = self._read_json_body()
        if payload is None:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid or missing JSON payload."})
            return

        if path == "/api/analyze":
            status, data = process_analyze(payload)
            self._send_json(status, data)
            return
        if path == "/api/generate":
            status, data = process_generate(payload)
            self._send_json(status, data)
            return
        if path == "/api/refine":
            status, data = process_refine(payload)
            self._send_json(status, data)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), CoverLetterHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI Cover Letter Generator Web Studio.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")), help="Bind port (default: 8000)")
    args = parser.parse_args()

    server = create_server(args.host, args.port)
    print(f"AI Cover Letter Generator running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
