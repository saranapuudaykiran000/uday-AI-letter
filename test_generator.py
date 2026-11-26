"""Unit tests for the AI Cover Letter Generator 3-stage pipeline and API."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import CoverLetterHandler, load_samples, process_analyze, process_generate, process_refine
from llm_client import detect_available_providers, load_dotenv
from nlp_utils import analyze_ats_alignment
from prompts import SYSTEM_PROMPT, build_generation_prompt, build_refinement_prompt


SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SAMPLE_RESUME = (SAMPLES_DIR / "sample_resume.md").read_text(encoding="utf-8")
SAMPLE_JD = (SAMPLES_DIR / "sample_jd.md").read_text(encoding="utf-8")


class TestCoverLetterGenerator(unittest.TestCase):
    def test_1_ats_keyword_extraction_and_scoring(self) -> None:
        """Stage 1 extracts multi-word skills and computes accurate matched/missing split."""
        analysis = analyze_ats_alignment(SAMPLE_RESUME, SAMPLE_JD)
        self.assertGreater(analysis.match_score, 50.0)
        self.assertIn("retrieval augmented generation", analysis.matched_keywords)
        self.assertIn("prompt engineering", analysis.matched_keywords)
        self.assertIn("kubernetes", analysis.missing_keywords)
        self.assertEqual(analysis.candidate_name, "Maya Lin")
        self.assertTrue(len(analysis.resume_highlights) >= 1)

    def test_2_structured_prompt_anti_hallucination(self) -> None:
        """Stage 2 separates SYSTEM_PROMPT anti-hallucination rules and ATS context."""
        analysis = analyze_ats_alignment(SAMPLE_RESUME, SAMPLE_JD)
        user_prompt = build_generation_prompt(
            resume_text=SAMPLE_RESUME,
            job_description=SAMPLE_JD,
            ats_analysis=analysis,
            tone="technical",
            length="medium",
            company_name="Veridian Labs",
            role_title="Generative AI Software Engineer",
        )
        self.assertIn("NEVER fabricate", SYSTEM_PROMPT)
        self.assertIn("Matched Keywords", user_prompt)
        self.assertIn("Missing Keywords (DO NOT claim or fabricate)", user_prompt)
        self.assertIn("Veridian Labs", user_prompt)
        self.assertIn("technical", user_prompt)

    def test_3_offline_demo_generation_respects_tone_and_grounding(self) -> None:
        """Stage 3 offline demo generates grounded cover letters across tones and lengths."""
        status, data = process_generate(
            {
                "resume_text": SAMPLE_RESUME,
                "job_description": SAMPLE_JD,
                "tone": "concise",
                "length": "short",
                "provider": "demo",
                "company_name": "Veridian Labs",
                "role_title": "Generative AI Software Engineer",
            }
        )
        self.assertEqual(status, 200)
        letter = data["cover_letter"]
        self.assertIn("Maya Lin", letter)
        self.assertIn("Veridian Labs", letter)
        # Missing skills like kubernetes/terraform must not be claimed as candidate experience.
        self.assertNotIn("kubernetes", letter.lower())
        self.assertNotIn("terraform", letter.lower())

    def test_4_human_in_the_loop_refinement(self) -> None:
        """Refinement endpoint incorporates user feedback and builds grounded refinement prompt."""
        analysis = analyze_ats_alignment(SAMPLE_RESUME, SAMPLE_JD)
        _, initial = process_generate(
            {
                "resume_text": SAMPLE_RESUME,
                "job_description": SAMPLE_JD,
                "provider": "demo",
            }
        )
        ref_prompt = build_refinement_prompt(
            current_draft=initial["cover_letter"],
            feedback="Emphasize prompt engineering and add bullet highlights.",
            resume_text=SAMPLE_RESUME,
            job_description=SAMPLE_JD,
            ats_analysis=analysis,
        )
        self.assertIn("REFINEMENT TASK", ref_prompt)

        status, refined = process_refine(
            {
                "current_draft": initial["cover_letter"],
                "feedback": "Emphasize prompt engineering and add bullet highlights.",
                "resume_text": SAMPLE_RESUME,
                "job_description": SAMPLE_JD,
                "provider": "demo",
            }
        )
        self.assertEqual(status, 200)
        self.assertIn("prompt engineering", refined["cover_letter"].lower())
        self.assertIn("- ", refined["cover_letter"])

    def test_5_http_handler_and_api_routes(self) -> None:
        """HTTP handler serves Web Studio HTML, samples, and JSON API routes."""
        def invoke_handler(method: str, path: str, body: dict | None = None) -> tuple[int, str]:
            raw_body = json.dumps(body).encode("utf-8") if body is not None else b""
            headers = f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
            if body is not None:
                headers += f"Content-Type: application/json\r\nContent-Length: {len(raw_body)}\r\n"
            headers += "\r\n"

            rfile = io.BytesIO(headers.encode("utf-8") + raw_body)
            wfile = io.BytesIO()

            handler = object.__new__(CoverLetterHandler)
            handler.rfile = rfile
            handler.wfile = wfile
            handler.requestline = ""
            handler.request_version = "HTTP/1.1"
            handler.client_address = ("127.0.0.1", 0)
            handler.server = None
            handler.close_connection = True
            handler.raw_requestline = handler.rfile.readline()
            handler.parse_request()
            if method == "GET":
                handler.do_GET()
            else:
                handler.do_POST()

            response_text = wfile.getvalue().decode("utf-8")
            head, _, payload = response_text.partition("\r\n\r\n")
            status_code = int(head.splitlines()[0].split()[1])
            return status_code, payload

        status_html, html = invoke_handler("GET", "/")
        self.assertEqual(status_html, 200)
        self.assertIn("AI Cover Letter Generator", html)

        status_health, health_raw = invoke_handler("GET", "/api/health")
        self.assertEqual(status_health, 200)
        self.assertEqual(json.loads(health_raw)["status"], "ok")

        status_samples, samples_raw = invoke_handler("GET", "/api/samples")
        self.assertEqual(status_samples, 200)
        self.assertIn("Maya Lin", json.loads(samples_raw)["resume_text"])

        status_analyze, analyze_raw = invoke_handler(
            "POST",
            "/api/analyze",
            {"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JD},
        )
        self.assertEqual(status_analyze, 200)
        self.assertIn("ats", json.loads(analyze_raw))

        status_gen, gen_raw = invoke_handler(
            "POST",
            "/api/generate",
            {
                "resume_text": SAMPLE_RESUME,
                "job_description": SAMPLE_JD,
                "provider": "demo",
                "tone": "professional",
                "length": "medium",
            },
        )
        self.assertEqual(status_gen, 200)
        gen_data = json.loads(gen_raw)
        self.assertIn("cover_letter", gen_data)
        self.assertIn("system_prompt", gen_data)
        self.assertIn("user_prompt", gen_data)

        # Also validate direct helper exports.
        samples = load_samples()
        self.assertTrue(samples["resume_text"] and samples["job_description"])
        analyze_code, _ = process_analyze({"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JD})
        self.assertEqual(analyze_code, 200)

    def test_6_groq_provider_routing_and_health(self) -> None:
        """Groq provider routes to Groq's OpenAI-compatible endpoint and reports in health."""
        providers = detect_available_providers()
        self.assertIn("groq", providers)
        self.assertEqual(providers["groq"]["default_model"], "llama-3.3-70b-versatile")

        captured: dict = {}

        def fake_post(url: str, payload: dict, headers: dict | None = None, timeout: int = 45) -> dict:
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = headers
            return {"choices": [{"message": {"content": "Dear Hiring Manager,\n\nGrounded Groq draft."}}]}

        with patch("llm_client._http_post_json", side_effect=fake_post):
            status, data = process_generate(
                {
                    "resume_text": SAMPLE_RESUME,
                    "job_description": SAMPLE_JD,
                    "provider": "groq",
                    "api_key": "gsk_test_key",
                    "allow_fallback": False,
                }
            )

        self.assertEqual(status, 200)
        self.assertEqual(data["provider"], "groq")
        self.assertEqual(data["model"], "llama-3.3-70b-versatile")
        self.assertEqual(captured["url"], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(captured["headers"], {"Authorization": "Bearer gsk_test_key"})
        self.assertIn("Grounded Groq draft.", data["cover_letter"])

    def test_7_dotenv_loading(self) -> None:
        """load_dotenv reads unquoted, quoted, and exported vars without overwriting existing env."""
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "# Comment line\n"
                "TEST_DOTENV_GROQ='gsk_from_dotenv'\n"
                'export TEST_DOTENV_MODEL="llama-3.3-70b-versatile"\n'
                "TEST_DOTENV_EXISTING=from_file\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TEST_DOTENV_EXISTING": "already_set"}, clear=False):
                os.environ.pop("TEST_DOTENV_GROQ", None)
                os.environ.pop("TEST_DOTENV_MODEL", None)
                load_dotenv(env_file)
                self.assertEqual(os.environ.get("TEST_DOTENV_GROQ"), "gsk_from_dotenv")
                self.assertEqual(os.environ.get("TEST_DOTENV_MODEL"), "llama-3.3-70b-versatile")
                self.assertEqual(os.environ.get("TEST_DOTENV_EXISTING"), "already_set")


if __name__ == "__main__":
    unittest.main()
