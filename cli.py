"""Command-line interface for the AI Cover Letter Generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_client import SUPPORTED_PROVIDERS, generate_text
from nlp_utils import analyze_ats_alignment
from prompts import SYSTEM_PROMPT, build_generation_prompt, build_refinement_prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and refine ATS-aligned cover letters from a resume and job description."
    )
    parser.add_argument("--resume", required=True, help="Path to candidate resume (.txt or .md)")
    parser.add_argument("--jd", required=True, help="Path to target job description (.txt or .md)")
    parser.add_argument(
        "--tone",
        default="professional",
        choices=("professional", "enthusiastic", "technical", "concise"),
        help="Cover letter tone (default: professional)",
    )
    parser.add_argument(
        "--length",
        default="medium",
        choices=("short", "medium", "detailed"),
        help="Target length (default: medium)",
    )
    parser.add_argument("--temperature", type=float, default=0.4, help="LLM sampling temperature (0.0 - 1.5)")
    parser.add_argument("--provider", default="demo", choices=SUPPORTED_PROVIDERS, help="LLM provider (default: demo)")
    parser.add_argument("--model", default="", help="Optional provider model override")
    parser.add_argument("--company", default="", help="Optional target company name")
    parser.add_argument("--role", default="", help="Optional target role title")
    parser.add_argument("--refine", default="", help="Optional refinement feedback to run a second pass")
    parser.add_argument("--output", default="", help="Optional file path to write the generated cover letter")
    parser.add_argument("--show-prompt", action="store_true", help="Print the engineered system and user prompts")
    parser.add_argument("--json", action="store_true", help="Output structured JSON instead of formatted text")

    args = parser.parse_args(argv)
    resume_path = Path(args.resume)
    jd_path = Path(args.jd)

    if not resume_path.exists():
        print(f"Error: Resume file not found: {resume_path}", file=sys.stderr)
        return 1
    if not jd_path.exists():
        print(f"Error: Job description file not found: {jd_path}", file=sys.stderr)
        return 1

    resume_text = resume_path.read_text(encoding="utf-8")
    jd_text = jd_path.read_text(encoding="utf-8")

    # Stage 1: ATS Keyword & Alignment Analysis
    ats = analyze_ats_alignment(resume_text, jd_text)

    # Stage 2: Structured Prompt Engineering
    user_prompt = build_generation_prompt(
        resume_text=resume_text,
        job_description=jd_text,
        ats_analysis=ats,
        tone=args.tone,
        length=args.length,
        company_name=args.company,
        role_title=args.role,
    )

    # Stage 3: Multi-Provider Generation
    result = generate_text(
        provider=args.provider,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        resume_text=resume_text,
        job_description=jd_text,
        ats_analysis=ats,
        tone=args.tone,
        length=args.length,
        temperature=args.temperature,
        model=args.model or None,
        company_name=args.company,
        role_title=args.role,
    )

    if args.refine.strip():
        refine_prompt = build_refinement_prompt(
            current_draft=result.cover_letter,
            feedback=args.refine,
            resume_text=resume_text,
            job_description=jd_text,
            ats_analysis=ats,
            tone=args.tone,
            length=args.length,
        )
        result = generate_text(
            provider=args.provider,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=refine_prompt,
            resume_text=resume_text,
            job_description=jd_text,
            ats_analysis=ats,
            tone=args.tone,
            length=args.length,
            temperature=args.temperature,
            model=args.model or None,
            current_draft=result.cover_letter,
            feedback=args.refine,
        )

    if args.output:
        Path(args.output).write_text(result.cover_letter, encoding="utf-8")

    if args.json:
        print(json.dumps({"ats": ats.to_dict(), **result.to_dict()}, indent=2))
        return 0

    print("=== ATS ALIGNMENT SUMMARY ===")
    print(f"Match Score : {ats.match_score}%")
    print(f"Matched     : {', '.join(ats.matched_keywords) or 'None'}")
    print(f"Missing     : {', '.join(ats.missing_keywords) or 'None'}")
    print(f"Provider    : {result.provider} ({result.model})")
    if result.notice:
        print(f"Note        : {result.notice}")

    if args.show_prompt:
        print("\n=== SYSTEM PROMPT ===")
        print(result.system_prompt)
        print("\n=== ENGINEERED USER PROMPT ===")
        print(result.user_prompt)

    print("\n=== GENERATED COVER LETTER ===\n")
    print(result.cover_letter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
