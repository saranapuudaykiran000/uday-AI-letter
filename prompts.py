"""Structured Prompt Engineering templates and builders."""

from __future__ import annotations

from typing import Dict

from nlp_utils import ATSAnalysis

SYSTEM_PROMPT: str = """You are an expert career strategist, technical recruiter, and professional cover letter writer.

Your task is to write a tailored, ATS-aligned cover letter grounded strictly in the candidate's verified resume and the target job description.

STRICT ANTI-HALLUCINATION & GROUNDING RULES:
1. NEVER fabricate skills, tools, job titles, employers, degrees, certifications, dates, or numerical metrics that do not appear in the candidate's resume.
2. NEVER claim the candidate has direct experience with any keyword listed under "Missing Keywords". If relevant, you may frame genuine adjacent or transferable experience from the resume honestly, without claiming expertise in the missing item.
3. Prioritize weaving the "Matched Keywords" and verified resume achievements naturally into compelling narrative paragraphs rather than keyword-stuffing.
4. Follow the requested Tone and Length constraints carefully.
5. Output ONLY the finished cover letter text (salutation, body paragraphs, and sign-off). Do not include markdown code fences, preambles, or meta-commentary."""

TONE_INSTRUCTIONS: Dict[str, str] = {
    "professional": "Balanced, polished, confident, and formal business tone suitable for corporate and enterprise hiring managers.",
    "enthusiastic": "Warm, energetic, and mission-driven tone that conveys genuine excitement for the role and company while remaining credible.",
    "technical": "Precision-oriented engineering tone highlighting architecture, systems thinking, measurable impact, and concrete technical execution.",
    "concise": "Direct, crisp, high-signal executive tone with zero fluff, short sentences, and immediate focus on value delivered.",
}

LENGTH_INSTRUCTIONS: Dict[str, str] = {
    "short": "Keep the cover letter tight and focused (~150-200 words, 3 concise paragraphs).",
    "medium": "Write a standard, well-paced cover letter (~250-320 words, 3-4 paragraphs).",
    "detailed": "Write a comprehensive, deeply tailored cover letter (~350-450 words, 4-5 paragraphs) connecting specific resume projects to job requirements.",
}


def validate_tone(tone: str) -> str:
    key = (tone or "professional").strip().lower()
    return key if key in TONE_INSTRUCTIONS else "professional"


def validate_length(length: str) -> str:
    key = (length or "medium").strip().lower()
    return key if key in LENGTH_INSTRUCTIONS else "medium"


def build_generation_prompt(
    resume_text: str,
    job_description: str,
    ats_analysis: ATSAnalysis,
    tone: str = "professional",
    length: str = "medium",
    company_name: str = "",
    role_title: str = "",
) -> str:
    """Build a context-enriched user prompt combining ATS overlap, constraints, and inputs."""
    tone_key = validate_tone(tone)
    length_key = validate_length(length)

    matched_str = ", ".join(ats_analysis.matched_keywords) if ats_analysis.matched_keywords else "None detected"
    missing_str = ", ".join(ats_analysis.missing_keywords) if ats_analysis.missing_keywords else "None"
    highlights_str = (
        "\n".join(f"- {item}" for item in ats_analysis.resume_highlights)
        if ats_analysis.resume_highlights
        else "- Use concrete roles and projects directly from the resume below."
    )

    target_role = role_title.strip() or "the target role described in the job description"
    target_company = company_name.strip() or "the hiring organization"

    return (
        f"### TARGET APPLICATION CONTEXT\n"
        f"- Candidate Name: {ats_analysis.candidate_name}\n"
        f"- Target Role: {target_role}\n"
        f"- Target Company: {target_company}\n"
        f"- Requested Tone ({tone_key}): {TONE_INSTRUCTIONS[tone_key]}\n"
        f"- Requested Length ({length_key}): {LENGTH_INSTRUCTIONS[length_key]}\n\n"
        f"### ATS KEYWORD ALIGNMENT ANALYSIS\n"
        f"- Current ATS Match Score: {ats_analysis.match_score}%\n"
        f"- Matched Keywords (incorporate naturally): {matched_str}\n"
        f"- Missing Keywords (DO NOT claim or fabricate): {missing_str}\n\n"
        f"### VERIFIED RESUME HIGHLIGHTS\n"
        f"{highlights_str}\n\n"
        f"### CANDIDATE RESUME\n"
        f"{resume_text.strip()}\n\n"
        f"### TARGET JOB DESCRIPTION\n"
        f"{job_description.strip()}\n\n"
        f"Write the tailored cover letter now, honoring every grounding rule above."
    )


def build_refinement_prompt(
    current_draft: str,
    feedback: str,
    resume_text: str,
    job_description: str,
    ats_analysis: ATSAnalysis,
    tone: str = "professional",
    length: str = "medium",
) -> str:
    """Build a human-in-the-loop refinement prompt grounded in the resume and feedback."""
    tone_key = validate_tone(tone)
    length_key = validate_length(length)
    matched_str = ", ".join(ats_analysis.matched_keywords) if ats_analysis.matched_keywords else "None detected"
    missing_str = ", ".join(ats_analysis.missing_keywords) if ats_analysis.missing_keywords else "None"

    return (
        f"### REFINEMENT TASK\n"
        f"Revise the current cover letter draft based on the user's feedback while keeping every claim strictly grounded in the candidate's resume.\n\n"
        f"- User Feedback: {feedback.strip()}\n"
        f"- Target Tone ({tone_key}): {TONE_INSTRUCTIONS[tone_key]}\n"
        f"- Target Length ({length_key}): {LENGTH_INSTRUCTIONS[length_key]}\n"
        f"- Matched Keywords: {matched_str}\n"
        f"- Missing Keywords (DO NOT fabricate): {missing_str}\n\n"
        f"### CURRENT COVER LETTER DRAFT\n"
        f"{current_draft.strip()}\n\n"
        f"### CANDIDATE RESUME (SOURCE OF TRUTH)\n"
        f"{resume_text.strip()}\n\n"
        f"### TARGET JOB DESCRIPTION\n"
        f"{job_description.strip()}\n\n"
        f"Return ONLY the revised cover letter."
    )
