"""Streamlit frontend for the AI Cover Letter Generator.

Wraps the existing 3-stage NLP + LLM pipeline (nlp_utils, prompts, llm_client)
with a Streamlit UI so the app runs correctly on Streamlit Cloud.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

# ── load .env if present (Streamlit Cloud uses Secrets instead, but local .env still works) ──
from llm_client import detect_available_providers, generate_text, load_dotenv
from nlp_utils import analyze_ats_alignment
from prompts import SYSTEM_PROMPT, build_generation_prompt, build_refinement_prompt, validate_length, validate_tone

load_dotenv()

# ── Streamlit Cloud Secrets → env vars ──────────────────────────────────────────────────────
for _key in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
    if _key not in os.environ and hasattr(st, "secrets") and _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

# ── Page config ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Cover Letter Generator",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


def load_samples() -> tuple[str, str]:
    r = SAMPLES_DIR / "sample_resume.md"
    j = SAMPLES_DIR / "sample_jd.md"
    return (
        r.read_text(encoding="utf-8") if r.exists() else "",
        j.read_text(encoding="utf-8") if j.exists() else "",
    )


# ── Custom CSS ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .pill-match {
        display:inline-block; background:#1a3a1a; color:#4ade80;
        border:1px solid rgba(74,222,128,.35); border-radius:999px;
        padding:2px 10px; margin:2px 3px; font-size:.78rem;
    }
    .pill-miss {
        display:inline-block; background:#3a1a1a; color:#fbbf24;
        border:1px solid rgba(251,191,36,.35); border-radius:999px;
        padding:2px 10px; margin:2px 3px; font-size:.78rem;
    }
    .stTextArea textarea { font-family: monospace; font-size: .85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────────────────────────────────────
st.markdown("# ✉️ AI Cover Letter Generator")
st.caption("Hybrid NLP + LLM 3-Stage Pipeline · ATS-Grounded Studio")
st.divider()

# ── Sidebar: Controls ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Generation Controls")

    provider_options = {
        "demo": "Offline Smart Demo (No API Key)",
        "groq": "Groq",
        "gemini": "Google Gemini",
        "openai": "OpenAI",
        "ollama": "Local Ollama",
    }
    provider = st.selectbox(
        "LLM Provider",
        options=list(provider_options.keys()),
        format_func=lambda k: provider_options[k],
    )

    tone = st.selectbox(
        "Tone",
        options=["professional", "enthusiastic", "technical", "concise"],
        format_func=lambda x: x.title(),
    )

    length = st.selectbox(
        "Length",
        options=["short", "medium", "detailed"],
        index=1,
        format_func=lambda x: {"short": "Short (~150–200w)", "medium": "Medium (~250–320w)", "detailed": "Detailed (~350–450w)"}[x],
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.2, value=0.4, step=0.1)

    with st.expander("🔑 Provider Overrides (optional)"):
        model_override = st.text_input("Model Override", placeholder="e.g. llama-3.3-70b-versatile")
        api_key_input = st.text_input(
            "Session API Key",
            type="password",
            placeholder="Paste key here if not set in Secrets/env",
        )

    st.divider()
    # Provider status
    providers_status = detect_available_providers()
    ready = [v["label"] for k, v in providers_status.items() if v.get("configured")]
    st.success(f"Ready: {', '.join(ready)}")

# ── Main columns ────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1️⃣ Candidate & Role Inputs")

    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        role_title = st.text_input("Target Role (optional)", placeholder="e.g. Generative AI Engineer")
    with meta_col2:
        company_name = st.text_input("Target Company (optional)", placeholder="e.g. Acme AI Labs")

    # Load sample button
    if st.button("📂 Load Sample Inputs", use_container_width=True):
        sr, sj = load_samples()
        st.session_state["resume_text"] = sr
        st.session_state["jd_text"] = sj
        st.success("Sample resume and job description loaded!")

    resume_text = st.text_area(
        "Candidate Resume",
        value=st.session_state.get("resume_text", ""),
        height=280,
        placeholder="Paste your resume here...",
        key="resume_input",
    )
    # Also support file upload
    resume_file = st.file_uploader("Upload Resume (.txt / .md)", type=["txt", "md"], key="resume_file")
    if resume_file:
        resume_text = resume_file.read().decode("utf-8")
        st.session_state["resume_text"] = resume_text

    jd_text = st.text_area(
        "Target Job Description",
        value=st.session_state.get("jd_text", ""),
        height=280,
        placeholder="Paste the job description here...",
        key="jd_input",
    )
    jd_file = st.file_uploader("Upload Job Description (.txt / .md)", type=["txt", "md"], key="jd_file")
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
        st.session_state["jd_text"] = jd_text

    # Action buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        analyze_clicked = st.button("🔍 Analyze ATS Alignment", use_container_width=True)
    with btn_col2:
        generate_clicked = st.button("✨ Generate Cover Letter", type="primary", use_container_width=True)

with col_right:
    st.subheader("2️⃣ ATS Keyword & Alignment")

    # ── ATS Analysis ────────────────────────────────────────────────────────────
    if analyze_clicked or generate_clicked:
        if not resume_text.strip() or not jd_text.strip():
            st.error("Please provide both a resume and a job description.")
        else:
            with st.spinner("Running ATS keyword analysis…"):
                ats = analyze_ats_alignment(resume_text, jd_text)
                st.session_state["ats"] = ats

    if "ats" in st.session_state:
        ats = st.session_state["ats"]
        score = ats.match_score
        st.metric("ATS Match Score", f"{score:.0f}%")
        st.progress(min(int(score), 100))

        m_col, x_col = st.columns(2)
        with m_col:
            st.markdown(f"**✅ Matched ({len(ats.matched_keywords)})**")
            if ats.matched_keywords:
                st.markdown(
                    " ".join(f'<span class="pill-match">{k}</span>' for k in ats.matched_keywords),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("None detected yet.")
        with x_col:
            st.markdown(f"**⚠️ Missing ({len(ats.missing_keywords)})**")
            if ats.missing_keywords:
                st.markdown(
                    " ".join(f'<span class="pill-miss">{k}</span>' for k in ats.missing_keywords),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("All keywords matched!")
    else:
        st.info("Run **Analyze ATS Alignment** or **Generate Cover Letter** to see keyword analysis.")

    st.divider()
    st.subheader("3️⃣ Tailored Cover Letter")

    # ── Generation ──────────────────────────────────────────────────────────────
    if generate_clicked and resume_text.strip() and jd_text.strip():
        effective_key = api_key_input.strip() or None
        effective_model = model_override.strip() or None

        with st.spinner("Running 3-stage NLP + LLM pipeline…"):
            ats = analyze_ats_alignment(resume_text, jd_text)
            st.session_state["ats"] = ats
            user_prompt = build_generation_prompt(
                resume_text=resume_text,
                job_description=jd_text,
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
                    job_description=jd_text,
                    ats_analysis=ats,
                    tone=tone,
                    length=length,
                    temperature=temperature,
                    model=effective_model,
                    api_key=effective_key,
                    company_name=company_name,
                    role_title=role_title,
                    allow_fallback=True,
                )
                st.session_state["result"] = result
                st.session_state["user_prompt"] = user_prompt
                if result.notice:
                    st.info(result.notice)
            except Exception as exc:
                st.error(f"Generation failed: {exc}")

    if "result" in st.session_state:
        result = st.session_state["result"]
        st.caption(f"Provider: **{result.provider}** · Model: **{result.model}**")

        draft = st.text_area(
            "Cover Letter (editable)",
            value=result.cover_letter,
            height=360,
            key="cover_letter_draft",
        )

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "⬇️ Download .txt",
                data=draft,
                file_name="cover-letter.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "⬇️ Download .md",
                data=f"# Cover Letter\n\n{draft}\n",
                file_name="cover-letter.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # ── Refinement ──────────────────────────────────────────────────────────
        st.divider()
        st.subheader("🔄 Human-in-the-Loop Refinement")

        chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
        chip_map = {
            "Make shorter": "Make it shorter and more concise.",
            "More technical": "Use a more technical engineering tone highlighting measurable metrics.",
            "More enthusiastic": "Make the opening more enthusiastic and mission-driven.",
            "Add bullets": "Highlight key achievements using concise bullet points.",
        }
        for btn_label, (chip_label, chip_text) in zip(
            [chip_col1, chip_col2, chip_col3, chip_col4], chip_map.items()
        ):
            if btn_label.button(chip_label, use_container_width=True):
                st.session_state["refine_feedback"] = chip_text

        feedback = st.text_input(
            "Refinement instruction",
            value=st.session_state.get("refine_feedback", ""),
            placeholder="e.g. 'Emphasise prompt engineering and trim paragraph 3'",
            key="refine_feedback_input",
        )

        if st.button("🔁 Refine Draft", type="primary", use_container_width=True):
            if not feedback.strip():
                st.warning("Enter a refinement instruction first.")
            else:
                effective_key = api_key_input.strip() or None
                effective_model = model_override.strip() or None
                with st.spinner("Refining draft from feedback…"):
                    ats = st.session_state.get("ats") or analyze_ats_alignment(resume_text, jd_text)
                    refine_prompt = build_refinement_prompt(
                        current_draft=draft,
                        feedback=feedback,
                        resume_text=resume_text,
                        job_description=jd_text,
                        ats_analysis=ats,
                        tone=tone,
                        length=length,
                    )
                    try:
                        refined = generate_text(
                            provider=provider,
                            system_prompt=SYSTEM_PROMPT,
                            user_prompt=refine_prompt,
                            resume_text=resume_text,
                            job_description=jd_text,
                            ats_analysis=ats,
                            tone=tone,
                            length=length,
                            temperature=temperature,
                            model=effective_model,
                            api_key=effective_key,
                            current_draft=draft,
                            feedback=feedback,
                            allow_fallback=True,
                        )
                        st.session_state["result"] = refined
                        st.success("Draft refined! Scroll up to see the updated letter.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Refinement failed: {exc}")

        # ── Prompt Inspector ────────────────────────────────────────────────────
        with st.expander("🔬 Prompt Inspector"):
            st.markdown("**SYSTEM_PROMPT (Anti-Hallucination Rules)**")
            st.code(result.system_prompt, language="text")
            if "user_prompt" in st.session_state:
                st.markdown("**Engineered User Prompt (ATS Context + Constraints)**")
                st.code(st.session_state["user_prompt"], language="text")
    else:
        st.info("Fill in your resume and job description, then click **Generate Cover Letter**.")
