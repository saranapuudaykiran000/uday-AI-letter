"""ATS Keyword & Alignment Analysis utilities using Python standard library."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple

# Curated multi-word technical, AI/ML, software engineering, and product skills.
MULTI_WORD_SKILLS: Tuple[str, ...] = (
    "retrieval augmented generation",
    "prompt engineering",
    "large language models",
    "large language model",
    "natural language processing",
    "generative ai",
    "machine learning",
    "deep learning",
    "reinforcement learning",
    "computer vision",
    "fine tuning",
    "vector database",
    "vector databases",
    "semantic search",
    "context engineering",
    "agentic workflows",
    "function calling",
    "structured outputs",
    "model evaluation",
    "human in the loop",
    "full stack",
    "rest api",
    "rest apis",
    "restful apis",
    "graphql api",
    "system design",
    "distributed systems",
    "microservices architecture",
    "cloud computing",
    "continuous integration",
    "continuous deployment",
    "unit testing",
    "integration testing",
    "test driven development",
    "version control",
    "data engineering",
    "data pipelines",
    "feature engineering",
    "time series",
    "a/b testing",
    "project management",
    "agile methodology",
    "cross functional",
    "technical writing",
    "responsive design",
    "user experience",
    "code review",
    "performance optimization",
)

# Canonical single-word technical and domain skills to recognize reliably.
KNOWN_SINGLE_SKILLS: Set[str] = {
    "python", "javascript", "typescript", "java", "go", "golang", "rust", "c++",
    "sql", "nosql", "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis",
    "html", "css", "react", "nextjs", "vue", "angular", "svelte", "node", "nodejs",
    "fastapi", "flask", "django", "express", "spring", "docker", "kubernetes", "k8s",
    "aws", "gcp", "azure", "terraform", "linux", "git", "github", "gitlab", "ci/cd",
    "llm", "llms", "rag", "nlp", "ml", "ai", "genai", "embeddings", "transformers",
    "pytorch", "tensorflow", "langchain", "llamaindex", "openai", "gemini", "ollama",
    "huggingface", "pandas", "numpy", "scikit-learn", "spark", "kafka", "airflow",
    "graphql", "grpc", "websockets", "microservices", "serverless", "oauth", "jwt",
    "testing", "pytest", "unittest", "jest", "playwright", "cypress", "monitoring",
    "observability", "prometheus", "grafana", "figma", "agile", "scrum", "analytics",
    "evaluation", "guardrails", "hallucination", "tokenization", "inference",
}

# Equivalent alias groups so e.g. "rag" matches "retrieval augmented generation"
# and canonical variants aren't double-counted as separate keywords.
SKILL_ALIASES: Dict[str, Set[str]] = {
    "retrieval augmented generation": {"rag", "retrieval-augmented generation"},
    "large language models": {"llm", "llms", "large language model"},
    "large language model": {"llm", "llms", "large language models"},
    "natural language processing": {"nlp"},
    "machine learning": {"ml"},
    "generative ai": {"genai", "gen ai"},
    "javascript": {"js"},
    "typescript": {"ts"},
    "kubernetes": {"k8s"},
    "postgresql": {"postgres"},
    "continuous integration": {"ci/cd", "ci"},
    "continuous deployment": {"ci/cd", "cd"},
    "rest api": {"rest apis", "restful api", "restful apis"},
    "rest apis": {"rest api", "restful api", "restful apis"},
    "vector databases": {"vector database"},
    "vector database": {"vector databases"},
    "fine tuning": {"fine-tuning", "finetuning"},
    "human in the loop": {"human-in-the-loop", "hitl"},
    "node": {"nodejs", "node.js"},
    "nodejs": {"node", "node.js"},
}

CANONICAL_MAP: Dict[str, str] = {
    "rag": "retrieval augmented generation",
    "llm": "large language models",
    "llms": "large language models",
    "large language model": "large language models",
    "nlp": "natural language processing",
    "ml": "machine learning",
    "genai": "generative ai",
    "gen ai": "generative ai",
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "rest api": "rest apis",
    "restful api": "rest apis",
    "restful apis": "rest apis",
    "vector database": "vector databases",
    "node": "nodejs",
    "node.js": "nodejs",
    "ci": "continuous integration",
    "cd": "continuous deployment",
}

STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "for", "from", "had", "has", "have", "he", "her", "his", "if", "in", "into",
    "is", "it", "its", "of", "on", "or", "our", "out", "over", "per", "she",
    "should", "so", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "to", "under", "up",
    "us", "use", "using", "was", "we", "were", "what", "when", "where", "which",
    "who", "will", "with", "within", "without", "would", "you", "your",
    # Common generic job-post filler words
    "ability", "about", "across", "all", "also", "apply", "background", "based",
    "benefits", "best", "build", "building", "candidate", "collaborate",
    "communication", "company", "create", "creating", "day", "degree", "deliver",
    "demonstrated", "description", "develop", "developing", "development", "drive",
    "effective", "end", "engineer", "engineering", "ensure", "environment",
    "equal", "etc", "excellent", "experience", "experienced", "familiarity",
    "fast", "focus", "good", "great", "help", "high", "highly", "ideal",
    "including", "industry", "join", "key", "knowledge", "lead", "level",
    "looking", "make", "manage", "minimum", "mission", "modern", "multiple",
    "must", "need", "new", "nice", "opportunity", "organization", "other",
    "paced", "passion", "plus", "position", "preferred", "problem", "problems",
    "product", "products", "proven", "qualifications", "quality", "related",
    "relevant", "require", "required", "requirements", "responsibilities",
    "responsible", "role", "scale", "seek", "seeking", "senior", "skills",
    "solid", "solutions", "solving", "stakeholders", "strong", "support",
    "team", "teams", "technical", "technology", "technologies", "tools",
    "understanding", "well", "work", "working", "written", "year", "years",
}


@dataclass
class ATSAnalysis:
    """Result of comparing job description keywords against a candidate resume."""

    match_score: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    jd_keywords: List[str]
    resume_highlights: List[str]
    candidate_name: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def normalize_text(text: str) -> str:
    """Lowercase text and normalize dashes/whitespace while keeping tech tokens."""
    cleaned = text.lower()
    cleaned = cleaned.replace("—", " ").replace("–", " ")
    # Normalize hyphenated phrases like "retrieval-augmented" -> "retrieval augmented"
    cleaned = re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _contains_term(normalized_text: str, term: str) -> bool:
    """Check whether `term` or any known alias appears as a whole phrase/token."""
    candidates = {term} | SKILL_ALIASES.get(term, set())
    for candidate in candidates:
        cand_norm = normalize_text(candidate)
        if not cand_norm:
            continue
        pattern = r"(?<![a-z0-9_+#.])" + re.escape(cand_norm) + r"(?![a-z0-9_+#.])"
        if re.search(pattern, normalized_text):
            return True
    return False


def _canonicalize(term: str) -> str:
    norm = normalize_text(term)
    return CANONICAL_MAP.get(norm, norm)


def _add_unique_keyword(found: List[str], seen_canonical: Set[str], term: str) -> bool:
    canon = _canonicalize(term)
    if canon in seen_canonical:
        return False
    seen_canonical.add(canon)
    found.append(canon)
    return True


def extract_keywords(text: str, max_keywords: int = 25) -> List[str]:
    """Extract multi-word skills and meaningful technical keywords from text."""
    if not text or not text.strip():
        return []

    norm = normalize_text(text)
    found: List[str] = []
    seen_canonical: Set[str] = set()
    consumed_tokens: Set[str] = set()

    # 1. Match curated multi-word skills first (longest phrases first).
    for phrase in sorted(MULTI_WORD_SKILLS, key=lambda s: len(s.split()), reverse=True):
        if _contains_term(norm, phrase):
            if _add_unique_keyword(found, seen_canonical, phrase):
                consumed_tokens.update(phrase.split())

    # 2. Match known single-word technical skills.
    tokens = re.findall(r"[a-z0-9][a-z0-9_+#./-]*", norm)
    for token in tokens:
        clean_tok = token.strip(".,;:()[]{}\"'")
        if not clean_tok or len(clean_tok) < 2:
            continue
        if clean_tok in KNOWN_SINGLE_SKILLS and clean_tok not in consumed_tokens:
            _add_unique_keyword(found, seen_canonical, clean_tok)

    # 3. Extract remaining high-signal domain terms (frequency-ranked) if room remains.
    if len(found) < max_keywords:
        counts: Dict[str, int] = {}
        for token in tokens:
            clean_tok = token.strip(".,;:()[]{}\"'/-")
            if (
                len(clean_tok) >= 4
                and clean_tok not in STOPWORDS
                and clean_tok not in consumed_tokens
                and not clean_tok.isdigit()
            ):
                counts[clean_tok] = counts.get(clean_tok, 0) + 1

        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for term, freq in ranked:
            if len(found) >= max_keywords:
                break
            if freq >= 2 or len(found) < 10:
                _add_unique_keyword(found, seen_canonical, term)

    return found[:max_keywords]


def extract_resume_highlights(resume_text: str, max_items: int = 6) -> List[str]:
    """Extract concrete achievement/bullet lines from the resume for grounding."""
    if not resume_text or not resume_text.strip():
        return []

    lines = [line.strip(" \t•-*") for line in resume_text.splitlines()]
    candidates: List[Tuple[int, str]] = []
    for line in lines:
        if len(line) < 20 or len(line) > 260:
            continue
        score = 0
        if re.search(r"\d+%|\$\d+|\d+\+|\d+x|\d+\s*(ms|s|users|requests|projects)", line, re.I):
            score += 3
        if re.search(
            r"\b(built|designed|implemented|developed|led|reduced|improved|launched|deployed|architected|automated)\b",
            line,
            re.I,
        ):
            score += 2
        if score > 0:
            candidates.append((score, line))

    candidates.sort(key=lambda item: -item[0])
    seen: Set[str] = set()
    highlights: List[str] = []
    for _, line in candidates:
        if line not in seen:
            seen.add(line)
            highlights.append(line)
        if len(highlights) >= max_items:
            break
    return highlights


def extract_candidate_name(resume_text: str) -> str:
    """Best-effort extraction of candidate name from the top of the resume."""
    for raw_line in resume_text.splitlines():
        line = raw_line.strip().strip("#").strip()
        if not line:
            continue
        if "@" in line or "http" in line.lower() or "|" in line:
            # Handle lines like "Jane Doe | jane@example.com"
            if "|" in line:
                first_part = line.split("|", 1)[0].strip()
                if 2 <= len(first_part.split()) <= 4 and first_part.replace(" ", "").isalpha():
                    return first_part
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(re.match(r"^[A-Za-z][A-Za-z'.-]*$", w) for w in words):
            lowered = line.lower()
            if lowered not in {"curriculum vitae", "resume", "professional summary", "work experience"}:
                return line
        break
    return "Candidate"


def analyze_ats_alignment(resume_text: str, job_description: str, max_keywords: int = 25) -> ATSAnalysis:
    """Compare JD keywords against resume text and compute ATS match score and gaps."""
    jd_keywords = extract_keywords(job_description, max_keywords=max_keywords)
    norm_resume = normalize_text(resume_text)

    matched: List[str] = []
    missing: List[str] = []
    for kw in jd_keywords:
        if _contains_term(norm_resume, kw):
            matched.append(kw)
        else:
            missing.append(kw)

    score = round((len(matched) / len(jd_keywords)) * 100.0, 1) if jd_keywords else 0.0
    highlights = extract_resume_highlights(resume_text)
    candidate_name = extract_candidate_name(resume_text)

    return ATSAnalysis(
        match_score=score,
        matched_keywords=matched,
        missing_keywords=missing,
        jd_keywords=jd_keywords,
        resume_highlights=highlights,
        candidate_name=candidate_name,
    )
