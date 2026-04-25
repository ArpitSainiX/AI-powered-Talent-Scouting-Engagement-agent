import json
import math
import re
from collections import Counter
from pathlib import Path


SKILL_KEYWORDS = [
    "python",
    "typescript",
    "javascript",
    "react",
    "next.js",
    "streamlit",
    "fastapi",
    "flask",
    "django",
    "openai api",
    "gemini",
    "langchain",
    "langgraph",
    "rag",
    "llms",
    "nlp",
    "vector db",
    "chroma",
    "pinecone",
    "supabase",
    "pgvector",
    "postgresql",
    "sql",
    "docker",
    "kubernetes",
    "aws",
    "mlops",
    "pytorch",
    "tensorflow",
    "tailwind css",
    "prompt engineering",
    "conversational ai",
    "ai agents",
]

DOMAIN_KEYWORDS = [
    "hr tech",
    "recruitment",
    "recruiting",
    "talent",
    "workflow automation",
    "ai assistants",
    "conversational ai",
    "search",
    "recommendation",
    "saas",
    "fintech",
    "healthcare",
]

SKILL_ALIASES = {
    "vector db": {"vector db", "chroma", "pinecone", "supabase", "pgvector", "postgresql"},
    "llms": {"llms", "openai api", "gemini", "rag", "prompt engineering"},
    "ai agents": {"ai agents", "langchain", "langgraph", "openai api", "conversational ai"},
    "postgresql": {"postgresql", "sql", "supabase"},
    "react": {"react", "next.js"},
    "next.js": {"next.js", "react"},
    "openai api": {"openai api", "llms", "rag", "prompt engineering"},
    "langchain": {"langchain", "langgraph", "rag"},
    "langgraph": {"langgraph", "langchain", "ai agents"},
    "rag": {"rag", "vector db", "chroma", "pinecone", "supabase", "pgvector"},
}


def load_candidates(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_jd(jd_text: str) -> dict:
    text = jd_text.lower()
    skills = [skill for skill in SKILL_KEYWORDS if skill in text]
    domains = [domain for domain in DOMAIN_KEYWORDS if domain in text]
    years = [int(match) for match in re.findall(r"(\d+)\+?\s*(?:years|yrs)", text)]
    min_experience = max(years) if years else 0

    work_mode = "Remote" if "remote" in text else "Hybrid" if "hybrid" in text else "Flexible"
    location_match = re.search(r"location\s*:\s*([^\n]+)", jd_text, flags=re.IGNORECASE)
    location = location_match.group(1).strip() if location_match else "Not specified"

    title = "AI Talent Role"
    title_patterns = [
        r"hiring\s+(?:a|an)?\s*([^\n.]+)",
        r"role\s*:\s*([^\n]+)",
        r"job title\s*:\s*([^\n]+)",
    ]
    for pattern in title_patterns:
        match = re.search(pattern, jd_text, flags=re.IGNORECASE)
        if match:
            title = match.group(1).strip(" .:-")
            break

    return {
        "title": title,
        "required_skills": sorted(set(skills)),
        "domain_keywords": sorted(set(domains)),
        "min_experience": min_experience,
        "location": location,
        "work_mode": work_mode,
    }


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9.+#-]*", text.lower())


def cosine_similarity(left: str, right: str) -> float:
    left_counts = Counter(tokenize(left))
    right_counts = Counter(tokenize(right))
    if not left_counts or not right_counts:
        return 0.0

    shared = set(left_counts) & set(right_counts)
    dot = sum(left_counts[token] * right_counts[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    return dot / (left_norm * right_norm)


def candidate_text(candidate: dict) -> str:
    return " ".join(
        [
            candidate["title"],
            candidate["summary"],
            " ".join(candidate["skills"]),
            " ".join(candidate["domains"]),
            candidate["location"],
            candidate["preferred_work_mode"],
        ]
    )


def score_match(jd_text: str, parsed_jd: dict, candidate: dict) -> dict:
    required = set(parsed_jd["required_skills"])
    candidate_skills = {skill.lower() for skill in candidate["skills"]}
    candidate_profile_text = candidate_text(candidate).lower()
    matched_skills = []
    for skill in required:
        aliases = SKILL_ALIASES.get(skill, {skill})
        if skill in candidate_skills or any(alias in candidate_skills or alias in candidate_profile_text for alias in aliases):
            matched_skills.append(skill)
    matched_skills = sorted(set(matched_skills))

    skill_score = (len(matched_skills) / len(required) * 50) if required else 25
    semantic_score = cosine_similarity(jd_text, candidate_text(candidate)) * 10

    min_exp = parsed_jd["min_experience"]
    if min_exp == 0:
        experience_score = 15
    else:
        ratio = min(candidate["experience_years"] / min_exp, 1.25)
        experience_score = min(ratio / 1.25 * 15, 15)

    domain_matches = sorted(
        set(parsed_jd["domain_keywords"])
        & {domain.lower() for domain in candidate["domains"]}
    )
    domain_score = min(len(domain_matches) * 5, 10)

    location_score = 7 if "remote" in parsed_jd["work_mode"].lower() and candidate["preferred_work_mode"] == "Remote" else 4
    availability_score = 8 if "immediately" in candidate["availability"].lower() or "2 weeks" in candidate["availability"].lower() else 4

    total = skill_score + semantic_score + experience_score + domain_score + location_score + availability_score
    total = max(0, min(round(total, 1), 100))

    explanation = [
        f"Matched skills: {', '.join(matched_skills) if matched_skills else 'limited direct skill overlap'}",
        f"{candidate['experience_years']} years experience vs JD minimum of {min_exp or 'unspecified'}",
        f"Relevant domains: {', '.join(domain_matches) if domain_matches else 'no strong domain match'}",
        f"Availability: {candidate['availability']}",
    ]

    return {
        "match_score": total,
        "matched_skills": matched_skills,
        "domain_matches": domain_matches,
        "match_explanation": explanation,
    }


def simulate_outreach(parsed_jd: dict, candidate: dict) -> dict:
    intro = (
        f"Hi {candidate['name'].split()[0]}, I found your background in "
        f"{', '.join(candidate['skills'][:3])} relevant for our {parsed_jd['title']} role. "
        "Would you be open to a quick conversation?"
    )

    style = candidate["response_style"]
    if style == "enthusiastic":
        reply = (
            "Thanks for reaching out. This sounds very aligned with the AI product work I want to do next. "
            f"My availability is {candidate['availability'].lower()}, and the compensation range looks worth discussing."
        )
        interest = 88
    elif style == "curious":
        reply = (
            "This looks interesting. I would like to learn more about the team, product roadmap, and interview process "
            "before deciding, but I am open to a first call."
        )
        interest = 74
    elif style == "balanced":
        reply = (
            "The role seems relevant, especially the NLP part. I am open to exploring it, though timing and scope will matter."
        )
        interest = 64
    elif style == "busy":
        reply = (
            "Thanks. I am not looking for a full-time switch right now, but I may consider consulting or a very strong fit."
        )
        interest = 48
    else:
        reply = (
            "Appreciate the message. I am not actively searching, but I can review details if the role is senior enough "
            "and the package is competitive."
        )
        interest = 42

    if parsed_jd["work_mode"] == candidate["preferred_work_mode"]:
        interest += 5
    if "immediately" in candidate["availability"].lower():
        interest += 5

    return {
        "outreach_message": intro,
        "candidate_reply": reply,
        "interest_score": min(100, interest),
    }


def rank_candidates(jd_text: str, candidates: list[dict]) -> list[dict]:
    parsed = parse_jd(jd_text)
    ranked = []
    for candidate in candidates:
        match = score_match(jd_text, parsed, candidate)
        outreach = simulate_outreach(parsed, candidate)
        final_score = round((0.65 * match["match_score"]) + (0.35 * outreach["interest_score"]), 1)
        ranked.append({**candidate, **match, **outreach, "final_score": final_score})

    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    return ranked


def rank_candidates_with_parsed_jd(jd_text: str, parsed_jd: dict, candidates: list[dict]) -> list[dict]:
    ranked = []
    for candidate in candidates:
        match = score_match(jd_text, parsed_jd, candidate)
        outreach = simulate_outreach(parsed_jd, candidate)
        final_score = round((0.65 * match["match_score"]) + (0.35 * outreach["interest_score"]), 1)
        ranked.append({**candidate, **match, **outreach, "final_score": final_score})

    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    return ranked
