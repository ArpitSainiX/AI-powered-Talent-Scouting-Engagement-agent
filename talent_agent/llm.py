from __future__ import annotations

import json
import os
from typing import Any


DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_MODEL = DEFAULT_OPENAI_MODEL


def has_openai() -> bool:
    try:
        import openai  # noqa: F401
    except Exception:
        return False
    return True


def has_gemini() -> bool:
    try:
        import google.genai  # noqa: F401
    except Exception:
        return False
    return True


def get_api_key(explicit_key: str | None = None, env_name: str = "OPENAI_API_KEY") -> str | None:
    if explicit_key:
        return explicit_key.strip()
    return os.getenv(env_name)


def _extract_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(value)
    return "\n".join(parts).strip()


def _json_from_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    return json.loads(cleaned)


def call_openai_json(prompt: str, api_key: str, model: str = DEFAULT_OPENAI_MODEL) -> dict | None:
    if not api_key or not has_openai():
        return None

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a precise recruiting automation engine. "
                    "Return only valid compact JSON. Do not include markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
    )
    return _json_from_text(_extract_text(response))


def call_gemini_json(prompt: str, api_key: str, model: str = DEFAULT_GEMINI_MODEL) -> dict | None:
    if not api_key or not has_gemini():
        return None

    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=(
            "You are a precise recruiting automation engine. "
            "Return only valid compact JSON. Do not include markdown.\n\n"
            f"{prompt}"
        ),
        config={
            "temperature": 0.25,
            "response_mime_type": "application/json",
        },
    )
    return _json_from_text(response.text or "")


def call_json(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_OPENAI_MODEL,
    provider: str = "OpenAI",
) -> dict | None:
    if provider == "Gemini":
        return call_gemini_json(prompt, api_key, model)
    return call_openai_json(prompt, api_key, model)


def parse_jd_with_ai(
    jd_text: str,
    api_key: str,
    model: str = DEFAULT_OPENAI_MODEL,
    provider: str = "OpenAI",
) -> dict | None:
    prompt = f"""
Extract structured recruiting requirements from this job description.

Return JSON with exactly these keys:
title: string
required_skills: array of lowercase strings
domain_keywords: array of lowercase strings
min_experience: integer
location: string
work_mode: one of Remote, Hybrid, Onsite, Flexible
seniority: string
must_haves: array of strings
nice_to_haves: array of strings

Job description:
{jd_text}
"""
    return call_json(prompt, api_key, model, provider)


def enrich_candidate_with_ai(
    jd_text: str,
    parsed_jd: dict,
    candidate: dict,
    api_key: str,
    model: str = DEFAULT_OPENAI_MODEL,
    provider: str = "OpenAI",
) -> dict | None:
    prompt = f"""
Act as a recruiter AI agent. Use the JD and candidate profile to produce outreach,
candidate reply simulation, interest score, and explainability.

Return JSON with exactly these keys:
outreach_message: string
candidate_reply: string
interest_score: integer from 0 to 100
ai_explanation: array of 3 short strings
next_action: string
risk_flags: array of short strings

Job description:
{jd_text}

Parsed JD:
{json.dumps(parsed_jd, ensure_ascii=True)}

Candidate:
{json.dumps(candidate, ensure_ascii=True)}
"""
    return call_json(prompt, api_key, model, provider)
