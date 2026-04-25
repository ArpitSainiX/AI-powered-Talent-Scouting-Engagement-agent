from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from talent_agent.engine import load_candidates, parse_jd, rank_candidates_with_parsed_jd
from talent_agent.llm import DEFAULT_MODEL, enrich_candidate_with_ai, get_api_key, has_openai, parse_jd_with_ai


ROOT = Path(__file__).parent
SAMPLE_JD = (ROOT / "data" / "sample_jd.txt").read_text(encoding="utf-8")
CANDIDATES = load_candidates(ROOT / "data" / "candidates.json")


def read_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None


def score_class(score: float) -> str:
    if score >= 75:
        return "high"
    if score >= 55:
        return "mid"
    return "low"


def score_bar(label: str, score: float) -> None:
    level = score_class(score)
    st.markdown(
        f"""
        <div class="score-wrap">
            <div class="score-row"><span>{label}</span><strong>{score:.1f}</strong></div>
            <div class="score-track"><div class="score-fill {level}" style="width:{max(3, min(score, 100))}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def candidate_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Rank": index + 1,
                "Candidate": row["name"],
                "Current Title": row["title"],
                "Location": row["location"],
                "Match": row["match_score"],
                "Interest": row["interest_score"],
                "Final": row["final_score"],
                "Availability": row["availability"],
                "Recommendation": row.get("next_action", "Follow up" if row["final_score"] >= 70 else "Keep warm"),
            }
            for index, row in enumerate(rows)
        ]
    )


st.set_page_config(page_title="ScoutFlow AI", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #17201b;
        --muted: #657169;
        --paper: #f7f8f3;
        --panel: #ffffff;
        --line: #e1e5da;
        --green: #0e7c66;
        --teal: #37a78f;
        --amber: #c68121;
        --rose: #c84b5c;
        --plum: #6e4f85;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    .block-container { padding-top: 1.2rem; max-width: 1320px; }
    h1, h2, h3 { letter-spacing: 0; }
    .hero {
        border: 1px solid var(--line);
        background:
            linear-gradient(135deg, rgba(14,124,102,.12), rgba(198,129,33,.10)),
            #ffffff;
        border-radius: 8px;
        padding: 26px 28px;
        margin-bottom: 18px;
    }
    .hero h1 { font-size: 2.15rem; margin: 0 0 8px; }
    .hero p { color: var(--muted); font-size: 1.02rem; margin: 0; max-width: 780px; }
    .mini-label {
        color: var(--green);
        font-weight: 800;
        text-transform: uppercase;
        font-size: .72rem;
        letter-spacing: .08rem;
        margin-bottom: 8px;
    }
    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px;
        min-height: 100%;
    }
    .metric-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px;
        background: #fff;
    }
    .metric-card span { color: var(--muted); font-size: .82rem; }
    .metric-card strong { display: block; font-size: 1.45rem; margin-top: 4px; }
    .chip {
        display: inline-block;
        border: 1px solid #cfd7cc;
        border-radius: 999px;
        padding: 4px 10px;
        margin: 3px 5px 3px 0;
        background: #fbfcf8;
        color: #26322c;
        font-size: .82rem;
    }
    .score-wrap { margin: 10px 0 14px; }
    .score-row { display: flex; justify-content: space-between; font-size: .88rem; margin-bottom: 6px; }
    .score-track { height: 9px; border-radius: 999px; background: #e7ebe2; overflow: hidden; }
    .score-fill { height: 100%; border-radius: 999px; }
    .score-fill.high { background: linear-gradient(90deg, var(--green), var(--teal)); }
    .score-fill.mid { background: linear-gradient(90deg, var(--amber), #e4b358); }
    .score-fill.low { background: linear-gradient(90deg, var(--rose), var(--plum)); }
    .candidate-head {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: flex-start;
        border-bottom: 1px solid var(--line);
        padding-bottom: 12px;
        margin-bottom: 12px;
    }
    .candidate-head h3 { margin: 0; font-size: 1.05rem; }
    .candidate-head p { margin: 4px 0 0; color: var(--muted); }
    .rank-pill {
        background: #17201b;
        color: white;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: .78rem;
        white-space: nowrap;
    }
    div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 8px; background: white; }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--green);
        background: var(--green);
        color: white;
        font-weight: 700;
    }
    .stTextArea textarea { border-radius: 8px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="mini-label">Recruiting Agent Prototype</div>
      <h1>ScoutFlow AI</h1>
      <p>Convert a job description into an explainable, engaged candidate shortlist with match scoring, simulated outreach, and recruiter-ready action outputs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

secret_key = read_secret("OPENAI_API_KEY")

with st.sidebar:
    st.markdown("### Control Room")
    use_ai = st.toggle("Use OpenAI enrichment", value=bool(secret_key), help="Fallback scoring works without an API key.")
    model = st.text_input("OpenAI model", value=read_secret("OPENAI_MODEL") or DEFAULT_MODEL)
    typed_key = st.text_input("OpenAI API key", type="password", help="Used only for this session. Do not commit keys to GitHub.")
    api_key = get_api_key(typed_key or secret_key)
    shortlist_size = st.slider("Shortlist size", 3, len(CANDIDATES), 5)
    enrich_count = st.slider("AI-enrich top candidates", 1, 5, min(3, shortlist_size), disabled=not use_ai)
    st.divider()
    st.markdown("### Scoring Formula")
    st.caption("Final Score = 65% Match Score + 35% Interest Score")
    st.caption("OpenAI improves parsing, outreach, simulated replies, and explanations. Core ranking still has a deterministic fallback.")

input_col, parsed_col = st.columns([1.15, 0.85], gap="large")

with input_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Job Description")
    jd_text = st.text_area(
        "Paste a JD",
        value=SAMPLE_JD,
        height=330,
        label_visibility="collapsed",
    )
    run = st.button("Run Talent Scout", type="primary", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with parsed_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Agent Status")
    status_cols = st.columns(3)
    status_cols[0].markdown(f'<div class="metric-card"><span>Candidates</span><strong>{len(CANDIDATES)}</strong></div>', unsafe_allow_html=True)
    status_cols[1].markdown(f'<div class="metric-card"><span>AI Mode</span><strong>{"On" if use_ai and api_key else "Off"}</strong></div>', unsafe_allow_html=True)
    status_cols[2].markdown(f'<div class="metric-card"><span>Shortlist</span><strong>{shortlist_size}</strong></div>', unsafe_allow_html=True)

    if use_ai and not api_key:
        st.warning("Add an API key in the sidebar or Streamlit secrets to enable AI enrichment.")
    if use_ai and api_key and not has_openai():
        st.error("The OpenAI package is not installed. Run `pip install -r requirements.txt`.")
    st.markdown("</div>", unsafe_allow_html=True)

if run or "ranked" not in st.session_state:
    parsed_jd = parse_jd(jd_text)
    ai_enabled = use_ai and bool(api_key) and has_openai()

    if ai_enabled:
        with st.spinner("Parsing JD with OpenAI..."):
            try:
                ai_parsed = parse_jd_with_ai(jd_text, api_key, model)
                if ai_parsed:
                    parsed_jd = {**parsed_jd, **ai_parsed}
            except Exception as exc:
                st.warning(f"AI JD parsing failed, using fallback parser. Details: {exc}")

    ranked = rank_candidates_with_parsed_jd(jd_text, parsed_jd, CANDIDATES)
    shortlist = ranked[:shortlist_size]

    if ai_enabled:
        with st.spinner("Generating personalized outreach and AI explanations..."):
            enriched = []
            for index, candidate in enumerate(shortlist):
                updated = dict(candidate)
                if index < enrich_count:
                    try:
                        ai_candidate = enrich_candidate_with_ai(jd_text, parsed_jd, candidate, api_key, model)
                        if ai_candidate:
                            updated.update(ai_candidate)
                            updated["interest_score"] = int(max(0, min(100, updated["interest_score"])))
                            updated["final_score"] = round(
                                (0.65 * updated["match_score"]) + (0.35 * updated["interest_score"]),
                                1,
                            )
                    except Exception as exc:
                        updated["ai_explanation"] = [f"AI enrichment failed for this candidate: {exc}"]
                enriched.append(updated)
            enriched.sort(key=lambda item: item["final_score"], reverse=True)
            shortlist = enriched

    st.session_state.parsed_jd = parsed_jd
    st.session_state.ranked = shortlist

parsed_jd = st.session_state.parsed_jd
shortlist = st.session_state.ranked

st.divider()
st.markdown("### Parsed Requirements")
req_cols = st.columns(4)
req_cols[0].markdown(f'<div class="metric-card"><span>Role</span><strong>{parsed_jd.get("title", "Role")}</strong></div>', unsafe_allow_html=True)
req_cols[1].markdown(f'<div class="metric-card"><span>Seniority</span><strong>{parsed_jd.get("seniority", "Not set")}</strong></div>', unsafe_allow_html=True)
req_cols[2].markdown(f'<div class="metric-card"><span>Experience</span><strong>{parsed_jd.get("min_experience", 0) or "N/A"} yrs</strong></div>', unsafe_allow_html=True)
req_cols[3].markdown(f'<div class="metric-card"><span>Work Mode</span><strong>{parsed_jd.get("work_mode", "Flexible")}</strong></div>', unsafe_allow_html=True)

skill_html = "".join(f'<span class="chip">{skill}</span>' for skill in parsed_jd.get("required_skills", []))
domain_html = "".join(f'<span class="chip">{domain}</span>' for domain in parsed_jd.get("domain_keywords", []))
must_html = "".join(f'<span class="chip">{item}</span>' for item in parsed_jd.get("must_haves", [])[:8])

chips_left, chips_right = st.columns(2)
with chips_left:
    st.markdown("**Required Skills**")
    st.markdown(skill_html or "No explicit skills detected", unsafe_allow_html=True)
with chips_right:
    st.markdown("**Domain and Must-Have Signals**")
    st.markdown((domain_html + must_html) or "No explicit domain signals detected", unsafe_allow_html=True)

st.divider()
st.markdown("### Ranked Shortlist")
table = candidate_table(shortlist)
st.dataframe(table, width="stretch", hide_index=True)

download_left, download_right = st.columns([1, 1])
download_left.download_button(
    "Download CSV",
    table.to_csv(index=False).encode("utf-8"),
    "scoutflow_shortlist.csv",
    "text/csv",
    width="stretch",
)
download_right.download_button(
    "Download JSON",
    json.dumps(shortlist, indent=2).encode("utf-8"),
    "scoutflow_shortlist.json",
    "application/json",
    width="stretch",
)

top = shortlist[0]
st.markdown("### Recruiter Recommendation")
rec_cols = st.columns([0.95, 1.05])
with rec_cols[0]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f"#### Prioritize {top['name']}")
    st.write(top.get("next_action", "Move this candidate to recruiter follow-up."))
    score_bar("Final Score", float(top["final_score"]))
    score_bar("Match Score", float(top["match_score"]))
    score_bar("Interest Score", float(top["interest_score"]))
    st.markdown("</div>", unsafe_allow_html=True)
with rec_cols[1]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Why this shortlist is actionable")
    st.write("The ranking separates role fit from candidate intent, so recruiters can avoid wasting time on profiles that look strong but show weak engagement.")
    st.write("Each profile includes a matching rationale, an outreach message, a simulated response, and the next best action.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Candidate Intelligence")
for index, candidate in enumerate(shortlist, start=1):
    with st.expander(f"#{index} {candidate['name']} - {candidate['title']} | Final {candidate['final_score']}", expanded=index == 1):
        st.markdown(
            f"""
            <div class="candidate-head">
              <div>
                <h3>{candidate['name']}</h3>
                <p>{candidate['title']} • {candidate['location']} • {candidate['experience_years']} years</p>
              </div>
              <div class="rank-pill">Rank #{index}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        profile_col, score_col, outreach_col = st.columns([1, 0.8, 1.2], gap="large")
        with profile_col:
            st.write("**Profile**")
            st.write(candidate["summary"])
            st.write("**Skills**")
            st.markdown("".join(f'<span class="chip">{skill}</span>' for skill in candidate["skills"]), unsafe_allow_html=True)
            st.write("**Availability**")
            st.write(candidate["availability"])
        with score_col:
            score_bar("Match", float(candidate["match_score"]))
            score_bar("Interest", float(candidate["interest_score"]))
            score_bar("Final", float(candidate["final_score"]))
            st.write("**Matched Skills**")
            st.write(", ".join(candidate["matched_skills"]) or "Limited overlap")
        with outreach_col:
            st.write("**Explainability**")
            explanations = candidate.get("ai_explanation") or candidate["match_explanation"]
            for reason in explanations:
                st.write(f"- {reason}")
            risks = candidate.get("risk_flags") or []
            if risks:
                st.write("**Risk Flags**")
                for risk in risks:
                    st.write(f"- {risk}")
            st.write("**Outreach Message**")
            st.info(candidate["outreach_message"])
            st.write("**Simulated Candidate Reply**")
            st.write(candidate["candidate_reply"])

st.markdown("### Recruiter API Output")
st.json(
    [
        {
            "candidate_id": candidate["id"],
            "name": candidate["name"],
            "match_score": candidate["match_score"],
            "interest_score": candidate["interest_score"],
            "final_score": candidate["final_score"],
            "matched_skills": candidate["matched_skills"],
            "recommendation": candidate.get("next_action", "Follow up" if candidate["final_score"] >= 70 else "Keep warm"),
        }
        for candidate in shortlist
    ]
)
