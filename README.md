# ScoutFlow AI

AI-powered talent scouting and engagement agent for recruiters.

ScoutFlow AI takes a job description, extracts role requirements, ranks candidates, simulates outreach conversations, and outputs a recruiter-ready shortlist scored on two dimensions:

- **Match Score**: how well the candidate fits the JD
- **Interest Score**: how likely the candidate is to engage

The app works without an API key using deterministic fallback logic, and becomes more impressive with Gemini or OpenAI enrichment enabled.

## Live Demo Flow

1. Paste a job description.
2. Click `Run Talent Scout`.
3. Review parsed requirements.
4. Inspect the ranked shortlist.
5. Open candidate intelligence panels.
6. Export CSV or JSON for recruiter action.

## Tech Stack

- Python
- Streamlit frontend
- Modular backend logic in `talent_agent/`
- Gemini or OpenAI API for optional AI parsing, outreach, reply simulation, and explanations
- Pandas for shortlist exports
- JSON seed dataset for safe candidate discovery simulation

## Project Structure

```text
.
├── app.py
├── data/
│   ├── candidates.json
│   └── sample_jd.txt
├── docs/
│   ├── architecture.md
│   └── demo-script.md
├── talent_agent/
│   ├── engine.py
│   └── llm.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## AI Provider Setup

Never commit your real API key.

For local Gemini development, set environment variables:

```bash
set GEMINI_API_KEY=your_key_here
set GEMINI_MODEL=gemini-2.5-flash
streamlit run app.py
```

For local OpenAI development, set environment variables:

```bash
set OPENAI_API_KEY=your_key_here
set OPENAI_MODEL=gpt-4.1-mini
streamlit run app.py
```

Or paste either key into the sidebar during a local session. The app does not store sidebar keys in the repository.

For Streamlit Community Cloud, add this in app secrets:

```toml
GEMINI_API_KEY = "your_key_here"
GEMINI_MODEL = "gemini-2.5-flash"

# Optional fallback
# OPENAI_API_KEY = "your_key_here"
# OPENAI_MODEL = "gpt-4.1-mini"
```

## GitHub Deployment Checklist

Before pushing:

- Confirm `.env` and `.streamlit/secrets.toml` are not committed.
- Confirm API keys are not present in any file.
- Push the repo to GitHub.
- Deploy on Streamlit Community Cloud.
- Add `GEMINI_API_KEY` or `OPENAI_API_KEY` in Streamlit secrets.

## Scoring

```text
Final Score = 0.65 * Match Score + 0.35 * Interest Score
```

Match Score considers:

- Required skill overlap
- Text similarity between JD and profile
- Experience fit
- Domain relevance
- Location/work-mode compatibility
- Availability

Interest Score considers:

- Candidate openness
- Availability
- Work-mode alignment
- Simulated reply quality

## Sample Output

```json
[
  {
    "candidate_id": "C005",
    "name": "Rohan Gupta",
    "match_score": 64.7,
    "interest_score": 93,
    "final_score": 74.6,
    "matched_skills": ["ai agents", "chroma", "fastapi", "langchain", "next.js", "openai api", "postgresql", "python", "rag", "react", "sql"],
    "recommendation": "Follow up"
  }
]
```

Scores can change when AI enrichment is enabled.

## Architecture

See [docs/architecture.md](docs/architecture.md).


## Future Upgrades

- GitHub public profile discovery
- Supabase pgvector or Chroma semantic search
- ATS integration
- Real email/LinkedIn outreach with approval gates
- Recruiter feedback loop for score calibration
