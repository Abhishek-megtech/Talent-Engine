# Redrob Talent Intelligence — India Runs Data & AI Challenge

Hybrid candidate ranking pipeline for the Redrob "Senior AI Engineer — Founding Team" JD.
Ranks 100,000 candidates in **~70 seconds** on a single CPU core. No network calls, no model downloads, no API keys.

## Quick start

```bash
pip install -r requirements.txt

# Produce submission CSV
python rank.py \
  --candidates path/to/candidates.jsonl \
  --out submission.csv

# Validate against challenge rules
python validate_submission.py submission.csv

# Run tests
python -m pytest tests/ -v
```

## Architecture

```
candidates.jsonl
      │
      ▼ (streaming, one pass)
┌─────────────────────────────┐
│  Structural Feature Scorer  │  features.py
│  ─────────────────────────  │
│  • Skill Match Trust (25%)  │
│  • Title / Seniority  (15%) │
│  • Career Quality     (20%) │
│  • Experience Fit     (10%) │
│  • Location Fit       (10%) │
│  • Behavioral Mult    (×)   │
│  • Honeypot Risk      (×)   │
└─────────────────────────────┘
      │ partial scores + notes
      ▼
┌─────────────────────────────┐
│  TF-IDF Semantic Fit (20%)  │  sklearn TfidfVectorizer
│  vectorized cosine sim      │  corpus = all candidates + JD
└─────────────────────────────┘
      │ sem_sim per candidate
      ▼
┌─────────────────────────────┐
│  Finalize + Sort + Tie-     │  scoring.py + rank.py
│  break (id ascending)       │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  Reasoning Generator        │  reasoning.py
│  polarity-tagged notes →    │
│  varied, grounded sentences │
└─────────────────────────────┘
      │
      ▼
  submission.csv
```

## Scoring formula

```
base = 0.20×semantic_fit + 0.25×skill_trust + 0.15×title_fit
     + 0.20×career_quality + 0.10×exp_fit + 0.10×location_fit

final = base × behavioral_multiplier × (1 - honeypot_risk)
```

Behavioral multiplier and honeypot risk are **multiplicative** (not additive) because an unreachable or fabricated candidate is not "70% as good" — they are not a usable recommendation regardless of how well their profile matches on paper.

## Design decisions (Stage 5 defence)

**Why TF-IDF not sentence-transformers?**
The ranking step must run CPU-only, ≤5 min, ≤16GB RAM, no network. Dense embedding models require a download, consume significant memory, and take minutes even vectorized. TF-IDF is sub-30s for 100K documents, fully deterministic, needs only scikit-learn. The majority of ranking signal in this dataset is *structured* (skills, career history, behavioral signals) — the JD itself says "look at the full picture", not "run semantic similarity". Semantic fit is the 20% term, not the whole score.

**Why multiplicative for behavioral and honeypot?**
A candidate who is inactive for 8 months and never responds to recruiters is, for practical hiring purposes, not available — regardless of how perfectly their profile matches. A 0.6× multiplier on top of a high structural score correctly reflects this. The same logic applies to honeypots: a fabricated profile should not appear in the top 100, period, not be "weighted down by 20%".

**Why polarity-tagged notes in features.py?**
Early versions had reasoning.py infer polarity from whether a subscore crossed a threshold. This produced contradictions like "strong on core skills... weak/no evidence of core skills" in the same sentence. Tagging polarity at the note source makes the reasoning always self-consistent.

**What was not built and why**
- No React/FastAPI dashboard (not scored, high dev cost)
- No FAISS index (no query-time vector search; TF-IDF is computed once over the full corpus, which is different from a retrieval index)
- No multi-folder "engine" architecture (marketing, no substance benefit for a solo submission)

## Repository structure

```
rank.py                  # entry point — produces submission.csv
jd_profile.py            # structured JD representation (all constants sourced from JD text)
features.py              # per-candidate structured scoring functions
scoring.py               # composite score formula
honeypot.py              # structural-contradiction detection
reasoning.py             # grounded, varied per-candidate explanations
requirements.txt
submission_metadata.yaml
tests/
  test_pipeline.py       # 29 unit + integration tests
```

## Known limitations

- The "company founded X years ago" honeypot example from the README cannot be checked without a company-founding-year dataset, which is not provided in candidate_schema.json. All other 9 structural-contradiction rules are implemented.
- TF-IDF does not capture paraphrase-level semantic similarity (e.g., "vector store" vs "vector database"). A sentence-transformer pass over a pre-filtered top-N shortlist would improve semantic precision, but adds deployment complexity.
- Score distribution is compressed (most candidates fall between 0.25–0.55) because the JD is quite specific and few candidates in a general 100K pool are ideal matches on all dimensions simultaneously.
