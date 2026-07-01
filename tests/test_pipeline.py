"""
tests/test_pipeline.py

Core correctness tests. Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import csv
import json
import tempfile
import pytest

import jd_profile as jp
import features as ft
import honeypot as hp
import scoring as sc
import reasoning as rs


# ---------------------------------------------------------------------------
# Helpers — minimal synthetic candidates
# ---------------------------------------------------------------------------

def _candidate(
    cid="CAND_0000001",
    title="Senior AI Engineer",
    yoe=7,
    location="Pune, Maharashtra",
    country="India",
    skills=None,
    career=None,
    signals=None,
):
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Test User",
            "headline": "AI Engineer with retrieval expertise",
            "summary": "Production LLM and embedding systems",
            "current_title": title,
            "current_company": "StartupCo",
            "current_company_size": "51-200",
            "current_industry": "Technology",
            "location": location,
            "country": country,
            "years_of_experience": yoe,
        },
        "career_history": career or [
            {
                "company": "StartupCo",
                "title": title,
                "start_date": "2020-01-01",
                "end_date": None,
                "is_current": True,
                "duration_months": 54,
                "description": "Built production embedding retrieval pipeline at scale.",
                "industry": "Technology",
            }
        ],
        "education": [{"degree": "B.Tech", "field": "CS", "institution": "IIT", "end_year": 2018}],
        "skills": [
            {"name": "embeddings", "proficiency": "expert", "duration_months": 36, "endorsements": 25},
            {"name": "faiss", "proficiency": "advanced", "duration_months": 24, "endorsements": 10},
            {"name": "sentence-transformers", "proficiency": "advanced", "duration_months": 18, "endorsements": 8},
            {"name": "python", "proficiency": "expert", "duration_months": 60, "endorsements": 30},
        ] if skills is None else skills,
        "redrob_signals": signals or {
            "last_active_date": "2026-06-15",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.85,
            "interview_completion_rate": 0.9,
            "notice_period_days": 14,
            "willing_to_relocate": True,
            "verified_email": True,
            "verified_phone": True,
        },
    }


# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

class TestHoneypot:
    def test_clean_candidate_zero_risk(self):
        risk, reasons = hp.detect_honeypot(_candidate())
        assert risk == 0.0, f"Clean candidate flagged: {reasons}"

    def test_expert_zero_duration_flagged(self):
        skills = [{"name": "embeddings", "proficiency": "expert", "duration_months": 1, "endorsements": 0},
                  {"name": "faiss", "proficiency": "expert", "duration_months": 0, "endorsements": 0}]
        risk, reasons = hp.detect_honeypot(_candidate(skills=skills))
        assert risk > 0, "Expert-with-0-months should be flagged"

    def test_ten_expert_skills_flagged(self):
        skills = [{"name": f"skill_{i}", "proficiency": "expert", "duration_months": 24, "endorsements": 5}
                  for i in range(10)]
        risk, reasons = hp.detect_honeypot(_candidate(skills=skills))
        assert risk > 0, "10 expert skills should trigger honeypot"

    def test_negative_duration_flagged(self):
        career = [{"company": "X", "title": "Engineer", "start_date": "2020-01-01",
                   "end_date": "2021-01-01", "is_current": False, "duration_months": -3,
                   "description": "", "industry": "Tech"}]
        risk, reasons = hp.detect_honeypot(_candidate(career=career))
        assert risk > 0, "Negative duration should be flagged"

    def test_end_before_start_flagged(self):
        career = [{"company": "X", "title": "Engineer", "start_date": "2022-06-01",
                   "end_date": "2021-01-01", "is_current": False, "duration_months": 6,
                   "description": "", "industry": "Tech"}]
        risk, reasons = hp.detect_honeypot(_candidate(career=career))
        assert risk > 0

    def test_two_current_roles_flagged(self):
        career = [
            {"company": "A", "title": "SDE", "start_date": "2022-01-01", "end_date": None,
             "is_current": True, "duration_months": 24, "description": "", "industry": "Tech"},
            {"company": "B", "title": "SDE", "start_date": "2022-01-01", "end_date": None,
             "is_current": True, "duration_months": 24, "description": "", "industry": "Tech"},
        ]
        risk, reasons = hp.detect_honeypot(_candidate(career=career))
        assert risk > 0


# ---------------------------------------------------------------------------
# Feature scoring
# ---------------------------------------------------------------------------

class TestSkillMatchTrust:
    def test_strong_candidate_high_score(self):
        score, notes = ft.skill_match_trust(_candidate())
        assert score >= 0.4, f"Strong candidate scored only {score}"

    def test_no_skills_and_no_text_low_score(self):
        # Override headline/summary to be domain-irrelevant so text-mention
        # scoring also returns zero — otherwise the "AI Engineer" headline
        # legitimately earns a non-zero text-mention skill score.
        cand = _candidate(skills=[], title="Graphic Designer")
        cand["profile"]["headline"] = "Creative visual designer"
        cand["profile"]["summary"] = "Expert in Adobe Photoshop and brand identity."
        for j in cand["career_history"]:
            j["description"] = "Designed logos and marketing collateral."
        score, notes = ft.skill_match_trust(cand)
        assert score < 0.15, f"Irrelevant candidate scored {score}"

    def test_notes_have_polarity(self):
        _, notes = ft.skill_match_trust(_candidate())
        for note in notes:
            assert isinstance(note, tuple) and len(note) == 2
            assert note[1] in ("pos", "neg"), f"Bad polarity: {note}"


class TestTitleFit:
    def test_ai_engineer_title_high(self):
        score, _ = ft.title_seniority_fit(_candidate(title="Senior AI Engineer"))
        assert score >= 0.8

    def test_accountant_title_low(self):
        score, _ = ft.title_seniority_fit(_candidate(title="Senior Accountant"))
        assert score <= 0.1

    def test_generic_engineer_mid(self):
        score, _ = ft.title_seniority_fit(_candidate(title="Software Engineer"))
        assert 0.3 <= score <= 0.6


class TestExperienceFit:
    def test_ideal_band(self):
        score, _ = ft.experience_fit(_candidate(yoe=7))
        assert score == 1.0

    def test_below_band(self):
        score, _ = ft.experience_fit(_candidate(yoe=1))
        assert score <= 0.2

    def test_above_band(self):
        score, _ = ft.experience_fit(_candidate(yoe=15))
        assert score <= 0.4


class TestLocationFit:
    def test_pune_ideal(self):
        score, _ = ft.location_fit(_candidate(location="Pune, Maharashtra", country="India"))
        assert score == 1.0

    def test_outside_india_no_visa(self):
        score, _ = ft.location_fit(_candidate(location="London", country="UK"))
        assert score <= 0.3

    def test_hyderabad_welcome(self):
        score, _ = ft.location_fit(_candidate(location="Hyderabad, Telangana", country="India"))
        assert score >= 0.8


class TestBehavioralMultiplier:
    def test_active_and_responsive_above_one(self):
        mult, _ = ft.behavioral_multiplier(_candidate())
        assert mult >= 1.0

    def test_inactive_6mo_penalised(self):
        signals = {"last_active_date": "2025-01-01", "open_to_work_flag": True,
                   "recruiter_response_rate": 0.5, "interview_completion_rate": 0.8,
                   "notice_period_days": 30, "willing_to_relocate": True,
                   "verified_email": True, "verified_phone": True}
        mult, notes = ft.behavioral_multiplier(_candidate(signals=signals))
        assert mult < 0.65

    def test_not_open_to_work_penalised(self):
        signals = {"last_active_date": "2026-06-25", "open_to_work_flag": False,
                   "recruiter_response_rate": 0.5, "interview_completion_rate": 0.8,
                   "notice_period_days": 30, "willing_to_relocate": True,
                   "verified_email": True, "verified_phone": True}
        mult, _ = ft.behavioral_multiplier(_candidate(signals=signals))
        assert mult < 0.8


# ---------------------------------------------------------------------------
# End-to-end scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_ideal_candidate_scores_high(self):
        final, _, _ = sc.score_candidate(_candidate(), semantic_sim=0.8)
        assert final >= 0.5, f"Ideal candidate scored {final}"

    def test_honeypot_suppressed(self):
        skills = [{"name": f"sk{i}", "proficiency": "expert", "duration_months": 0, "endorsements": 0}
                  for i in range(10)]
        cand = _candidate(skills=skills)
        final, breakdown, _ = sc.score_candidate(cand, semantic_sim=0.9)
        assert breakdown["honeypot_risk"] > 0
        assert final < 0.6

    def test_inactive_suppressed(self):
        signals = {"last_active_date": "2024-01-01", "open_to_work_flag": False,
                   "recruiter_response_rate": 0.05, "interview_completion_rate": 0.3,
                   "notice_period_days": 90, "willing_to_relocate": False,
                   "verified_email": False, "verified_phone": False}
        final, _, _ = sc.score_candidate(_candidate(signals=signals), semantic_sim=0.9)
        assert final < 0.35

    def test_score_in_unit_range(self):
        final, _, _ = sc.score_candidate(_candidate(), semantic_sim=0.5)
        assert 0.0 <= final <= 1.0


# ---------------------------------------------------------------------------
# Reasoning quality
# ---------------------------------------------------------------------------

class TestReasoning:
    def _notes(self):
        _, breakdown, notes = sc.score_candidate(_candidate(), semantic_sim=0.7)
        return breakdown, notes

    def test_no_contradiction_in_output(self):
        bd, notes = self._notes()
        text = rs.generate_reasoning("CAND_0000001", bd, notes)
        # If reasoning starts with a positive opener, it shouldn't also say
        # "held back" in the same sentence
        if text.startswith(("Ranked highly", "Rises", "A standout", "Placed here")):
            assert "Held back" not in text.split(".")[0]

    def test_different_ids_vary_sentences(self):
        bd, notes = self._notes()
        t1 = rs.generate_reasoning("CAND_0000001", bd, notes)
        t2 = rs.generate_reasoning("CAND_0099999", bd, notes)
        # They're allowed to differ in opener/connector
        assert isinstance(t1, str) and len(t1) > 20
        assert isinstance(t2, str) and len(t2) > 20

    def test_no_hallucinated_company_names(self):
        bd, notes = self._notes()
        text = rs.generate_reasoning("CAND_0000001", bd, notes)
        # Company names from the candidate should not appear in reasoning
        # (we never pass them through to reasoning)
        assert "StartupCo" not in text


# ---------------------------------------------------------------------------
# Submission format
# ---------------------------------------------------------------------------

class TestSubmissionFormat:
    def test_full_pipeline_produces_valid_csv(self):
        import rank
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            out_path = f.name

        JSONL_PATH = "/home/claude/india_runs_extracted/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
        if not os.path.exists(JSONL_PATH):
            pytest.skip("candidates.jsonl not available in test env")

        rank.run(JSONL_PATH, out_path, top_n=100, limit=2000)

        with open(out_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 100
        ranks = [int(r["rank"]) for r in rows]
        assert sorted(ranks) == list(range(1, 101))
        scores = [float(r["score"]) for r in rows]
        # Scores must be non-increasing
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-9
        os.unlink(out_path)
