#!/usr/bin/env python3
"""
rank.py

Produces the final ranked-candidate submission CSV.

Usage:
    python rank.py --candidates path/to/candidates.jsonl --out team_xxx.csv [--top-n 100]

Design notes (for the Stage 5 "defend your choices" interview):

  - Semantic fit uses TF-IDF + cosine similarity, NOT a downloaded dense
    embedding model (sentence-transformers/BGE/E5). This is a deliberate
    tradeoff, not a shortcut:
      1. The ranking step must run CPU-only, no network, <=5 min, <=16GB RAM.
         TF-IDF fit+transform over 100K short documents is sub-second to a
         few seconds, fully deterministic, and needs zero external model
         download — so it's reproducible on literally any judge's machine
         with just scikit-learn installed.
      2. Dense embeddings would add real value for true semantic
         paraphrase-level matching, but most of the differentiating signal
         in this dataset is structured (career history, skills, behavioral
         signals) rather than prose-similarity — see features.py, which
         carries 80% of the composite weight.
      3. If swapping in dense embeddings later: encode_candidates() is the
         only function that needs to change; everything downstream consumes
         a plain similarity array.

  - Everything that can be computed per-record (structural features,
    honeypot risk, behavioral multiplier) is computed in a single streaming
    pass over the JSONL so we never hold all 100K raw candidate dicts in
    memory at once — only the lightweight derived results.
"""

import argparse
import csv
import json
import sys
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import jd_profile as jp
import features as ft
import scoring as sc
import reasoning as rs


def build_reasoning_summary_fields(candidate: dict) -> dict:
    """Small subset of raw fields kept around purely so reasoning.py /
    debugging has human-readable context, without retaining the full
    candidate object for all 100K rows."""
    profile = candidate.get("profile", {}) or {}
    return {
        "title": profile.get("current_title", ""),
        "years_of_experience": profile.get("years_of_experience", 0),
        "location": profile.get("location", ""),
    }


def run(candidates_path: str, out_path: str, top_n: int = 100, limit: int = None):
    t0 = time.time()

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )

    candidate_ids = []
    texts = []
    base_partials = []
    behav_mults = []
    risks = []
    breakdowns = []
    notes_list = []
    summaries = []

    n_read = 0
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            n_read += 1

            text = ft.all_text(candidate)
            base_partial, behav_mult, risk, partial_breakdown, notes = sc.score_structural(candidate)

            candidate_ids.append(candidate["candidate_id"])
            texts.append(text)
            base_partials.append(base_partial)
            behav_mults.append(behav_mult)
            risks.append(risk)
            breakdowns.append(partial_breakdown)
            notes_list.append(notes)
            summaries.append(build_reasoning_summary_fields(candidate))

            if limit and n_read >= limit:
                break

    t_parse = time.time()
    print(f"[rank.py] parsed + scored structural features for {n_read} candidates "
          f"in {t_parse - t0:.1f}s", file=sys.stderr)

    # --- Semantic fit: TF-IDF over (candidate texts + JD text), vectorized.
    corpus = texts + [jp.JD_TEXT]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vec = tfidf_matrix[-1]
    candidate_matrix = tfidf_matrix[:-1]
    sem_sims = cosine_similarity(candidate_matrix, jd_vec).ravel()

    t_tfidf = time.time()
    print(f"[rank.py] TF-IDF semantic fit computed in {t_tfidf - t_parse:.1f}s "
          f"(vocab size {len(vectorizer.vocabulary_)})", file=sys.stderr)

    # --- Combine into final scores
    results = []
    for i in range(n_read if not limit else len(candidate_ids)):
        final, base = sc.finalize_score(base_partials[i], float(sem_sims[i]), behav_mults[i], risks[i])
        breakdown = dict(breakdowns[i])
        breakdown["semantic_fit"] = round(float(sem_sims[i]), 4)
        breakdown["base_weighted_score"] = base
        breakdown["final_score"] = round(final, 4)
        results.append((candidate_ids[i], final, breakdown, notes_list[i], summaries[i]))

    # Sort: score descending (rounded to 4dp, matching CSV output), tie-break
    # candidate_id ascending (validator requirement — ties must sort by id asc)
    results.sort(key=lambda r: (-round(r[1], 4), r[0]))
    top = results[:top_n]

    t_score = time.time()
    print(f"[rank.py] scored + sorted all {len(results)} candidates in "
          f"{t_score - t_tfidf:.1f}s", file=sys.stderr)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, final, breakdown, notes, summary) in enumerate(top, start=1):
            reasoning_text = rs.generate_reasoning(cid, breakdown, notes)
            writer.writerow([cid, rank, f"{final:.4f}", reasoning_text])

    t_end = time.time()
    print(f"[rank.py] wrote {len(top)} rows to {out_path}", file=sys.stderr)
    print(f"[rank.py] TOTAL runtime: {t_end - t0:.1f}s", file=sys.stderr)

    return top


def main():
    parser = argparse.ArgumentParser(description="Rank candidates against the JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only read first N lines (for fast local testing)")
    args = parser.parse_args()
    run(args.candidates, args.out, top_n=args.top_n, limit=args.limit)


if __name__ == "__main__":
    main()
