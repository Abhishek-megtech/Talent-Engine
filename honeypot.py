"""
honeypot.py

Rule-based detection of "subtly impossible" candidate profiles per
redrob_signals_doc.md / README.md:

  "~80 honeypot candidates with subtly impossible profiles (e.g., 8 years of
   experience at a company founded 3 years ago; 'expert' proficiency in 10
   skills with 0 years used)."

We do NOT have a company-founding-year field in the schema, so the literal
"company founded 3 years ago" example can't be checked directly — that's a
known, documented limitation (see README.md "Known limitations"). Everything
else below is checkable from candidate_schema.json fields and is intentionally
conservative: each rule is a hard structural contradiction, not a judgment
call, so we don't accidentally honeypot-flag unusual-but-real careers.

Returns a risk score in [0, 1] plus the list of triggered rule names, so the
reasoning generator can cite *why* a candidate was downweighted.
"""

from datetime import date, datetime


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def detect_honeypot(candidate: dict) -> tuple[float, list[str]]:
    reasons = []
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    education = candidate.get("education", []) or []
    skills = candidate.get("skills", []) or []

    yoe = profile.get("years_of_experience") or 0
    yoe_months = yoe * 12

    # --- Rule 1: any single skill's claimed duration exceeds total experience.
    for sk in skills:
        dm = sk.get("duration_months") or 0
        if dm > yoe_months + 6:  # 6-month grace for rounding / concurrent learning
            reasons.append(
                f"skill '{sk.get('name')}' duration ({dm}mo) exceeds total "
                f"experience ({yoe} yrs)"
            )
            break  # one citation is enough; don't spam reasons

    # --- Rule 2: "expert" proficiency with ~zero time spent on the skill.
    # This is the literal example given in the README.
    expert_zero = [
        sk.get("name") for sk in skills
        if sk.get("proficiency") == "expert" and (sk.get("duration_months") or 0) <= 2
    ]
    if len(expert_zero) >= 2:
        reasons.append(
            f"{len(expert_zero)} skills marked 'expert' with <=2 months use "
            f"(e.g. {expert_zero[0]})"
        )

    # --- Rule 3: implausibly large number of "expert" skills overall —
    # the doc's "expert proficiency in 10 skills" pattern.
    expert_count = sum(1 for sk in skills if sk.get("proficiency") == "expert")
    if expert_count >= 8:
        reasons.append(f"{expert_count} skills marked 'expert' (implausibly broad)")

    # --- Rule 4: career_history end_date before start_date, or duration_months
    # inconsistent with the start/end date delta by more than ~3 months.
    for job in career:
        sd = _parse_date(job.get("start_date"))
        ed = _parse_date(job.get("end_date")) if job.get("end_date") else None
        dm = job.get("duration_months")
        if sd and ed and ed < sd:
            reasons.append(f"role at '{job.get('company')}' ends before it starts")
            break
        if sd and ed and dm is not None:
            delta_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if abs(delta_months - dm) > 3:
                reasons.append(
                    f"role at '{job.get('company')}' duration_months ({dm}) doesn't "
                    f"match start/end dates (~{delta_months}mo)"
                )
                break

    # --- Rule 5: more than one role flagged is_current=True (can't currently
    # hold two jobs in this dataset's modeling).
    current_count = sum(1 for job in career if job.get("is_current"))
    if current_count > 1:
        reasons.append(f"{current_count} roles simultaneously flagged is_current")

    # --- Rule 6: total career_history duration grossly exceeds stated
    # years_of_experience (allowing generous overlap tolerance, since people
    # can hold concurrent roles/consulting gigs).
    total_career_months = sum(job.get("duration_months") or 0 for job in career)
    if total_career_months > yoe_months * 1.6 + 12:
        reasons.append(
            f"career_history total ({total_career_months}mo) far exceeds stated "
            f"experience ({yoe} yrs)"
        )

    # --- Rule 7: earliest job started before the candidate could plausibly
    # have entered the workforce given years_of_experience (e.g., experience
    # says 3 years but career history starts 12 years ago).
    if career:
        starts = [_parse_date(job.get("start_date")) for job in career]
        starts = [s for s in starts if s]
        if starts:
            earliest = min(starts)
            years_since_earliest = (date.today() - earliest).days / 365.25
            if years_since_earliest > yoe + 1.5:
                reasons.append(
                    f"earliest role started {years_since_earliest:.1f} yrs ago but "
                    f"stated experience is {yoe} yrs"
                )

    # --- Rule 8: education end_year is after the start_date of an
    # already-started job (graduated after starting full-time work, beyond a
    # reasonable internship/part-time overlap of 1 year).
    for edu in education:
        end_year = edu.get("end_year")
        if not end_year or not career:
            continue
        for job in career:
            sd = _parse_date(job.get("start_date"))
            if sd and sd.year < end_year - 1:
                reasons.append(
                    f"role at '{job.get('company')}' started {sd.year}, before "
                    f"stated education end year {end_year}"
                )
                break
        else:
            continue
        break

    # --- Rule 9: negative or zero-length roles.
    if any((job.get("duration_months") or 0) <= 0 for job in career):
        reasons.append("a career_history entry has zero/negative duration_months")

    # Risk score: scale with number of independent contradictions found.
    # 1 minor rule -> modest risk; 2+ -> treat as a strong honeypot signal.
    n = len(reasons)
    if n == 0:
        risk = 0.0
    elif n == 1:
        risk = 0.35
    elif n == 2:
        risk = 0.7
    else:
        risk = 0.95

    return risk, reasons
