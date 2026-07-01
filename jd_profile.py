"""
jd_profile.py

Structured representation of the Redrob "Senior AI Engineer — Founding Team" JD.

This is NOT generic config — every list/weight below is traceable to a specific
sentence in job_description.md. Comments cite the source line so this can be
defended in a Stage 5 interview without hand-waving.

If the JD changes for a different run of this hackathon, this is the only file
that should need to change.
"""

# ---------------------------------------------------------------------------
# Free-text JD summary used to build the TF-IDF query vector for semantic fit.
# Deliberately weighted toward the "what you'd actually be doing" and "ideal
# candidate" sections, since those carry the real signal (vs. boilerplate).
# ---------------------------------------------------------------------------
JD_TEXT = """
Senior AI Engineer Founding Team. Own the intelligence layer of a recruiting
product: ranking, retrieval, and matching systems. Audit and rebuild a
candidate-JD matching system: embeddings, hybrid retrieval, LLM-based
re-ranking. Set up evaluation infrastructure: offline benchmarks, online A/B
testing, NDCG, MRR, MAP, P@K, offline-to-online correlation. Production
experience with embeddings-based retrieval systems: sentence-transformers,
OpenAI embeddings, BGE, E5. Handled embedding drift, index refresh,
retrieval-quality regression in production. Production experience with vector
databases or hybrid search infrastructure: Pinecone, Weaviate, Qdrant, Milvus,
OpenSearch, Elasticsearch, FAISS. Strong Python, code quality. Hands-on
experience designing evaluation frameworks for ranking systems. LLM
fine-tuning LoRA QLoRA PEFT. Learning-to-rank models XGBoost neural ranking.
HR-tech recruiting tech marketplace products. Distributed systems large-scale
inference optimization. Shipped an end-to-end ranking search or recommendation
system to real users at meaningful scale. Strong opinions about retrieval
hybrid vs dense, evaluation offline vs online, LLM integration fine-tune vs
prompt. Applied ML AI roles at product companies, not pure services.
"""

# ---------------------------------------------------------------------------
# Skills — "Things you absolutely need" (JD section: "The skills inventory")
# These get the heaviest weight in skill_match_trust.
# ---------------------------------------------------------------------------
CORE_REQUIRED_SKILLS = [
    "embeddings", "sentence-transformers", "sentence transformers",
    "openai embeddings", "bge", "e5", "embedding", "retrieval",
    "vector database", "vector search", "hybrid search",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "semantic search",
    "ndcg", "mrr", "map", "a/b testing", "ab testing",
    "evaluation framework", "ranking", "re-ranking", "reranking",
    "python",
]

# "Things we'd like you to have but won't reject you for"
NICE_TO_HAVE_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "xgboost", "learning to rank", "ltr",
    "hr-tech", "recruiting tech", "marketplace",
    "distributed systems", "large-scale inference", "inference optimization",
    "open source", "open-source",
]

# Signals of NLP / information-retrieval grounding, used to penalize
# CV/speech/robotics-only profiles per:
# "People whose primary expertise is computer vision, speech, or robotics
#  without significant NLP/IR exposure ... you'd be re-learning fundamentals"
NLP_IR_SIGNALS = [
    "nlp", "natural language", "retrieval", "search", "ranking", "embedding",
    "llm", "language model", "transformer", "bert", "gpt", "rag",
    "information retrieval", "text classification", "tokeniz",
]
NON_NLP_DOMAIN_SIGNALS = [
    "computer vision", "image classification", "object detection",
    "speech recognition", "asr", "tts", "robotics", "slam",
    "autonomous", "lidar", "gan", "image segmentation",
]

# Disqualifying employer pattern per:
# "People who have only worked at consulting firms ... in their entire career"
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
]

# Used to detect "pure research environment ... without any production
# deployment" disqualifier.
RESEARCH_ONLY_SIGNALS = [
    "research scientist", "research fellow", "research assistant",
    "phd candidate", "postdoc", "post-doctoral", "research intern",
    "academic", "university research", "research lab",
]
PRODUCTION_SIGNALS = [
    "production", "deployed", "shipped", "scale", "users", "live",
    "latency", "uptime", "on-call", "rollout", "real-time", "real time",
]

# Title-chaser detection per:
# "trajectory shows optimizing for Senior -> Staff -> Principal by switching
#  companies every 1.5 years"
TITLE_LADDER_KEYWORDS = ["senior", "staff", "principal", "lead", "head", "director"]

# "Senior engineer who hasn't written production code in the last 18 months
#  because they moved into 'architecture' or 'tech lead' roles"
NON_CODING_TITLE_KEYWORDS = ["architect", "engineering manager", "tech lead",
                             "director", "vp ", "head of"]

# Location: "Pune/Noida-preferred but flexible... Candidates in Hyderabad,
# Pune, Mumbai, Delhi NCR welcome to apply... Open to relocation candidates
# from Tier-1 Indian cities... Outside India: case-by-case, no visa sponsorship"
PRIMARY_LOCATIONS = ["pune", "noida"]
WELCOME_LOCATIONS = ["hyderabad", "pune", "mumbai", "delhi", "gurgaon",
                      "gurugram", "noida", "ncr"]
TIER1_INDIAN_CITIES = [
    "bangalore", "bengaluru", "mumbai", "delhi", "pune", "hyderabad",
    "chennai", "kolkata", "noida", "gurgaon", "gurugram", "ahmedabad",
]

# "we'd love sub-30-day notice. We can buy out up to 30 days."
NOTICE_PERIOD_IDEAL_DAYS = 30

# Experience band: "5-9 years... we'll seriously consider candidates outside
# the band if other signals are strong" -> soft band, not a hard filter.
EXPERIENCE_IDEAL_MIN = 5
EXPERIENCE_IDEAL_MAX = 9
EXPERIENCE_SOFT_MIN = 3   # below this, penalize harder
EXPERIENCE_SOFT_MAX = 13  # above this, mild penalty (overqualified / pure-architect risk)

# ---------------------------------------------------------------------------
# Composite weights. These sum to 1.0 across the non-multiplicative terms;
# honeypot risk and behavioral reliability are applied as multipliers on top,
# not as additive terms, because they should be able to sink an otherwise
# strong score (an unreachable or fabricated candidate is not "70% as good" -
# they're not usable).
# ---------------------------------------------------------------------------
WEIGHTS = {
    "semantic_fit": 0.20,       # TF-IDF cosine vs JD text
    "skill_match_trust": 0.25,  # core/nice-to-have skill coverage, trust-weighted
    "title_seniority_fit": 0.15,
    "career_quality": 0.20,     # production vs research, consulting-only, stability
    "experience_fit": 0.10,
    "location_fit": 0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
