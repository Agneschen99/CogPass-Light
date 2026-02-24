# task_classifier.py
# Purpose: classify user tasks into Deep / Normal / Quick + provide an explanation (reason).
# MVP version: rule/keyword-based (stable, explainable, zero cost), with an easy hook for future LLMs.

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import re


@dataclass
class ClassificationResult:
    category: str                 # "deep" | "normal" | "quick"
    scores: Dict[str, int]        # {"deep":2,"normal":0,"quick":1}
    matched: Dict[str, List[str]] # {"deep":["research"],"quick":["reply"]}
    reason: str                   # single-sentence explanation for the user
    load_score: float             # 0–1 (used later for Top3 ranking)
    duration_minutes: int         # final estimated duration (user estimate preferred)


class TaskClassifier:
    def __init__(self):
        """
        Neuroscience analogy:
        - Deep Tasks   = high-load prefrontal cortex work (sustained focus)
        - Normal Tasks = medium cognitive load (requires some attention)
        - Quick Tasks  = more automatic processing (can be handled by basal ganglia)
        """

        # Keyword banks: prefer "verb + academic/cognitive object"
        self.deep_keywords = [
            "write", "design", "create", "analyze", "research", "plan",
            "develop", "code", "study", "learn", "solve", "calculate",
            "thesis", "paper", "project", "exam", "review", "chapter",
            "derivation", "proof", "statistics", "model", "experiment",
        ]

        self.normal_keywords = [
            "organize", "summarize", "prepare", "outline", "practice",
            "edit", "format", "compile", "arrange", "drill", "read",
            "review notes", "plan day",
        ]

        self.quick_keywords = [
            "check", "send", "reply", "scan", "print", "file",
            "email", "call", "message", "submit", "upload", "book",
            "buy", "pay", "schedule",
        ]

        # Pre‑compile English word-boundary regex to avoid false positives like "email" in "female"
        self._deep_patterns_en = [self._compile_word_pat(w) for w in self.deep_keywords if self._is_english(w)]
        self._normal_patterns_en = [self._compile_word_pat(w) for w in self.normal_keywords if self._is_english(w)]
        self._quick_patterns_en = [self._compile_word_pat(w) for w in self.quick_keywords if self._is_english(w)]

        # Chinese / non‑English still use substring matching (no whitespace word boundaries)
        self._deep_terms_zh = [w for w in self.deep_keywords if not self._is_english(w)]
        self._normal_terms_zh = [w for w in self.normal_keywords if not self._is_english(w)]
        self._quick_terms_zh = [w for w in self.quick_keywords if not self._is_english(w)]

    # ---------- Public API ----------

    def classify(
        self,
        task_name: str,
        estimated_minutes: Optional[int] = None,
        due_in_days: Optional[int] = None,  # reserved for future urgency modeling
    ) -> ClassificationResult:
        """
        Inputs:
          - task_name: task title
          - estimated_minutes: user‑provided duration in minutes (optional)
          - due_in_days: days until deadline (optional, reserved)

        Output: category + explanation + load_score + duration_minutes
        """

        title = (task_name or "").strip()
        title_en = title.lower()

        # 1) Keyword-based scores
        deep_score, deep_hit = self._score(title, title_en, "deep")
        normal_score, normal_hit = self._score(title, title_en, "normal")
        quick_score, quick_hit = self._score(title, title_en, "quick")

        scores = {"deep": deep_score, "normal": normal_score, "quick": quick_score}
        matched = {"deep": deep_hit, "normal": normal_hit, "quick": quick_hit}

        # 2) Conflict rule: when both quick + deep hit, do not let quick win.
        # Example: "Reply professor about research paper" → closer to deep/normal.
        category = self._pick_category(scores)

        # 3) Duration: prefer user input; otherwise base on category + complexity tweaks.
        duration_minutes = self._final_duration(title, category, estimated_minutes)

        # 4) load_score: 0–1, used for Top3 ranking (higher = more cognitive load).
        load_score = self._compute_load_score(category, scores, duration_minutes)

        # 5) Build human‑readable explanation (reason).
        reason = self._build_reason(category, duration_minutes, matched, estimated_minutes)

        return ClassificationResult(
            category=category,
            scores=scores,
            matched=matched,
            reason=reason,
            load_score=load_score,
            duration_minutes=duration_minutes,
        )

    # ---------- Internals ----------

    def _pick_category(self, scores: Dict[str, int]) -> str:
        d, n, q = scores["deep"], scores["normal"], scores["quick"]

        # Conflict: both deep and quick > 0 → do not allow quick to win.
        if d > 0 and q > 0:
            # Deep is treated as the "core" task type unless normal is clearly stronger.
            if n >= d:
                return "normal"
            return "deep"

        # Otherwise, winner‑take‑all.
        if d >= n and d >= q and d > 0:
            return "deep"
        if n >= q and n > 0:
            return "normal"
        if q > 0:
            return "quick"

        # No hits at all: default to normal.
        return "normal"

    def _final_duration(self, title: str, category: str, estimated_minutes: Optional[int]) -> int:
        # User estimate has priority (only clamp to a reasonable range).
        if estimated_minutes is not None:
            try:
                m = int(estimated_minutes)
                return max(5, min(m, 240))  # 5~240分钟
            except Exception:
                pass

        # Otherwise, use category‑based baseline.
        base = {"deep": 50, "normal": 30, "quick": 10}[category]
        bonus = 0

        # Complexity tweaks (a bit better than naïve "length > 30").
        # 1) Multiple steps: and/then → more complex.
        if re.search(r"\b(and|then)\b", title.lower()):
            bonus += 10

        # 2) Academic / high‑load objects → longer.
        if re.search(r"\b(chapter|thesis|paper|proof|derivation|exam|model|experiment)\b", title.lower()):
            bonus += 10 if category != "quick" else 5

        # 3) Quantifiers / numbers: 3 problems / 5 pages / 2 chapters → small bonus.
        if re.search(r"\d+", title):
            bonus += 5

        return max(5, min(base + bonus, 240))

    def _compute_load_score(self, category: str, scores: Dict[str, int], duration_minutes: int) -> float:
        # Category baseline.
        base = {"quick": 0.2, "normal": 0.5, "deep": 0.8}[category]

        # Keyword strength: more deep hits → higher; more quick hits → slightly lower.
        bump = min(scores["deep"] * 0.05, 0.15) - min(scores["quick"] * 0.03, 0.09)

        # Duration effect: 10–90 minutes mapped to 0–0.15.
        dur_bump = 0.0
        if duration_minutes > 10:
            dur_bump = min((duration_minutes - 10) / 80 * 0.15, 0.15)

        x = base + bump + dur_bump
        return max(0.0, min(x, 1.0))

    def _build_reason(
        self,
        category: str,
        duration_minutes: int,
        matched: Dict[str, List[str]],
        estimated_minutes: Optional[int],
    ) -> str:
        # One‑sentence explanation for UI.
        cat_en = {"deep": "deep task", "normal": "normal task", "quick": "quick task"}[category]

        hits = matched.get(category, [])
        hit_text = ""
        if hits:
            # At most 2 keywords to avoid being too verbose.
            hit_text = f"Matched keywords: {', '.join(hits[:2])}. "

        if estimated_minutes is not None:
            time_text = f"Using your estimated time of {duration_minutes} minutes."
        else:
            time_text = f"Estimated duration: about {duration_minutes} minutes."

        return f"Classified as a {cat_en}. {hit_text}{time_text}"

    def _score(self, title: str, title_en: str, bucket: str) -> Tuple[int, List[str]]:
        """
        bucket: "deep" | "normal" | "quick"
        Returns: (score, matched_keywords)
        """
        score = 0
        matched: List[str] = []

        # English: word‑boundary matching.
        patterns = {
            "deep": self._deep_patterns_en,
            "normal": self._normal_patterns_en,
            "quick": self._quick_patterns_en,
        }[bucket]

        keywords_en = [w for w in getattr(self, f"{bucket}_keywords") if self._is_english(w)]
        for pat, kw in zip(patterns, keywords_en):
            if pat.search(title_en):
                score += 1
                matched.append(kw)

        # Chinese / non‑English: substring matching.
        terms_zh = getattr(self, f"_{bucket}_terms_zh")
        for kw in terms_zh:
            if kw in title:
                score += 1
                matched.append(kw)

        return score, matched

    @staticmethod
    def _is_english(word: str) -> bool:
        # Treat as English keyword if it contains any a–z (good enough for our use case).
        return bool(re.search(r"[a-zA-Z]", word))

    @staticmethod
    def _compile_word_pat(word: str) -> re.Pattern:
        # Support multi‑word patterns like "review chapter": \breview\s+chapter\b
        word = word.strip().lower()
        parts = [re.escape(p) for p in word.split()]
        pat = r"\b" + r"\s+".join(parts) + r"\b"
        return re.compile(pat, re.IGNORECASE)


# ---- quick manual test ----
if __name__ == "__main__":
    clf = TaskClassifier()
    tests = [
        ("Reply professor about research paper", 10),
        ("Review chapter 3 statistics", None),
        ("Email landlord to pay rent", 5),
        ("整理文献并写论文大纲", None),
        ("买菜，然后提交作业", None),
        ("Flashcards sprint", 8),
    ]
    for t, mins in tests:
        r = clf.classify(t, estimated_minutes=mins)
        print(t, "=>", r.category, r.duration_minutes, r.load_score, r.reason, r.matched, r.scores)

