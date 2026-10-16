"""Query-Sensitive Adaptive Router for Federated RAG.

Provides:
- Domain Intent Classification via keyword matching
- Verbatim Extraction Probing Detection (structural & syntactic anomaly checks)
- Query Complexity Analysis (Single-Domain vs Multi-Domain / Ambiguous)
- Dynamic Client Routing: Selective client broadcast for single-domain queries,
  full broadcast fallback for multi-domain, ambiguous, or probing queries.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set
from src.client import ClientNode


DEFAULT_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "Healthcare / Medical": [
        "cardiovascular", "cardiology", "heart", "blood pressure", "hypertension",
        "statin", "ldl", "cholesterol", "diabetes", "metformin", "glucose",
        "hba1c", "mrna", "vaccine", "spike-protein", "immunity", "humoral",
        "oncology", "chemotherapy", "melanoma", "antibodies", "immunotherapy",
        "ehr", "hipaa", "phi", "antimicrobial", "mrsa", "c. difficile",
        "stroke", "ischemic", "tpa", "sepsis", "resuscitation", "antibiotics",
        "lactate", "clinical", "hospital", "medical", "patient", "therapy",
        "pharmacotherapy", "triage", "disease", "treatment", "pediatric",
        "asthma", "alzheimer", "dialysis", "coronary",
    ],
    "Education / Academics": [
        "formative", "summative", "assessment", "educator", "student", "retention",
        "cognitive load", "instructional design", "schema", "working memory",
        "accreditation", "peer review", "faculty", "curriculum", "flipped classroom",
        "pedagogy", "lecture", "seminar", "syllabus", "rubric", "active learning",
        "stem", "lms", "learning management", "academic", "university",
        "teaching", "school", "exam", "dissertation", "thesis", "pedagogies",
        "instruction", "higher education", "grading", "learning objectives",
    ],
    "Industrial / Engineering": [
        "predictive maintenance", "cnc", "milling", "vibration", "spindle",
        "bearing", "plc", "programmable logic", "ladder logic", "fieldbus",
        "modbus", "profinet", "scada", "telemetry", "substation", "pipeline",
        "hydraulic", "fea", "finite element", "stress", "fatigue", "tensile",
        "palletizer", "agv", "warehouse", "heat exchanger", "counterflow",
        "refinery", "six sigma", "dmaic", "refractory", "furnace", "industrial",
        "engineering", "manufacturing", "machinery", "factory", "assembly line",
        "automation", "bearing failure", "mechanical", "metallurgy",
    ],
    "General Knowledge / History": [
        "renaissance", "florence", "humanism", "classical greek", "roman",
        "industrial revolution", "steam engine", "james watt", "textile",
        "mesopotamia", "tigris", "euphrates", "cuneiform", "canal irrigation",
        "western roman empire", "476 ce", "germanic", "silk road", "trade route",
        "space race", "apollo", "lunar landing", "nasa", "magna carta",
        "runnymede", "rule of law", "feudal", "enlightenment", "rationalism",
        "history", "ancient", "historical", "century", "empire", "civilization",
        "reformation", "middle ages", "bastille", "revolution",
    ],
}


class QueryAnalyzerAndRouter:
    """Analyzes incoming search queries and determines optimal client routing."""

    def __init__(
        self,
        domain_keywords: Optional[Dict[str, List[str]]] = None,
        probe_length_threshold: int = 18,
    ) -> None:
        """Initialize the QueryAnalyzerAndRouter.

        Args:
            domain_keywords: Custom keyword dictionary mapping domain names to word lists.
            probe_length_threshold: Word count threshold for detecting verbatim document probes.
        """
        self.domain_keywords: Dict[str, List[str]] = (
            domain_keywords if domain_keywords is not None else DEFAULT_DOMAIN_KEYWORDS
        )
        self.probe_length_threshold: int = probe_length_threshold

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Performs domain classification, complexity analysis, and probe detection.

        Args:
            query: Raw search query text.

        Returns:
            Dict[str, Any]: Analysis containing:
                - 'query': Original query string
                - 'is_probe': Boolean flag for verbatim probe detection
                - 'domain_scores': Dict mapping domain names to keyword match counts
                - 'matched_domains': List of domains with at least 1 keyword match
                - 'complexity': 'Single-Domain (Low Complexity)' or 'Multi-Domain / Ambiguous (High Complexity)'
                - 'primary_domain': Dominant domain if single-domain, else None
        """
        clean_query = query.strip()
        words = clean_query.split()
        lower_query = clean_query.lower()

        # 1. Structural Probe Detection
        # Verbatim extraction probes often feature sentence-level length, verbatim quotes, or punctuation
        has_quotes = ('"' in clean_query or "'" in clean_query) and len(words) >= 6
        is_long_quote = len(words) >= self.probe_length_threshold
        has_dense_punctuation = clean_query.count(",") + clean_query.count(";") >= 2 and len(words) >= 12

        is_probe = bool(is_long_quote or has_quotes or has_dense_punctuation)

        # 2. Keyword Matching across domains
        domain_scores: Dict[str, int] = {}
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for kw in keywords:
                # Whole-phrase or whole-word regex matching
                pattern = r"\b" + re.escape(kw.lower()) + r"\b"
                matches = len(re.findall(pattern, lower_query))
                score += matches
            domain_scores[domain] = score

        # Identify domains with matches
        matched_domains = [d for d, s in domain_scores.items() if s > 0]
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        top_domain, top_score = sorted_domains[0]
        second_domain, second_score = sorted_domains[1] if len(sorted_domains) > 1 else (None, 0)

        # 3. Complexity Classification
        if is_probe:
            complexity = "Probing Query (High Risk / Full Broadcast)"
            primary_domain = top_domain if top_score > 0 else None
        elif len(matched_domains) == 0:
            complexity = "Multi-Domain / Ambiguous (High Complexity)"
            primary_domain = None
        elif len(matched_domains) == 1:
            complexity = "Single-Domain (Low Complexity)"
            primary_domain = top_domain
        else:
            # Multiple domains matched: check for clear dominance
            if top_score >= 2 and top_score >= 2 * second_score:
                complexity = "Single-Domain (Low Complexity)"
                primary_domain = top_domain
            else:
                complexity = "Multi-Domain / Ambiguous (High Complexity)"
                primary_domain = None

        return {
            "query": query,
            "is_probe": is_probe,
            "domain_scores": domain_scores,
            "matched_domains": matched_domains,
            "complexity": complexity,
            "primary_domain": primary_domain,
        }

    def select_clients(
        self,
        query_analysis: Dict[str, Any],
        clients: List[ClientNode],
    ) -> List[ClientNode]:
        """Selects target ClientNodes based on query analysis.

        - If multi-domain, ambiguous, or detected verbatim probe: routes to all clients.
        - If single-domain: routes only to matched client node(s), skipping irrelevant partitions.

        Args:
            query_analysis: Result dictionary from analyze_query().
            clients: Full list of available ClientNode instances.

        Returns:
            List[ClientNode]: Subset of ClientNodes to broadcast the query to.
        """
        is_probe = query_analysis.get("is_probe", False)
        primary_domain = query_analysis.get("primary_domain")
        complexity = query_analysis.get("complexity", "")

        # Fallback to full broadcast if query is probing, multi-domain, or ambiguous
        if is_probe or primary_domain is None or "Single-Domain" not in complexity:
            return list(clients)

        # Filter down to the matching domain's client(s)
        selected = [c for c in clients if c.domain_name.strip().lower() == primary_domain.strip().lower()]

        # Safety fallback: if no matching client found by exact name, broadcast to all
        return selected if selected else list(clients)
