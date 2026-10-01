"""
Agent Forest for neuroimaging diagnosis (Li et al., "More Agents Is All You Need").

N role-specialized MedGemma instances independently diagnose the same scan.
Ballots are canonicalized (eval.labels.canonical_label) and combined by majority
vote (or summed confidence); ties go to the label voted first in agent order.

The forest replaces the single triage node. All downstream nodes (CNN, SAM3,
BiomedCLIP, verification, report) run unchanged on the consensus routing decision.
"""

import time
from collections import Counter
from pathlib import Path

from agents.medgemma_agent import MedGemmaAgent, MedicalDiagnosis

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_BASE_PROMPT = (_PROMPTS_DIR / "system_prompt.txt").read_text()

FOREST_ROLES = [
    {
        "name": "radiologist",
        "prompt_file": "forest_radiologist.txt",
        "description": "Specialist neuroradiologist — visual pattern recognition",
    },
    {
        "name": "conservative",
        "prompt_file": "forest_conservative.txt",
        "description": "Conservative clinician — specificity-focused",
    },
    {
        "name": "emergency",
        "prompt_file": "forest_emergency.txt",
        "description": "Emergency specialist — sensitivity-focused",
    },
    {
        "name": "differential",
        "prompt_file": "forest_differential.txt",
        "description": "Differential diagnostician — uncertainty-aware",
    },
]


ROLE_NAMES = tuple(role["name"] for role in FOREST_ROLES)
VOTE_MODES = ("majority", "confidence")


def parse_roles(value) -> tuple | None:
    """'radiologist,emergency' (or an iterable) → validated role-name tuple; None = default."""
    if value is None:
        return None
    items = value.split(",") if isinstance(value, str) else list(value)
    roles = tuple(str(r).strip() for r in items if str(r).strip())
    unknown = [r for r in roles if r not in ROLE_NAMES]
    if unknown or not roles:
        raise ValueError(f"forest roles must be names from {list(ROLE_NAMES)}, got {items!r}")
    return roles


class AgentForest:
    """
    Ensemble of N role-specialized MedGemma agents with majority voting.

    Usage:
        forest = AgentForest(medgemma_agent)
        votes = forest.run(image_path, n_agents=3)
        consensus, winner_dx = forest.vote(votes, task="binary_tumor")

    roles: role names cycled to n_agents (default: FOREST_ROLES order).
    temperature/top_p/seed: 0.0 = greedy (published runs); >0 samples, agent i
    with seed + i. Duplicate roles under greedy decoding are refused (identical votes).
    """

    def __init__(
        self,
        medgemma: MedGemmaAgent,
        roles=None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        vote_mode: str = "majority",
    ):
        if vote_mode not in VOTE_MODES:
            raise ValueError(f"forest vote must be one of {VOTE_MODES}, got {vote_mode!r}")
        self.medgemma = medgemma
        self.role_names = parse_roles(roles) or ROLE_NAMES
        self.temperature = float(temperature or 0.0)
        self.top_p = float(top_p)
        self.seed = int(seed)
        self.vote_mode = vote_mode
        self._roles: dict[str, dict] = {}
        for role in FOREST_ROLES:
            prefix = (_PROMPTS_DIR / role["prompt_file"]).read_text().strip()
            self._roles[role["name"]] = {
                "name": role["name"],
                "description": role["description"],
                "prompt": prefix + "\n\n" + _BASE_PROMPT,
            }

    def roles_for(self, n_agents: int) -> list[dict]:
        """The role assignment for n_agents (cycled); raises on an invalid forest."""
        if not isinstance(n_agents, int) or n_agents < 1:
            raise ValueError(f"forest n_agents must be >= 1, got {n_agents!r}")
        names = [self.role_names[i % len(self.role_names)] for i in range(n_agents)]
        if self.temperature == 0 and len(set(names)) < len(names):
            raise ValueError(
                f"forest roles {names} contain duplicates under greedy decoding "
                "(temperature 0): duplicate agents would cast identical votes. "
                "Use distinct roles, fewer agents, or a temperature > 0."
            )
        return [self._roles[name] for name in names]

    def run(self, image_path: str, n_agents: int = 3) -> list[dict]:
        """
        Run n_agents instances on image_path (roles cycled, see roles_for).
        Returns list of vote dicts: {agent_idx, role, temperature, seed, diagnosis_name,
        diagnosis_detailed, diagnosis_confidence, parse_failed, _dx}.
        """
        roles_to_use = self.roles_for(n_agents)
        votes = []
        for i, role in enumerate(roles_to_use):
            seed = self.seed + i if self.temperature > 0 else None
            print(f"[forest] Agent {i + 1}/{n_agents} ({role['name']})...", flush=True)
            t0 = time.perf_counter()
            dx: MedicalDiagnosis = self.medgemma.diagnose_with_role(
                image_path, role["prompt"],
                temperature=self.temperature, top_p=self.top_p, seed=seed,
            )
            elapsed = time.perf_counter() - t0
            print(
                f"[forest] Agent {i + 1} done ({elapsed:.1f}s): "
                f"{dx.diagnosis_name} conf={dx.diagnosis_confidence:.2f}"
            )
            votes.append({
                "agent_idx": i,
                "role": role["name"],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": seed,
                "diagnosis_name": dx.diagnosis_name,
                "diagnosis_detailed": dx.diagnosis_detailed,
                "diagnosis_confidence": dx.diagnosis_confidence,
                "parse_failed": bool(getattr(dx, "parse_failed", False)),
                "_dx": dx,
            })
        return votes

    @staticmethod
    def vote_field(task: str | None) -> str:
        """Which diagnosis field carries the discriminative label for `task`.

        For multiclass_tumor the schema always emits diagnosis_name="tumor" and puts
        the subtype in diagnosis_detailed, so voting on diagnosis_name is degenerate
        (every agent trivially agrees). Mirrors eval_analysis.mg_diag_pred().
        """
        return "diagnosis_detailed" if task == "multiclass_tumor" else "diagnosis_name"

    @classmethod
    def ballot(cls, vote: dict, task: str | None) -> str:
        """Canonical label of one ballot; "" = abstain (empty or unparsable)."""
        from eval.labels import canonical_label

        if vote.get("parse_failed"):
            return ""
        primary = cls.vote_field(task)
        fallback = "diagnosis_name" if primary == "diagnosis_detailed" else "diagnosis_detailed"
        for field in (primary, fallback):
            label = canonical_label(vote.get(field), task)
            if label and label != "unknown":
                return label
        return ""

    def vote(
        self, votes: list[dict], task: str | None = None, mode: str | None = None
    ) -> tuple[dict, MedicalDiagnosis]:
        """
        Combine ballots over canonical labels (see ballot()).

        mode "majority" counts ballots, "confidence" sums diagnosis_confidence per
        label; either way ties go to the label cast first in agent order. Abstentions
        count in n_agents (so they lower vote_fraction) but never win.

        Returns:
            consensus: serializable dict (winner, winner_detailed, winner_canonical,
                       vote_counts, vote_fraction, dissent_rate, n_agents, n_abstain,
                       tie_broken, ...).
            winner_dx: MedicalDiagnosis of the first agent voting for the winner
                       (the first agent overall if every ballot abstained).
        """
        mode = mode or self.vote_mode
        if mode not in VOTE_MODES:
            raise ValueError(f"forest vote must be one of {VOTE_MODES}, got {mode!r}")
        if not votes:
            raise ValueError("forest vote needs at least one ballot")
        labels = [self.ballot(v, task) for v in votes]
        n = len(votes)
        n_abstain = sum(1 for lbl in labels if not lbl)
        counts: Counter = Counter()
        scores: dict[str, float] = {}
        for vote, lbl in zip(votes, labels):
            if not lbl:
                continue
            counts[lbl] += 1
            weight = 1.0 if mode == "majority" else float(vote.get("diagnosis_confidence") or 0.0)
            scores[lbl] = scores.get(lbl, 0.0) + weight

        if counts:
            order = list(dict.fromkeys(lbl for lbl in labels if lbl))  # agent order
            best = max(scores.values())
            tied = [lbl for lbl in order if scores[lbl] == best]
            winner_canonical = tied[0]
            tie_broken = len(tied) > 1
            winner_votes = [v for v, lbl in zip(votes, labels) if lbl == winner_canonical]
            winner_count = counts[winner_canonical]
            conf_weighted = sum(v["diagnosis_confidence"] for v in winner_votes) / len(winner_votes)
            names = [v["diagnosis_name"] for v in winner_votes if v.get("diagnosis_name")]
            details = [v["diagnosis_detailed"] for v in winner_votes if v.get("diagnosis_detailed")]
            winner_label = Counter(names).most_common(1)[0][0] if names else winner_canonical
            winner_detailed = Counter(details).most_common(1)[0][0] if details else None
            winner_dx = winner_votes[0]["_dx"]
        else:
            winner_canonical, tie_broken, winner_count = "", False, 0
            winner_label, winner_detailed, conf_weighted = None, None, 0.0
            winner_dx = votes[0]["_dx"]

        consensus = {
            "winner": winner_label,
            "winner_detailed": winner_detailed,
            "winner_canonical": winner_canonical,
            "vote_field": self.vote_field(task),
            "vote_mode": mode,
            "vote_counts": dict(counts),
            "vote_scores": {k: round(v, 4) for k, v in scores.items()},
            "vote_fraction": round(winner_count / n, 4),
            "confidence_weighted_confidence": round(conf_weighted, 4),
            "dissent_rate": round((n - winner_count) / n, 4),
            "n_agents": n,
            "n_abstain": n_abstain,
            "tie_broken": tie_broken,
        }
        return consensus, winner_dx
