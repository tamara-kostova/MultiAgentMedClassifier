"""
Agent Forest for neuroimaging diagnosis (Li et al., "More Agents Is All You Need").

N role-specialized MedGemma instances independently diagnose the same scan.
Results are combined via majority voting with confidence-weighted tiebreaking.

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


class AgentForest:
    """
    Ensemble of N role-specialized MedGemma agents with majority voting.

    Usage:
        forest = AgentForest(medgemma_agent)
        votes = forest.run(image_path, n_agents=3)
        consensus, winner_dx = forest.vote(votes)
    """

    def __init__(self, medgemma: MedGemmaAgent):
        self.medgemma = medgemma
        self._roles: list[dict] = []
        for role in FOREST_ROLES:
            prefix = (_PROMPTS_DIR / role["prompt_file"]).read_text().strip()
            self._roles.append({
                "name": role["name"],
                "description": role["description"],
                "prompt": prefix + "\n\n" + _BASE_PROMPT,
            })

    def run(self, image_path: str, n_agents: int = 3) -> list[dict]:
        """
        Run n_agents role-specialized instances on image_path.
        Cycles through roles if n_agents > 4.
        Returns list of vote dicts: {role, diagnosis_name, diagnosis_detailed,
        diagnosis_confidence, _dx}.
        """
        roles_to_use = (self._roles * ((n_agents // len(self._roles)) + 1))[:n_agents]
        votes = []
        for i, role in enumerate(roles_to_use):
            print(f"[forest] Agent {i + 1}/{n_agents} ({role['name']})...", flush=True)
            t0 = time.perf_counter()
            dx: MedicalDiagnosis = self.medgemma.diagnose_with_role(image_path, role["prompt"])
            elapsed = time.perf_counter() - t0
            print(
                f"[forest] Agent {i + 1} done ({elapsed:.1f}s): "
                f"{dx.diagnosis_name} conf={dx.diagnosis_confidence:.2f}"
            )
            votes.append({
                "role": role["name"],
                "diagnosis_name": dx.diagnosis_name or "unknown",
                "diagnosis_detailed": dx.diagnosis_detailed,
                "diagnosis_confidence": dx.diagnosis_confidence,
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

    def vote(
        self, votes: list[dict], task: str | None = None
    ) -> tuple[dict, MedicalDiagnosis]:
        """
        Majority vote with confidence-weighted tiebreaking.

        The vote is taken over the field that actually discriminates the task's
        classes (see vote_field): diagnosis_detailed for multiclass tumor subtyping,
        diagnosis_name otherwise. Agents whose vote field is empty fall back to
        diagnosis_name so a missing subtype never silently drops a ballot.

        Returns:
            consensus: serializable dict with winner, vote_counts, dissent_rate, etc.
            winner_dx: MedicalDiagnosis object for the first winning agent
                       (used for downstream routing decision derivation).
        """
        field = self.vote_field(task)

        def ballot(v: dict) -> str:
            return v.get(field) or v.get("diagnosis_name") or "unknown"

        labels = [ballot(v) for v in votes]
        counts = Counter(labels)
        majority_label, majority_count = counts.most_common(1)[0]

        winner_votes = [v for v in votes if ballot(v) == majority_label]
        conf_weighted = sum(v["diagnosis_confidence"] for v in winner_votes) / len(winner_votes)

        if field == "diagnosis_detailed":
            # The subtype is the winner; report the coarse label of its supporters.
            winner_detailed = majority_label
            coarse = [v["diagnosis_name"] for v in winner_votes if v.get("diagnosis_name")]
            winner_label = Counter(coarse).most_common(1)[0][0] if coarse else majority_label
        else:
            winner_label = majority_label
            detailed_labels = [
                v["diagnosis_detailed"] for v in winner_votes if v["diagnosis_detailed"]
            ]
            winner_detailed = (
                Counter(detailed_labels).most_common(1)[0][0] if detailed_labels else None
            )

        consensus = {
            "winner": winner_label,
            "winner_detailed": winner_detailed,
            "vote_field": field,
            "vote_counts": dict(counts),
            "vote_fraction": round(majority_count / len(votes), 4),
            "confidence_weighted_confidence": round(conf_weighted, 4),
            "dissent_rate": round((len(votes) - majority_count) / len(votes), 4),
            "n_agents": len(votes),
        }
        winner_dx = winner_votes[0]["_dx"]
        return consensus, winner_dx
