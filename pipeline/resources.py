"""
Shared model handles for stage pipelines.

One Resources instance is shared by every stage of every pipeline built from
it, so MedGemma / CNN / SAM3 / BiomedCLIP load once (same idea as the pipeline
cache in ui/demo.py). Everything loads lazily on first use.
"""

import threading


class Resources:
    def __init__(self, cfg=None):
        if cfg is None:
            from config import DEFAULT_CONFIG

            cfg = DEFAULT_CONFIG
        self.cfg = cfg
        self._lock = threading.Lock()
        self._agents = None
        self._forest = None
        self._debate = None

    @property
    def agents(self) -> tuple:
        """(medgemma, cnn, sam3, clip), as returned by pipeline.graph.load_agents."""
        with self._lock:
            if self._agents is None:
                from pipeline.graph import load_agents

                self._agents = load_agents(self.cfg)
            return self._agents

    @property
    def medgemma(self):
        return self.agents[0]

    @property
    def cnn(self):
        return self.agents[1]

    @property
    def sam3(self):
        return self.agents[2]

    @property
    def clip(self):
        return self.agents[3]

    @property
    def forest(self):
        """AgentForest over the shared MedGemma, configured from cfg as in assemble_forest_pipeline."""
        medgemma = self.medgemma
        with self._lock:
            if self._forest is None:
                from pipeline.graph import make_forest

                self._forest = make_forest(medgemma, self.cfg)
            return self._forest

    @property
    def debate(self):
        """DebateOrchestrator over the shared MedGemma, as in assemble_debate_pipeline."""
        medgemma = self.medgemma
        with self._lock:
            if self._debate is None:
                from agents.debate import DebateOrchestrator

                self._debate = DebateOrchestrator(medgemma)
            return self._debate
