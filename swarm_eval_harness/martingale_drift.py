"""
Martingale State-Drift Estimator for Multi-Turn Agent Trajectories.

Models context decay and goal deviation across multi-turn agent execution
using a discrete-time sub-Martingale formulation:
Delta M_t = E[D(S_{t+1}, S_anchor) | F_t] - D(S_t, S_anchor)
Detects silent context poisoning and infinite tool execution loops.
"""

from __future__ import annotations
import dataclasses
import math
import re
from typing import Sequence


@dataclasses.dataclass(frozen=True)
class TrajectoryTurn:
    turn_index: int
    user_prompt: str
    agent_thought: str
    tool_called: str
    tool_args: str
    observation: str


@dataclasses.dataclass
class DriftTelemetry:
    current_turn: int
    semantic_distance: float
    martingale_value: float
    is_drift_critical: bool
    drift_velocity: float


class MartingaleDriftEstimator:
    """
    Tracks context entropy and semantic goal divergence over long-horizon trajectories.
    """

    def __init__(
        self,
        anchor_intent: str,
        critical_threshold: float = 0.65,
        drift_rate_limit: float = 0.08,
    ):
        self.anchor_intent = anchor_intent
        self.anchor_tokens = self._tokenize(anchor_intent)
        self.critical_threshold = critical_threshold
        self.drift_rate_limit = drift_rate_limit
        self.turn_history: list[TrajectoryTurn] = []
        self.martingale_trajectory: list[float] = [0.0]

    def _tokenize(self, text: str) -> set[str]:
        words = re.findall(r"\w+", text.lower())
        # Filter basic stopwords
        stopwords = {"the", "a", "an", "is", "in", "to", "for", "of", "and", "or", "on"}
        return {w for w in words if w not in stopwords and len(w) > 2}

    def _compute_semantic_distance(self, current_text: str) -> float:
        """
        Computes goal-retention semantic distance:
        D(S_t, S_anchor) = 1.0 - (|tokens_context & tokens_anchor| / |tokens_anchor|).
        """
        current_tokens = self._tokenize(current_text)
        if not self.anchor_tokens:
            return 0.0
        if not current_tokens:
            return 1.0

        intersection = self.anchor_tokens.intersection(current_tokens)
        coverage = len(intersection) / len(self.anchor_tokens)
        return max(0.0, min(1.0, 1.0 - coverage))

    def ingest_turn(self, turn: TrajectoryTurn) -> DriftTelemetry:
        """
        Ingests a new execution step and updates the sub-Martingale drift statistic.
        """
        self.turn_history.append(turn)
        t = len(self.turn_history)

        # Context text aggregates current thought + tool arguments + observation
        context_str = f"{turn.user_prompt} {turn.agent_thought} {turn.tool_called} {turn.tool_args} {turn.observation}"
        dist_t = self._compute_semantic_distance(context_str)

        # Martingale update: M_t = M_{t-1} + (dist_t - prior_mean)
        prior_M = self.martingale_trajectory[-1]
        prior_mean = sum(self.martingale_trajectory) / len(self.martingale_trajectory)
        delta = dist_t - prior_mean
        new_M = prior_M + delta
        self.martingale_trajectory.append(new_M)

        # Calculate drift velocity (slope of distance over past 3 turns)
        if t >= 3:
            recent_turns = [
                self._compute_semantic_distance(
                    f"{trn.user_prompt} {trn.agent_thought} {trn.tool_called} {trn.tool_args}"
                )
                for trn in self.turn_history[-3:]
            ]
            velocity = (recent_turns[-1] - recent_turns[0]) / 2.0
        else:
            velocity = 0.0

        # Track consecutive high-drift turns
        recent_dists = [
            self._compute_semantic_distance(
                f"{trn.user_prompt} {trn.agent_thought} {trn.tool_called} {trn.tool_args} {trn.observation}"
            )
            for trn in self.turn_history[-2:]
        ]
        consecutive_high_drift = len(recent_dists) >= 2 and all(d >= self.critical_threshold for d in recent_dists)

        # Critical drift detection:
        # 1. Consecutive turns exceeding threshold
        # 2. Distance threshold breached with positive velocity
        # 3. Cumulative Martingale drift exceeds bounds M_t > gamma * sqrt(t)
        martingale_bound = 1.0 * math.sqrt(max(1, t))
        is_critical = consecutive_high_drift or (dist_t >= self.critical_threshold and velocity > self.drift_rate_limit) or (new_M > martingale_bound)

        return DriftTelemetry(
            current_turn=t,
            semantic_distance=dist_t,
            martingale_value=new_M,
            is_drift_critical=is_critical,
            drift_velocity=velocity,
        )
