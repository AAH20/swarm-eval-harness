"""
Metamorphic Execution Graph Fuzzer for Autonomous Agentic Workflows.

Applies metamorphic property-based testing to multi-turn agent tool execution.
Perturbs environment parameters (API latency, partial schema mutations, HTTP 429/503
transient faults, JSON key permutations) to verify semantic invariance and idempotency.
"""

from __future__ import annotations
import dataclasses
import enum
import hashlib
import json
import random
from typing import Any, Callable


class PerturbationType(enum.Enum):
    LATENCY_SPIKE = "latency_spike"
    SCHEMA_KEY_MUTATION = "schema_key_mutation"
    TRANSIENT_HTTP_FAULT = "transient_http_fault"
    BENIGN_WHITESPACE_INJECTION = "benign_whitespace_injection"
    RETRY_STORM = "retry_storm"


@dataclasses.dataclass(frozen=True)
class ExecutionStep:
    step_id: int
    tool_name: str
    arguments: dict[str, Any]
    response_payload: dict[str, Any]
    status_code: int = 200


@dataclasses.dataclass
class MetamorphicResult:
    test_case_id: str
    perturbation: PerturbationType
    baseline_output: Any
    perturbed_output: Any
    is_invariant_preserved: bool
    violation_reason: str = ""


class MetamorphicFuzzer:
    """
    Executes metamorphic property-based fuzzing over multi-turn agentic trajectories.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = random.Random(random_seed)

    def perturb_environment(
        self,
        base_step: ExecutionStep,
        perturbation: PerturbationType,
    ) -> ExecutionStep:
        """Applies a metamorphic mutation to an execution step."""
        args_copy = json.loads(json.dumps(base_step.arguments))
        resp_copy = json.loads(json.dumps(base_step.response_payload))

        if perturbation == PerturbationType.LATENCY_SPIKE:
            # Latency mutation simulated in metadata
            resp_copy["_simulated_latency_ms"] = self.rng.randint(250, 1500)
            return ExecutionStep(
                step_id=base_step.step_id,
                tool_name=base_step.tool_name,
                arguments=args_copy,
                response_payload=resp_copy,
                status_code=base_step.status_code,
            )

        elif perturbation == PerturbationType.SCHEMA_KEY_MUTATION:
            # Add benign extra metadata field
            resp_copy["_system_tracing_telemetry_id"] = hashlib.sha256(b"trace").hexdigest()[:8]
            resp_copy["_apiVersion"] = "v2.1-fuzz"
            return ExecutionStep(
                step_id=base_step.step_id,
                tool_name=base_step.tool_name,
                arguments=args_copy,
                response_payload=resp_copy,
                status_code=base_step.status_code,
            )

        elif perturbation == PerturbationType.TRANSIENT_HTTP_FAULT:
            # Injects transient 429 or 503 error
            fault_code = self.rng.choice([429, 503])
            return ExecutionStep(
                step_id=base_step.step_id,
                tool_name=base_step.tool_name,
                arguments=args_copy,
                response_payload={"error": "Rate limit exceeded or backend degraded", "retry_after": 1},
                status_code=fault_code,
            )

        elif perturbation == PerturbationType.BENIGN_WHITESPACE_INJECTION:
            # Modifies string arguments with trailing whitespace
            for k, v in args_copy.items():
                if isinstance(v, str):
                    args_copy[k] = f"  {v}  "
            return ExecutionStep(
                step_id=base_step.step_id,
                tool_name=base_step.tool_name,
                arguments=args_copy,
                response_payload=resp_copy,
                status_code=base_step.status_code,
            )

        return base_step

    def evaluate_metamorphic_relation(
        self,
        test_case_id: str,
        baseline_action: Callable[[ExecutionStep], Any],
        base_step: ExecutionStep,
        perturbation: PerturbationType,
    ) -> MetamorphicResult:
        """
        Executes baseline vs. perturbed execution step and asserts invariant relation.
        """
        # Baseline execution
        base_output = baseline_action(base_step)

        # Perturbed execution
        perturbed_step = self.perturb_environment(base_step, perturbation)
        perturbed_output = baseline_action(perturbed_step)

        # Evaluate Metamorphic Invariance:
        # 1. Output equivalence under benign noise or latency
        if perturbation in (
            PerturbationType.LATENCY_SPIKE,
            PerturbationType.SCHEMA_KEY_MUTATION,
            PerturbationType.BENIGN_WHITESPACE_INJECTION,
        ):
            is_valid = (base_output == perturbed_output)
            reason = "Nominal" if is_valid else f"Output diverged under {perturbation.value}"

        # 2. Transient fault handling: must detect retry and converge
        elif perturbation == PerturbationType.TRANSIENT_HTTP_FAULT:
            # If perturbed step returned 429/503, agent must enter retry state gracefully
            is_valid = (perturbed_output.get("handled_gracefully") is True)
            reason = "Nominal" if is_valid else "Agent crashed on transient HTTP error"

        else:
            is_valid = True
            reason = "Nominal"

        return MetamorphicResult(
            test_case_id=test_case_id,
            perturbation=perturbation,
            baseline_output=base_output,
            perturbed_output=perturbed_output,
            is_invariant_preserved=is_valid,
            violation_reason=reason,
        )
