"""Deterministic experiment protocol execution and procedure compliance engine.

Components:
- protocol_schema.py: Pydantic protocol models, step definition validation, violations.
- protocol_loader.py: JSON file and dict loader with strict schema and integrity checks.
- alerts.py: Cooldown throttling and deterministic alert formatting.
- rule_engine.py: Deterministic rule evaluator (no LLM, explainable transitions).
- engine.py: Procedural finite-state machine and session lifecycle coordinator.
"""

from vyom.protocol.alerts import AlertCooldownTracker, ProtocolAlert
from vyom.protocol.engine import ProcedureEngine
from vyom.protocol.protocol_loader import (
    ProtocolLoadError,
    ProtocolNotFoundError,
    ProtocolValidationError,
    load_protocol_from_dict,
    load_protocol_from_file,
    load_protocol_from_json,
)
from vyom.protocol.protocol_schema import (
    ProcedureEvaluationResult,
    ProcedureViolation,
    ProtocolDefinition,
    ProtocolLifecycleState,
    ProtocolState,
    ProtocolStep,
    StepProgress,
    StepState,
    TargetZoneRequirement,
    ViolationType,
)
from vyom.protocol.rule_engine import ProcedureRuleEngine, RuleEvaluationOutcome

__all__ = [
    "AlertCooldownTracker",
    "ProtocolAlert",
    "ProcedureEngine",
    "ProcedureRuleEngine",
    "RuleEvaluationOutcome",
    "ProtocolDefinition",
    "ProtocolStep",
    "TargetZoneRequirement",
    "StepState",
    "ProtocolState",
    "ProtocolLifecycleState",
    "ViolationType",
    "ProcedureViolation",
    "StepProgress",
    "ProcedureEvaluationResult",
    "load_protocol_from_file",
    "load_protocol_from_json",
    "load_protocol_from_dict",
    "ProtocolLoadError",
    "ProtocolNotFoundError",
    "ProtocolValidationError",
]
