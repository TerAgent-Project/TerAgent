"""teragent.pipeline.tracing — TAP tracing for self-RL data constitution

Phase 10: Self-Reinforcement Learning Data Constitution

TAP traces are the core output of the teragent library. Every TAP call
(request → response) is recorded as a structured trace. Combined with
deterministic checklist results, these traces automatically form DPO
(Direct Preference Optimization) training pairs.

Design principles (Data Constitution):
    1. TAP traces are a core library output, independent of specific Agent flows
    2. Preference labels come from deterministic checks (AST, syntax, runnability),
       not from human annotation
    3. Data belongs to the user — the library never uploads traces

Architecture:
    TAPRequest  →  TAPTracer.record_request()  →  JSONL trace file
    TAPResponse →  TAPTracer.record_response() →  JSONL trace file
    Checklist   →  TAPTracer.record_checklist() →  JSONL trace file
                                                          ↓
                                           TAPTracer.export_dpo_pairs()
                                                          ↓
                                         (chosen=PASS, rejected=FAIL) pairs

Usage:
    from teragent.pipeline.tracing import TAPTracer

    tracer = TAPTracer(trace_dir="/project/.agent/traces")

    # Record a TAP request
    trace_id = await tracer.record_request(tap_request)

    # Record a TAP response
    await tracer.record_response(tap_response, task_id="1.1", trace_id=trace_id)

    # Record checklist result (for DPO pair generation)
    await tracer.record_checklist("1.1", checklist_data)

    # Export DPO preference pairs
    pairs = tracer.export_dpo_pairs()
    # [{"prompt": ..., "chosen": ..., "rejected": ..., "task_id": ..., "intent": ...}, ...]

    # Export all trace records
    traces = tracer.export_traces()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from teragent.core.tap import TAPRequest, TAPResponse

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TraceRecord:
    """A single trace record stored as one JSONL line.

    Each record captures a discrete event in the TAP lifecycle:
    - tap_request: Before model compilation
    - tap_response: After model response
    - checklist_result: After deterministic code check

    Attributes:
        trace_id: Unique identifier linking request → response → checklist
        timestamp: Unix timestamp (seconds since epoch)
        record_type: One of "tap_request", "tap_response", "checklist_result"
        task_id: Task identifier (e.g., "1.1", "2.3")
        intent: TAP intent (e.g., "code_generation", "design")
        data: Type-specific payload dict
    """

    trace_id: str = ""
    timestamp: float = 0.0
    record_type: str = ""  # "tap_request" | "tap_response" | "checklist_result"
    task_id: str = ""
    intent: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict for JSONL storage."""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "record_type": self.record_type,
            "task_id": self.task_id,
            "intent": self.intent,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TraceRecord:
        """Deserialize from dict (JSONL line)."""
        return cls(
            trace_id=d.get("trace_id", ""),
            timestamp=d.get("timestamp", 0.0),
            record_type=d.get("record_type", ""),
            task_id=d.get("task_id", ""),
            intent=d.get("intent", ""),
            data=d.get("data", {}),
        )


@dataclass
class DPOPair:
    """A Direct Preference Optimization training pair.

    Represents a (chosen, rejected) preference pair derived from TAP traces
    and deterministic checklist results.

    Preference label source:
        - chosen: TAP response where deterministic checklist PASS
        - rejected: TAP response where deterministic checklist FAIL
        - Labels come from objective code verification (AST, syntax, runnability),
          not from subjective human annotation

    Attributes:
        prompt: The TAPRequest serialized as text (the "question")
        chosen: TAP response text from a PASS result (the "good" answer)
        rejected: TAP response text from a FAIL result (the "bad" answer)
        task_id: Task identifier for grouping
        intent: TAP intent (e.g., "code_generation")
        source: Where the preference label came from (always "deterministic_check")
        metadata: Additional metadata (compiler, adapter, model, etc.)
    """

    prompt: str = ""
    chosen: str = ""
    rejected: str = ""
    task_id: str = ""
    intent: str = ""
    source: str = "deterministic_check"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict for export."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "task_id": self.task_id,
            "intent": self.intent,
            "source": self.source,
            "metadata": self.metadata,
        }

    def validate(self, allow_partial: bool = False) -> list[str]:
        """Validate the DPO pair completeness.

        Args:
            allow_partial: If True, allow pairs with only chosen or only
                rejected (partial pairs). If False (default), both must be
                non-empty.

        Returns:
            List of validation error strings (empty = valid)
        """
        errors: list[str] = []
        if not self.prompt:
            errors.append("DPOPair.prompt is empty")
        if not self.task_id:
            errors.append("DPOPair.task_id is empty")

        if allow_partial:
            # Partial pairs: at least one of chosen/rejected must be non-empty
            if not self.chosen and not self.rejected:
                errors.append("DPOPair: both chosen and rejected are empty")
        else:
            # Full pairs: both chosen and rejected must be non-empty
            if not self.chosen:
                errors.append("DPOPair.chosen is empty")
            if not self.rejected:
                errors.append("DPOPair.rejected is empty")

        if self.chosen and self.rejected and self.chosen == self.rejected:
            errors.append("DPOPair.chosen and rejected are identical — no preference signal")
        return errors


@dataclass
class DataConstitution:
    """Self-RL Data Constitution — the governing principles for TAP trace data.

    This is a declarative data structure that captures the three core principles
    of the teragent data constitution. It is attached to TAPTracer instances
    and included in exported data to ensure downstream consumers understand
    the provenance and constraints of the data.

    Principles:
        1. TAP traces are a core library output, independent of specific Agent flows
        2. Preference labels come from deterministic checks (AST, syntax, runnability),
           not from human annotation
        3. Data belongs to the user — the library never uploads traces

    Attributes:
        version: Constitution version (for forward compatibility)
        principles: The three core principles as strings
        preference_source: How preference labels are derived
        data_ownership: Who owns the data
        upload_policy: Library's stance on data uploading
    """

    version: str = "1.0"
    principles: list[str] = field(default_factory=lambda: [
        "TAP traces are a core library output, independent of specific Agent flows",
        "Preference labels come from deterministic checks (AST, syntax, runnability), "
        "not from human annotation",
        "Data belongs to the user — the library never uploads traces",
    ])
    preference_source: str = "deterministic_check"
    data_ownership: str = "user"
    upload_policy: str = "never"

    def to_dict(self) -> dict:
        """Serialize to dict for inclusion in exported data."""
        return {
            "version": self.version,
            "principles": self.principles,
            "preference_source": self.preference_source,
            "data_ownership": self.data_ownership,
            "upload_policy": self.upload_policy,
        }


# ============================================================================
# Trace Statistics
# ============================================================================

@dataclass
class TraceStats:
    """Statistics about collected TAP traces.

    Attributes:
        total_records: Total number of trace records
        request_count: Number of tap_request records
        response_count: Number of tap_response records
        checklist_count: Number of checklist_result records
        task_ids: Set of unique task IDs seen
        intents: Set of unique intents seen
        dpo_pair_count: Number of DPO pairs that can be generated
        pass_count: Number of tasks that passed checklist
        fail_count: Number of tasks that failed checklist
    """

    total_records: int = 0
    request_count: int = 0
    response_count: int = 0
    checklist_count: int = 0
    task_ids: set[str] = field(default_factory=set)
    intents: set[str] = field(default_factory=set)
    dpo_pair_count: int = 0
    pass_count: int = 0
    fail_count: int = 0

    def to_dict(self) -> dict:
        """Serialize to dict (converts sets to sorted lists)."""
        return {
            "total_records": self.total_records,
            "request_count": self.request_count,
            "response_count": self.response_count,
            "checklist_count": self.checklist_count,
            "task_ids": sorted(self.task_ids),
            "intents": sorted(self.intents),
            "dpo_pair_count": self.dpo_pair_count,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
        }


# ============================================================================
# TAPTracer — Core Tracing Engine
# ============================================================================

class TAPTracer:
    """TAP trace recorder and DPO pair generator.

    The central class for Phase 10's self-RL data constitution. Records
    TAP requests, responses, and checklist results as structured JSONL traces,
    then generates DPO preference pairs by matching responses with their
    deterministic check outcomes.

    Thread safety:
        All write operations are thread-safe via an internal lock.
        Read operations (export, stats) operate on a snapshot.

    File format:
        Each trace file is a JSONL file where each line is a valid JSON object
        representing a TraceRecord. Lines are appended atomically (write + flush + fsync).

    DPO pair generation:
        Pairs are generated by matching task_ids across request → response → checklist
        records. If the same task_id has both a PASS and a FAIL response, they form
        a (chosen=PASS, rejected=FAIL) pair. If only one outcome exists, partial
        pairs are still generated (with only chosen or only rejected populated).

    Usage:
        tracer = TAPTracer(trace_dir="/project/.agent/traces")

        # Auto-tracing with ModelProvider
        provider = ModelProvider(compiler=..., adapter=..., model=..., tracer=tracer)

        # Manual tracing
        trace_id = await tracer.record_request(tap_request)
        await tracer.record_response(tap_response, task_id="1.1", trace_id=trace_id)

        # After running deterministic checks
        await tracer.record_checklist("1.1", {
            "fail_count": 0, "warn_count": 2, "ok_count": 3,
            "needs_repair": False, ...
        })

        # Export
        pairs = tracer.export_dpo_pairs()
        traces = tracer.export_traces()
        stats = tracer.get_trace_stats()
    """

    def __init__(
        self,
        trace_dir: str = ".agent/traces",
        enabled: bool = True,
        max_trace_size_mb: float = 100.0,
        constitution: DataConstitution | None = None,
    ) -> None:
        """Initialize TAPTracer.

        Args:
            trace_dir: Directory for JSONL trace files (created if not exists)
            enabled: Whether tracing is active (can be disabled for testing)
            max_trace_size_mb: Maximum total trace file size before rotation
            constitution: Data constitution (defaults to standard if not provided)
        """
        self.trace_dir = trace_dir
        self.enabled = enabled
        self.max_trace_size_mb = max_trace_size_mb
        self.constitution = constitution or DataConstitution()

        # Internal state
        self._lock = threading.Lock()
        self._records: list[TraceRecord] = []
        self._trace_file: str = ""
        self._session_id: str = uuid.uuid4().hex[:12]

        if self.enabled:
            os.makedirs(self.trace_dir, exist_ok=True)
            self._trace_file = os.path.join(
                self.trace_dir,
                f"trace_{self._session_id}_{int(time.time())}.jsonl"
            )
            logger.info(f"TAPTracer initialized: trace_dir={self.trace_dir}, session={self._session_id}")

    # ===== Core Recording Methods =====

    def _generate_trace_id(self, task_id: str) -> str:
        """Generate a unique trace ID for linking request → response → checklist."""
        return f"{task_id}_{uuid.uuid4().hex[:8]}"

    def _sync_write_record(self, record: TraceRecord) -> None:
        """Synchronously write a trace record to JSONL file (for run_in_executor)."""
        if not self._trace_file:
            return
        try:
            line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
            with open(self._trace_file, 'a', encoding='utf-8') as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.warning(f"Failed to write trace record: {e}")

    async def record_request(
        self,
        request: TAPRequest,
        trace_id: str | None = None,
    ) -> str:
        """Record a TAP request before compilation.

        Args:
            request: The TAP request to record
            trace_id: Optional trace ID (auto-generated if not provided)

        Returns:
            The trace_id for linking with subsequent record_response() and record_checklist()
        """
        if not self.enabled:
            return trace_id or self._generate_trace_id(request.meta.get("task_id", "unknown"))

        task_id = request.meta.get("task_id", "unknown")
        intent = request.meta.get("intent", "unknown")

        if trace_id is None:
            trace_id = self._generate_trace_id(task_id)

        record = TraceRecord(
            trace_id=trace_id,
            timestamp=time.time(),
            record_type="tap_request",
            task_id=task_id,
            intent=intent,
            data={
                "meta": request.meta,
                "instruction": request.instruction,
                "constraints": request.constraints,
                "output_format_hint": request.output_format_hint,
                "context_keys": list(request.context.keys()),
            },
        )

        with self._lock:
            self._records.append(record)

        # Write to file asynchronously
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_write_record, record)

        return trace_id

    async def record_response(
        self,
        response: TAPResponse,
        task_id: str,
        trace_id: str = "",
        intent: str = "",
    ) -> None:
        """Record a TAP response after model API call.

        Args:
            response: The TAP response to record
            task_id: Task identifier (for linking with request and checklist)
            trace_id: Trace ID from the corresponding record_request() call
            intent: TAP intent (for indexing)
        """
        if not self.enabled:
            return

        record = TraceRecord(
            trace_id=trace_id or self._generate_trace_id(task_id),
            timestamp=time.time(),
            record_type="tap_response",
            task_id=task_id,
            intent=intent,
            data={
                "raw_text_length": len(response.raw_text) if response.raw_text else 0,
                "raw_text": response.raw_text or "",
                "usage": response.usage,
                "is_empty": not (response.raw_text and response.raw_text.strip()),
            },
        )

        with self._lock:
            self._records.append(record)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_write_record, record)

    async def record_checklist(
        self,
        task_id: str,
        checklist_data: dict,
        trace_id: str = "",
        intent: str = "",
    ) -> None:
        """Record a deterministic checklist result.

        This is the key method for DPO pair generation. The checklist data
        determines whether a (request, response) pair is "chosen" (PASS)
        or "rejected" (FAIL).

        Args:
            task_id: Task identifier
            checklist_data: Structured checklist result dict with keys:
                - fail_count: Number of FAIL issues
                - warn_count: Number of WARN issues
                - ok_count: Number of OK checks
                - has_critical_warn: Whether critical warnings were found
                - needs_repair: Whether the code needs repair
                - issues: List of issue dicts
            trace_id: Trace ID linking to the original request
            intent: TAP intent
        """
        if not self.enabled:
            return

        # Derive pass/fail from checklist data
        passed = (
            checklist_data.get("fail_count", 0) == 0
            and not checklist_data.get("has_critical_warn", False)
            and checklist_data.get("warn_count", 0) <= 3
        )

        record = TraceRecord(
            trace_id=trace_id or self._generate_trace_id(task_id),
            timestamp=time.time(),
            record_type="checklist_result",
            task_id=task_id,
            intent=intent,
            data={
                **checklist_data,
                "passed": passed,
            },
        )

        with self._lock:
            self._records.append(record)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_write_record, record)

    # ===== DPO Pair Generation =====

    def export_dpo_pairs(
        self,
        min_pairs: int = 1,
        include_partial: bool = False,
    ) -> list[dict]:
        """Generate DPO preference pairs from collected traces.

        Pairs are generated by matching task_ids across records:
        - Find tasks that have both request + response + checklist records
        - For each task, determine PASS/FAIL from checklist
        - Group by (task_id, intent) and form (chosen=PASS, rejected=FAIL) pairs

        Two pairing strategies:
        1. Full pairs: Same task_id has both PASS and FAIL responses
           → (chosen=PASS_response, rejected=FAIL_response)
        2. Cross-task pairs: Different task_ids with same intent
           → (chosen=PASS_task_response, rejected=FAIL_task_response)

        Args:
            min_pairs: Minimum number of pairs required (log warning if fewer)
            include_partial: Whether to include pairs with only chosen or rejected

        Returns:
            List of DPO pair dicts with keys:
            prompt, chosen, rejected, task_id, intent, source, metadata, constitution
        """
        with self._lock:
            records_snapshot = list(self._records)

        # Index records by (task_id, record_type)
        by_task: dict[str, dict[str, list[TraceRecord]]] = {}
        for r in records_snapshot:
            if r.task_id not in by_task:
                by_task[r.task_id] = {}
            by_task[r.task_id].setdefault(r.record_type, []).append(r)

        # Classify tasks as PASS or FAIL based on checklist
        task_outcomes: dict[str, dict] = {}  # task_id → {passed: bool, request, response, checklist, intent}
        for task_id, type_records in by_task.items():
            requests = type_records.get("tap_request", [])
            responses = type_records.get("tap_response", [])
            checklists = type_records.get("checklist_result", [])

            if not requests and not responses:
                continue

            # Get the latest records for each type
            latest_request = requests[-1] if requests else None
            latest_response = responses[-1] if responses else None
            latest_checklist = checklists[-1] if checklists else None

            # Determine PASS/FAIL
            passed = False
            if latest_checklist:
                passed = latest_checklist.data.get("passed", False)
            elif latest_response:
                # No checklist — infer from response emptiness
                passed = not latest_response.data.get("is_empty", True)

            intent = (latest_request or latest_response or latest_checklist).intent

            task_outcomes[task_id] = {
                "passed": passed,
                "request": latest_request,
                "response": latest_response,
                "checklist": latest_checklist,
                "intent": intent,
            }

        # Group by intent for cross-task pairing
        by_intent: dict[str, dict[str, list[str]]] = {}  # intent → {pass: [task_ids], fail: [task_ids]}
        for task_id, outcome in task_outcomes.items():
            intent = outcome["intent"] or "unknown"
            if intent not in by_intent:
                by_intent[intent] = {"pass": [], "fail": []}
            if outcome["passed"]:
                by_intent[intent]["pass"].append(task_id)
            else:
                by_intent[intent]["fail"].append(task_id)

        # Generate DPO pairs
        pairs: list[dict] = []

        # Strategy 1: Same-task pairs (task with both PASS and FAIL attempts)
        # This happens when a task is retried after failing checklist.
        #
        # Responses and checklists are associated by trace_id (not by
        # positional index) because a single task may have multiple
        # response/checklist pairs from retries, and positional pairing
        # can produce incorrect matchings when the counts differ.
        for task_id, type_records in by_task.items():
            responses = type_records.get("tap_response", [])
            checklists = type_records.get("checklist_result", [])
            requests = type_records.get("tap_request", [])

            if len(responses) < 2 or len(checklists) < 2:
                continue

            # Build a mapping from trace_id to response and checklist records
            trace_to_response: dict[str, TraceRecord] = {}
            trace_to_checklist: dict[str, TraceRecord] = {}
            for resp in responses:
                tid = resp.trace_id
                if tid:
                    trace_to_response[tid] = resp
            for chk in checklists:
                tid = chk.trace_id
                if tid:
                    trace_to_checklist[tid] = chk

            # Find PASS and FAIL responses by matching trace_ids
            pass_responses: list[TraceRecord] = []
            fail_responses: list[TraceRecord] = []

            # Match via shared trace_id between response and checklist
            for tid, chk in trace_to_checklist.items():
                is_pass = chk.data.get("passed", False)
                resp = trace_to_response.get(tid)
                if resp:
                    if is_pass:
                        pass_responses.append(resp)
                    else:
                        fail_responses.append(resp)

            # Also check checklists without matching trace_id (fallback)
            unmatched_checklists = [
                chk for chk in checklists
                if chk.trace_id not in trace_to_response
            ]
            for chk in unmatched_checklists:
                is_pass = chk.data.get("passed", False)
                # Find closest response by timestamp
                closest_resp = None
                min_dt = float('inf')
                for resp in responses:
                    dt = abs(resp.timestamp - chk.timestamp)
                    if dt < min_dt:
                        min_dt = dt
                        closest_resp = resp
                if closest_resp and min_dt < 60.0:  # within 60 seconds
                    if is_pass:
                        pass_responses.append(closest_resp)
                    else:
                        fail_responses.append(closest_resp)

            # Form pairs
            for pass_resp in pass_responses:
                for fail_resp in fail_responses:
                    prompt_text = self._build_prompt_text(requests[0] if requests else None)
                    pair = DPOPair(
                        prompt=prompt_text,
                        chosen=pass_resp.data.get("raw_text", ""),
                        rejected=fail_resp.data.get("raw_text", ""),
                        task_id=task_id,
                        intent=task_outcomes.get(task_id, {}).get("intent", "unknown"),
                        source="deterministic_check",
                        metadata={
                            "pairing_strategy": "same_task_retry",
                            "pass_trace_id": pass_resp.trace_id,
                            "fail_trace_id": fail_resp.trace_id,
                        },
                    )
                    validation_errors = pair.validate()
                    if not validation_errors:
                        pair_dict = pair.to_dict()
                        pair_dict["constitution"] = self.constitution.to_dict()
                        pairs.append(pair_dict)

        # Strategy 2: Cross-task pairs (different tasks with same intent)
        for intent, groups in by_intent.items():
            pass_task_ids = groups["pass"]
            fail_task_ids = groups["fail"]

            if not pass_task_ids or not fail_task_ids:
                continue

            for pass_task_id in pass_task_ids:
                for fail_task_id in fail_task_ids:
                    # Skip same-task pairs (already handled above)
                    if pass_task_id == fail_task_id:
                        continue

                    pass_outcome = task_outcomes[pass_task_id]
                    fail_outcome = task_outcomes[fail_task_id]

                    # Both must have request and response
                    if not pass_outcome["request"] or not pass_outcome["response"]:
                        continue
                    if not fail_outcome["request"] or not fail_outcome["response"]:
                        continue

                    # Use the PASS task's request as prompt (more representative of "good")
                    prompt_text = self._build_prompt_text(pass_outcome["request"])
                    pair = DPOPair(
                        prompt=prompt_text,
                        chosen=pass_outcome["response"].data.get("raw_text", ""),
                        rejected=fail_outcome["response"].data.get("raw_text", ""),
                        task_id=f"{pass_task_id}_vs_{fail_task_id}",
                        intent=intent,
                        source="deterministic_check",
                        metadata={
                            "pairing_strategy": "cross_task",
                            "pass_task_id": pass_task_id,
                            "fail_task_id": fail_task_id,
                        },
                    )
                    validation_errors = pair.validate()
                    if not validation_errors:
                        pair_dict = pair.to_dict()
                        pair_dict["constitution"] = self.constitution.to_dict()
                        pairs.append(pair_dict)

        # Strategy 3: Partial pairs (include_partial mode)
        # Partial pairs have only chosen or rejected (not both); they are
        # validated with allow_partial=True so they pass even when one
        # side is empty.
        if include_partial:
            for task_id, outcome in task_outcomes.items():
                if not outcome["request"] or not outcome["response"]:
                    continue

                # Check if already in a full pair
                already_paired = any(
                    p.get("task_id") == task_id or
                    task_id in p.get("task_id", "").split("_vs_")
                    for p in pairs
                )
                if already_paired:
                    continue

                prompt_text = self._build_prompt_text(outcome["request"])
                response_text = outcome["response"].data.get("raw_text", "")

                if outcome["passed"]:
                    pair = DPOPair(
                        prompt=prompt_text,
                        chosen=response_text,
                        rejected="",
                        task_id=task_id,
                        intent=outcome["intent"],
                        source="deterministic_check",
                        metadata={"pairing_strategy": "partial_chosen_only"},
                    )
                else:
                    pair = DPOPair(
                        prompt=prompt_text,
                        chosen="",
                        rejected=response_text,
                        task_id=task_id,
                        intent=outcome["intent"],
                        source="deterministic_check",
                        metadata={"pairing_strategy": "partial_rejected_only"},
                    )

                # Partial pair validation: allow one side to be empty
                validation_errors = pair.validate(allow_partial=True)
                if not validation_errors:
                    pair_dict = pair.to_dict()
                    pair_dict["constitution"] = self.constitution.to_dict()
                    pairs.append(pair_dict)

        if len(pairs) < min_pairs:
            logger.info(
                f"TAPTracer: Generated {len(pairs)} DPO pairs "
                f"(minimum requested: {min_pairs}). "
                f"Collect more traces with checklist results for better pair generation."
            )

        return pairs

    def _build_prompt_text(self, request_record: TraceRecord | None) -> str:
        """Build a text representation of a TAP request for DPO prompt field.

        This serializes the key fields of a TAPRequest into a structured text
        format suitable for DPO training data.
        """
        if request_record is None:
            return ""

        data = request_record.data
        parts: list[str] = []

        if data.get("instruction"):
            parts.append(f"Instruction: {data['instruction']}")

        if data.get("constraints"):
            constraints = data["constraints"]
            if isinstance(constraints, list):
                parts.append(f"Constraints: {json.dumps(constraints, ensure_ascii=False)}")

        if data.get("output_format_hint"):
            parts.append(f"Output format: {data['output_format_hint']}")

        if data.get("context_keys"):
            parts.append(f"Context: {', '.join(data['context_keys'])}")

        return "\n".join(parts)

    # ===== Export Methods =====

    def export_traces(self) -> list[dict]:
        """Export all trace records as a list of dicts.

        Returns:
            List of trace record dicts, each with constitution metadata
        """
        with self._lock:
            records = [r.to_dict() for r in self._records]

        return records

    def export_traces_jsonl(self, output_path: str | None = None) -> str:
        """Export all trace records as JSONL.

        Args:
            output_path: Output file path (defaults to trace_file)

        Returns:
            Path to the exported JSONL file
        """
        if output_path is None:
            output_path = self._trace_file or os.path.join(
                self.trace_dir, f"export_{int(time.time())}.jsonl"
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with self._lock:
            records = list(self._records)

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write constitution header
            header = {
                "type": "constitution",
                "data": self.constitution.to_dict(),
                "session_id": self._session_id,
                "export_time": time.time(),
            }
            f.write(json.dumps(header, ensure_ascii=False) + "\n")

            # Write all records
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(records)} trace records to {output_path}")
        return output_path

    def export_dpo_pairs_jsonl(self, output_path: str | None = None) -> str:
        """Export DPO preference pairs as JSONL.

        Args:
            output_path: Output file path (defaults to trace_dir/dpo_pairs_{ts}.jsonl)

        Returns:
            Path to the exported JSONL file
        """
        if output_path is None:
            output_path = os.path.join(
                self.trace_dir, f"dpo_pairs_{int(time.time())}.jsonl"
            )

        pairs = self.export_dpo_pairs()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write constitution header
            header = {
                "type": "dpo_export",
                "constitution": self.constitution.to_dict(),
                "pair_count": len(pairs),
                "export_time": time.time(),
            }
            f.write(json.dumps(header, ensure_ascii=False) + "\n")

            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(pairs)} DPO pairs to {output_path}")
        return output_path

    # ===== Statistics =====

    def get_trace_stats(self) -> TraceStats:
        """Get statistics about collected traces.

        Returns:
            TraceStats with counts, task_ids, intents, and DPO pair potential
        """
        with self._lock:
            records = list(self._records)

        stats = TraceStats(total_records=len(records))

        for r in records:
            if r.record_type == "tap_request":
                stats.request_count += 1
            elif r.record_type == "tap_response":
                stats.response_count += 1
            elif r.record_type == "checklist_result":
                stats.checklist_count += 1
                if r.data.get("passed", False):
                    stats.pass_count += 1
                else:
                    stats.fail_count += 1

            if r.task_id:
                stats.task_ids.add(r.task_id)
            if r.intent:
                stats.intents.add(r.intent)

        # Estimate DPO pair count
        stats.dpo_pair_count = len(self.export_dpo_pairs())

        return stats

    # ===== Management Methods =====

    def clear(self) -> None:
        """Clear all in-memory trace records (does not delete files)."""
        with self._lock:
            self._records.clear()

    def load_from_file(self, file_path: str | None = None) -> int:
        """Load trace records from a JSONL file into memory.

        Args:
            file_path: Path to JSONL file (defaults to current trace_file)

        Returns:
            Number of records loaded
        """
        if file_path is None:
            file_path = self._trace_file

        if not file_path or not os.path.isfile(file_path):
            return 0

        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    # Skip constitution header lines
                    if d.get("type") in ("constitution", "dpo_export"):
                        continue
                    record = TraceRecord.from_dict(d)
                    with self._lock:
                        self._records.append(record)
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        logger.info(f"Loaded {count} trace records from {file_path}")
        return count

    def load_all_traces(self) -> int:
        """Load all JSONL trace files from trace_dir.

        Returns:
            Total number of records loaded
        """
        if not os.path.isdir(self.trace_dir):
            return 0

        total = 0
        for filename in sorted(os.listdir(self.trace_dir)):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.trace_dir, filename)
                total += self.load_from_file(filepath)

        return total

    @property
    def session_id(self) -> str:
        """Current tracing session ID."""
        return self._session_id

    @property
    def trace_file(self) -> str:
        """Current trace file path."""
        return self._trace_file

    @property
    def is_enabled(self) -> bool:
        """Whether tracing is currently enabled."""
        return self.enabled

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __repr__(self) -> str:
        return (
            f"TAPTracer("
            f"session={self._session_id}, "
            f"records={len(self)}, "
            f"enabled={self.enabled}, "
            f"trace_dir={self.trace_dir!r})"
        )
