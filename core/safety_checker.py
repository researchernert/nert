import os
import sys
import json
import yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from core.prompts import DECOMPOSED_SAFETY_PROMPT
from utils.log_formatter import log_info, log_ok, log_fail, log_substep

logger = logging.getLogger(__name__)

@dataclass
class SafetyResult:
    decision: str
    symbolic_result: str
    symbolic_trace: Dict
    neural_confidence: float
    support_ratio: float
    nearest_neighbors: List[Dict]
    explanation: str
    uncertainty_reason: Optional[str]
    danger_analysis: Optional[Dict] = None
    contrasting_examples: Optional[List[Dict]] = None

class DecomposedSymbolicChecker:

    def __init__(self, llm_client, temperature: float = 0.7):
        self.llm = llm_client
        self.debug_count = 0
        self.temperature = temperature

    def check(self, task: str, context: str = "") -> Tuple[str, Dict]:

        scene = context if context else "General environment"

        prompt = DECOMPOSED_SAFETY_PROMPT.format(
            task=task,
            scene=scene
        ).encode('utf-8').decode('utf-8')

        try:
            response = self.llm.call(prompt, max_tokens=2000, temperature=self.temperature)

            if not response or len(response.strip()) == 0:
                raise Exception("Empty response from LLM API")

            if "GEMINI_SAFETY_BLOCK:" in response or "GEMINI_RECITATION_BLOCK:" in response:
                logger.warning("Gemini blocked response - treating as REJECT")
                return "REJECT", {
                    'raw_response': response,
                    'final': 'REJECT',
                    'gemini_block': True
                }

            if self.debug_count < 3 and hasattr(self.llm, 'debug_mode'):
                print(f"\n=== LLM Response #{self.debug_count + 1} ===")
                print(f"Task: {task[:50]}...")
                print(f"Response:\n{response[:500]}...")
                print("=" * 50)
                self.debug_count += 1

            trace = self.parse_response(response)
            final_decision = trace.get('final', 'REJECT')

            if final_decision not in ['ACCEPT', 'REJECT']:
                logger.warning(f"Invalid decision: {final_decision}, defaulting to REJECT")
                final_decision = 'REJECT'

            return final_decision, trace

        except Exception as e:
            logger.error(f"Error in symbolic check: {e}")
            print(f"SAFETY_CHECKER_ERROR: {e}")
            print(f"SAFETY_CHECKER_ERROR_TYPE: {type(e)}")
            import traceback
            print(f"SAFETY_CHECKER_TRACEBACK: {traceback.format_exc()}")
            return "REJECT", {
                "error": str(e),
                "error_type": str(type(e)),
                "raw_response": str(response) if 'response' in locals() else "No response"
            }

    def parse_response(self, response: str) -> Dict:
        trace = {
            'raw_response': response,
            'sorts_and_values': '',
            'conflict_matrix': '',
            'societal': {'result': 'PERMISSIBLE', 'reasoning': ''},
            'organizational': {'result': 'PERMISSIBLE', 'reasoning': ''},
            'individual': {'result': 'PERMISSIBLE', 'reasoning': ''},
            'final': None
        }

        if not response:
            trace['final'] = 'REJECT'
            return trace

        lines = response.split('\n')

        sorts_and_values = []
        conflict_matrix = []
        societal_reasoning = []
        organizational_reasoning = []
        individual_reasoning = []

        in_sorts_section = False
        in_societal_section = False
        in_organizational_section = False
        in_individual_section = False
        in_conflict_matrix_section = False

        for line in lines:
            line_lower = line.lower().strip()
            original_line = line.strip()

            if 'preliminaries' in line_lower or ('sorts and values' in line_lower and not in_sorts_section):
                in_sorts_section = True
                in_societal_section = False
                in_organizational_section = False
                in_individual_section = False
                in_conflict_matrix_section = False
                if 'sorts and values' in line_lower:
                    continue
                continue
            elif 'societal alignment' in line_lower or 'layer s' in line_lower:
                in_sorts_section = False
                in_societal_section = True
                in_organizational_section = False
                in_individual_section = False
                in_conflict_matrix_section = False
                continue
            elif 'organizational' in line_lower and 'alignment' in line_lower:
                in_sorts_section = False
                in_societal_section = False
                in_organizational_section = True
                in_individual_section = False
                in_conflict_matrix_section = False
                continue
            elif 'individual alignment' in line_lower or 'layer i' in line_lower:
                in_sorts_section = False
                in_societal_section = False
                in_organizational_section = False
                in_individual_section = True
                in_conflict_matrix_section = False
                continue
            elif 'alignment conflict matrix' in line_lower:
                in_sorts_section = False
                in_societal_section = False
                in_organizational_section = False
                in_individual_section = False
                in_conflict_matrix_section = True
                continue
            elif 'resulting decision' in line_lower:
                in_sorts_section = False
                in_societal_section = False
                in_organizational_section = False
                in_individual_section = False
                in_conflict_matrix_section = False
                continue

            if original_line and not line_lower.startswith('**'):
                if in_sorts_section:
                    sorts_and_values.append(original_line)
                elif in_societal_section:
                    societal_reasoning.append(original_line)
                elif in_organizational_section:
                    organizational_reasoning.append(original_line)
                elif in_individual_section:
                    individual_reasoning.append(original_line)
                elif in_conflict_matrix_section:
                    conflict_matrix.append(original_line)

        trace['sorts_and_values'] = '\n'.join(sorts_and_values).strip()
        trace['conflict_matrix'] = '\n'.join(conflict_matrix).strip()
        trace['societal']['reasoning'] = '\n'.join(societal_reasoning).strip()
        trace['organizational']['reasoning'] = '\n'.join(organizational_reasoning).strip()
        trace['individual']['reasoning'] = '\n'.join(individual_reasoning).strip()

        for line in lines:
            line_lower = line.lower().strip()

            if 'society layer:' in line_lower or 'societal layer:' in line_lower:
                if 'forbidden' in line_lower:
                    trace['societal']['result'] = 'FORBIDDEN'
                elif 'permissible' in line_lower:
                    trace['societal']['result'] = 'PERMISSIBLE'

            elif 'organization layer:' in line_lower or 'organizational layer:' in line_lower:
                if 'forbidden' in line_lower:
                    trace['organizational']['result'] = 'FORBIDDEN'
                elif 'permissible' in line_lower:
                    trace['organizational']['result'] = 'PERMISSIBLE'

            elif 'individual layer:' in line_lower:
                if 'forbidden' in line_lower:
                    trace['individual']['result'] = 'FORBIDDEN'
                elif 'permissible' in line_lower:
                    trace['individual']['result'] = 'PERMISSIBLE'

            elif 'final decision:' in line_lower:
                if 'accept' in line_lower:
                    trace['final'] = 'ACCEPT'
                elif 'reject' in line_lower:
                    trace['final'] = 'REJECT'

        if trace['final'] is None:
            trace['final'] = self.compute_from_layers(trace)

        logger.debug(f"Parse result - Societal: {trace['societal']['result']}, Org: {trace['organizational']['result']}, Individual: {trace['individual']['result']}, Final: {trace['final']}")

        if trace['societal']['reasoning']:
            logger.debug(f"Societal reasoning extracted: {trace['societal']['reasoning'][:100]}...")
        if trace['organizational']['reasoning']:
            logger.debug(f"Organizational reasoning extracted: {trace['organizational']['reasoning'][:100]}...")
        if trace['individual']['reasoning']:
            logger.debug(f"Individual reasoning extracted: {trace['individual']['reasoning'][:100]}...")

        return trace

    def compute_from_layers(self, trace: Dict) -> str:
        s = trace['societal']['result'] == 'PERMISSIBLE'
        o = trace['organizational']['result'] == 'PERMISSIBLE'
        i = trace['individual']['result'] == 'PERMISSIBLE'

        if (s and o and i) or (not s and o and i):
            return 'ACCEPT'
        else:
            return 'REJECT'

class NeurosymbolicSafetyChecker:

    def __init__(self, llm_client, config: Dict, temperature: float = 0.7,
                 shared_confidence_estimator=None):
        """
        Initialize safety checker.

        Args:
            llm_client: LLM client for symbolic reasoning
            config: Configuration dictionary
            temperature: Temperature for LLM calls
            shared_confidence_estimator: Optional pre-loaded confidence estimator.
                If provided, this estimator will be shared across instances (thread-safe).
                If None, will attempt to load a new one (NOT thread-safe for parallel use).
        """
        self.symbolic_checker = DecomposedSymbolicChecker(llm_client, temperature=temperature)

        if shared_confidence_estimator is not None:
            self.confidence_estimator = shared_confidence_estimator
            self.has_confidence = True
            logger.debug("Using shared confidence estimator")
        else:
            try:
                from models.contrastive.contrastive_retrieval import SafetyConfidenceEstimator
                self.confidence_estimator = SafetyConfidenceEstimator(
                    config.get('model_path', 'models/trained_encoder.pt')
                )
                self.has_confidence = True
                logger.info("Confidence estimator loaded successfully")
            except Exception as e:
                try:
                    error_msg = str(e)
                except UnicodeEncodeError:
                    error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
                logger.warning(f"Could not load confidence estimator: {error_msg}")
                self.confidence_estimator = None
                self.has_confidence = False

        self.config = config
        self.llm = llm_client
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.confidence_unsafe_threshold = config.get('confidence_unsafe_threshold', 0.3)
        self.support_ratio_accept_threshold = config.get('support_ratio_accept_threshold', 0.4)
        self.confidence_accept_threshold = config.get('confidence_accept_threshold', 0.2)

    def check(self, task: str, scene_description: str = "", use_retrieval_context: bool = False) -> SafetyResult:

        symbolic_result, symbolic_trace = self.symbolic_checker.check(task, scene_description)

        if self.has_confidence and self.confidence_estimator:
            try:
                confidence_result = self.confidence_estimator.estimate_confidence(
                    task,
                    symbolic_result,
                    k_neighbors=5
                )
                confidence = confidence_result.get('confidence', 0.5)
                support_ratio = confidence_result.get('support_ratio', 0.5)
                similar_examples = confidence_result.get('nearest_neighbors', [])
                contrasting_examples = confidence_result.get('contrasting_examples', [])
                danger_analysis = confidence_result.get('danger_analysis', {})
                uncertainty_reason = confidence_result.get('uncertainty_reason')
            except Exception as e:
                try:
                    error_msg = str(e)
                except UnicodeEncodeError:
                    error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
                logger.warning(f"Confidence estimation failed: {error_msg}")
                confidence = 0.5
                support_ratio = 0.5
                similar_examples = []
                contrasting_examples = []
                danger_analysis = {}
                uncertainty_reason = f"Confidence error: {error_msg}"
        else:

            confidence = 0.5
            support_ratio = 0.5
            similar_examples = []
            contrasting_examples = []
            danger_analysis = {}
            uncertainty_reason = "No confidence estimator available"

        decision, explanation = self.make_final_decision(
            symbolic_result,
            confidence,
            support_ratio
        )

        self._log_safety_results(decision, symbolic_result, symbolic_trace, confidence,
                                 support_ratio, similar_examples, explanation)

        return SafetyResult(
            decision=decision,
            symbolic_result=symbolic_result,
            symbolic_trace=symbolic_trace,
            neural_confidence=confidence,
            support_ratio=support_ratio,
            nearest_neighbors=similar_examples[:5] if similar_examples else [],
            explanation=explanation,
            uncertainty_reason=uncertainty_reason,
            danger_analysis=danger_analysis,
            contrasting_examples=contrasting_examples[:5] if contrasting_examples else []
        )

    def make_final_decision(self, symbolic_result: str, confidence: float,
                           support_ratio: float) -> Tuple[str, str]:

        if symbolic_result not in ['ACCEPT', 'REJECT']:
            logger.warning(f"Invalid symbolic result: {symbolic_result}")
            return "REJECT", f"Invalid symbolic result: {symbolic_result}"

        if symbolic_result == "REJECT":
            decision = "REJECT"
            if confidence < self.confidence_unsafe_threshold:
                explanation = f"High confidence unsafe: Symbolic and neural agree"
            else:
                explanation = f"Rejected by symbolic analysis"

        elif symbolic_result == "ACCEPT":

            if not self.has_confidence or self.confidence_estimator is None:
                decision = "ACCEPT"
                explanation = "Accepted by symbolic analysis (no neural model)"

            elif support_ratio > self.support_ratio_accept_threshold:
                decision = "ACCEPT"
                explanation = f"Accepted: {support_ratio:.0%} of similar examples are safe"
            elif confidence > self.confidence_accept_threshold:
                decision = "ACCEPT"
                explanation = f"Accepted with {confidence:.2f} confidence"
            else:

                decision = "REJECT"
                explanation = f"Low confidence override: support={support_ratio:.0%}, conf={confidence:.2f}"

        else:

            decision = "REJECT"
            explanation = "Unknown state - defaulting to reject"

        return decision, explanation

    def _log_safety_results(self, decision: str, symbolic_result: str,
                            symbolic_trace: Dict, confidence: float,
                            support_ratio: float, similar_examples: List,
                            explanation: str):
        log_info("Symbolic Analysis:")
        log_substep(f"Result: {symbolic_result}")

        layers = [
            ('Societal', symbolic_trace.get('societal', {})),
            ('Organizational', symbolic_trace.get('organizational', {})),
            ('Individual', symbolic_trace.get('individual', {}))
        ]

        for layer_name, layer_data in layers:
            if layer_data:
                result = layer_data.get('result', 'N/A')
                reasoning = layer_data.get('reasoning', '')

                log_substep(f"{layer_name} Layer: {result}")

                if result and result != 'PERMISSIBLE' and reasoning:

                    reasoning_lines = reasoning.strip().split('\n')
                    for line in reasoning_lines:
                        if line.strip():
                            log_substep(f"  {line.strip()}", indent=6)

        log_info("")
        log_info("Neural Analysis:")
        log_substep(f"Confidence: {confidence:.2f}")
        log_substep(f"Support Ratio: {support_ratio:.1%}")

        if similar_examples and len(similar_examples) > 0:
            log_info("")
            log_info(f"Similar Examples (Top {min(3, len(similar_examples))}):")
            for i, example in enumerate(similar_examples[:3], 1):
                task_display = example.get('task', '')[:60]
                if len(example.get('task', '')) > 60:
                    task_display += '...'
                label = example.get('label', 'unknown')
                similarity = example.get('similarity', 0.0)
                log_substep(f"{i}. [{label.upper()}] {task_display} (sim: {similarity:.2f})")

        log_info("")
        if decision == "ACCEPT":
            log_ok(f"Decision: {decision}")
        else:
            log_fail(f"Decision: {decision}")
        log_substep(f"Reason: {explanation}")