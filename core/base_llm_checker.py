# nert/core/base_llm_checker.py
"""Base LLM safety checker and code generator (No NERT)."""

import logging
import time
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def check_safety_base_llm(task: str, scene_description: str, llm_client, temperature: float = 0.0) -> Dict:
    """
    BASE_LLM safety check for benchmarking.

    Uses the same prompt and parsing logic as the BASE_LLM interface.

    Args:
        task: Task description to classify
        scene_description: Environmental context (optional)
        llm_client: LLM client instance
        temperature: LLM temperature (0.0 = deterministic)

    Returns:
        dict with 'prediction': 'safe' or 'unsafe'
    """
    from core.prompts import BASE_LLM_SAFETY_PROMPT

    prompt = BASE_LLM_SAFETY_PROMPT.format(
        task=task,
        scene=scene_description if scene_description else 'General environment',
        objects='General household objects',
        skills='Standard household robot capabilities'
    )

    try:
        response = llm_client.call(prompt, max_tokens=1000, temperature=temperature)
        decision = _parse_safety_response_for_benchmark(response)
        prediction = "safe" if decision == "ACCEPT" else "unsafe"
        logger.info(f"BASE_LLM benchmark classification: {prediction} (decision: {decision})")
        return {
            'prediction': prediction,
            'success': True
        }

    except Exception as e:
        logger.error(f"BASE_LLM safety check failed: {e}")
        return {
            'prediction': 'unsafe',
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }


def _parse_safety_response_for_benchmark(response: str) -> str:
    """
    Parse LLM response for benchmark (same logic as interface).

    Expects format:
    Line 1: "Task Good to Assign" or "Task Not Approved"
    Line 2: Brief reason

    Args:
        response: Raw LLM response

    Returns:
        'ACCEPT' or 'REJECT'
    """
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]

    decision = 'REJECT'

    for i, line in enumerate(lines[:3]):
        line_lower = line.lower()

        if 'task good to assign' in line_lower:
            decision = 'ACCEPT'
            break
        elif 'task not approved' in line_lower or 'task not good to assign' in line_lower:
            decision = 'REJECT'
            break

    if decision == 'REJECT':
        response_lower = response.lower()
        if 'task good to assign' in response_lower:
            decision = 'ACCEPT'
        elif 'task not approved' in response_lower or 'task not good to assign' in response_lower:
            decision = 'REJECT'

    logger.debug(f"Parsed benchmark safety decision: {decision}")

    return decision


class BaseLLMChecker:
    """
    Simple LLM-based safety checker without neurosymbolic reasoning.

    Provides a baseline comparison to NERT by using direct LLM prompting
    for both safety evaluation and code generation.
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def safety_check(self, task: str, scene: str, objects: List[str],
                     skills: List[str]) -> Dict:
        from core.prompts import BASE_LLM_SAFETY_PROMPT

        prompt = BASE_LLM_SAFETY_PROMPT.format(
            task=task,
            scene=scene if scene else 'General environment',
            objects=', '.join(objects) if objects else 'None specified',
            skills='\n'.join([f'- {skill}' for skill in skills]) if skills else 'None specified'
        )

        start_time = time.time()

        try:
            response = self.llm.call(prompt, max_tokens=1000)
            response_time = time.time() - start_time

            decision, explanation = self._parse_safety_response(response)

            logger.info(f"Base LLM safety check: {decision} ({response_time:.2f}s)")

            return {
                'decision': decision,
                'explanation': explanation,
                'model_used': getattr(self.llm, 'model_name', 'unknown'),
                'response_time': response_time,
                'raw_response': response
            }

        except Exception as e:
            logger.error(f"Base LLM safety check failed: {e}")
            response_time = time.time() - start_time

            return {
                'decision': 'REJECT',
                'explanation': f'Safety check failed: {str(e)}',
                'model_used': getattr(self.llm, 'model_name', 'unknown'),
                'response_time': response_time,
                'error': str(e)
            }

    def _parse_safety_response(self, response: str) -> Tuple[str, str]:
        """
        Parse LLM response expecting 2-line format:
        Line 1: "Task Good to Assign" or "Task Not Approved"
        Line 2: Brief reason (max 100 chars)

        Returns:
            (decision, reason) tuple
        """
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]

        decision = 'REJECT'
        reason = 'Could not parse response'

        for i, line in enumerate(lines[:3]):
            line_lower = line.lower()

            if 'task good to assign' in line_lower:
                decision = 'ACCEPT'
                if i + 1 < len(lines):
                    reason = lines[i + 1][:100]  
                else:
                    reason = 'No reason provided'
                break
            elif 'task not approved' in line_lower:
                decision = 'REJECT'
                if i + 1 < len(lines):
                    reason = lines[i + 1][:100]
                else:
                    reason = 'No reason provided'
                break

        if decision == 'REJECT' and reason == 'Could not parse response':
            response_lower = response.lower()
            if 'task good to assign' in response_lower:
                decision = 'ACCEPT'
                reason = 'Parsed from verbose response'
            elif 'task not approved' in response_lower:
                decision = 'REJECT'
                reason = 'Parsed from verbose response'

        logger.info(f"Parsed safety decision: {decision} - {reason}")

        return decision, reason

    def generate_code(self, task: str, objects: List[str],
                     skills: List[str]) -> Dict:
        """
        Single-shot LLM code generation.

        Args:
            task: User task description
            objects: Available objects in scene
            skills: Robot capabilities

        Returns:
            Dict with code, success, error
        """
        from core.prompts import BASE_LLM_CODEGEN_PROMPT

        prompt = BASE_LLM_CODEGEN_PROMPT.format(
            task=task,
            objects=', '.join(objects) if objects else 'None available',
            skills='\n'.join([f'- {skill}' for skill in skills]) if skills else 'None available'
        )

        start_time = time.time()

        try:
            response = self.llm.call(prompt, max_tokens=2000)
            response_time = time.time() - start_time

            code = self._clean_code(response)
            code = self._normalize_robot_identifiers(code)

            logger.info(f"Base LLM code generation: {len(code)} chars ({response_time:.2f}s)")

            return {
                'code': code,
                'success': True,
                'response_time': response_time,
                'raw_response': response
            }

        except Exception as e:
            logger.error(f"Base LLM code generation failed: {e}")
            response_time = time.time() - start_time

            return {
                'code': None,
                'success': False,
                'error': str(e),
                'response_time': response_time
            }

    def _clean_code(self, code: str) -> str:
        """Clean generated code (same as NERT)."""
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        return code.strip()

    def _normalize_robot_identifiers(self, code: str) -> str:
        pattern = r'\brobot([2-9]|\d{2,})\b'
        normalized_code = re.sub(pattern, 'robot1', code)
        return normalized_code


def run_base_llm_pipeline(task: str, scene: str, floor_plan: int, model: str,
                          objects: List[str], skills: List[str],
                          llm_client, disabled_skills: List[str] = None) -> Dict:
    import platform

    if disabled_skills:
        skills = [s for s in skills if s not in disabled_skills]

    checker = BaseLLMChecker(llm_client)

    logger.info("Base LLM Step 1: Safety evaluation")
    safety_result = checker.safety_check(task, scene, objects, skills)

    result = {
        'mode': 'base_llm',
        'model': model,
        'stages': {
            'safety': safety_result
        }
    }

    if safety_result['decision'] == 'REJECT':
        logger.info("Task rejected by base LLM safety check")
        result['final_status'] = 'REJECTED_SAFETY'
        result['decision'] = 'REJECT'
        return result

    logger.info("Base LLM Step 2: Code generation")
    code_result = checker.generate_code(task, objects, skills)

    result['stages']['code_generation'] = code_result

    if not code_result['success'] or not code_result['code']:
        logger.error("Code generation failed")
        result['final_status'] = 'FAILED_CODE_GENERATION'
        result['decision'] = 'REJECT'
        return result

    logger.info("Base LLM Step 3: Execution/Simulation")

    if platform.system().lower() == 'linux':
        execution_result = {
            'platform': 'linux',
            'execution_mode': 'simulation_only',
            'code': code_result['code'],
            'message': 'Execution on Linux would happen here'
        }
    else:
        execution_result = {
            'platform': 'windows',
            'execution_mode': 'simulation_info',
            'code': code_result['code'],
            'message': 'Generated code ready for simulation'
        }

    result['stages']['execution'] = execution_result
    result['final_status'] = 'ACCEPTED_BASE_LLM'
    result['decision'] = 'ACCEPT'
    result['code'] = code_result['code']

    logger.info("Base LLM pipeline completed successfully")

    return result
