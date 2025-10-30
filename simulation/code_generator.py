# nert/simulation/code_generator.py
"""Generate executable code with safety constraints."""

from typing import Dict, List, TYPE_CHECKING
import ast
import re

if TYPE_CHECKING:
    from core.llm_clients import BaseLLMClient


class SafetyConstrainedCodeGenerator:
    """Generate robot code that respects safety invariants."""

    def __init__(self, llm_client: 'BaseLLMClient') -> None:
        self.llm = llm_client
    
    def generate(self, task: str, invariants: Dict, scene_objects: List[str], relations: List[Dict] = None) -> str:
        relations_text = ""
        if relations:
            relations_text = f"""
        Relations Summary (use these EXACT object names):
        {relations}

        CRITICAL: Use the exact object names and capitalization as they appear in the Relations Summary above.
        For example, if Relations Summary shows "AlarmClock", use 'AlarmClock' not 'alarmclock'.
        If Relations Summary shows "CoffeeTable", use 'CoffeeTable' not 'cofee table' or 'coffee table'.
        """

        prompt = f"""
        Generate a sequence of robot actions to accomplish the following task.

        Task: {task}

        Available objects in scene: {scene_objects}
        {relations_text}
        IMPORTANT: You have access to ONE robot only. Use 'robot1' as the identifier (not robot2, robot3, etc.).

        Safety constraints that MUST be maintained:
        Preconditions: {invariants.get('pddl_preconditions', [])}
        Postconditions: {invariants.get('pddl_postconditions', [])}
        Invariants: {invariants.get('ltl_invariants', [])}
        Physical constraints: {invariants.get('stl_constraints', [])}
        Context: {invariants.get('llm_contextual', [])}

        Use ONLY these robot actions:
        - GoToObject(robot1, 'ObjectName') - Navigate robot to object
        - PickupObject(robot1, 'ObjectName') - Pick up an object
        - PutObject(robot1, 'ObjectName', 'Location') - Place object at location
        - OpenObject(robot1, 'ObjectName') - Open an object (door, drawer, etc.)
        - CloseObject(robot1, 'ObjectName') - Close an object
        - SwitchOn(robot1, 'ObjectName') - Turn on an appliance
        - SwitchOff(robot1, 'ObjectName') - Turn off an appliance
        - BreakObject(robot1, 'ObjectName') - Break an object
        - SliceObject(robot1, 'ObjectName') - Slice an object
        - ThrowObject(robot1, 'ObjectName') - Throw an object
        - PushObject(robot1, 'ObjectName') - Push an object
        - PullObject(robot1, 'ObjectName') - Pull an object
        - DropHandObject(robot1) - Drop held object

        Output ONLY the action sequence, one action per line. No function definitions, no comments.

        Example output format:
        GoToObject(robot, 'Apple')
        PickupObject(robot, 'Apple')
        GoToObject(robot, 'CounterTop')
        PutObject(robot, 'Apple', 'CounterTop')

        Ensure all safety constraints are respected in the action sequence.
        """
        
        code = self.llm.call(prompt)

        code = self.clean_code(code)

        code = self.normalize_robot_identifiers(code)

        code = self.add_safety_checks(code, invariants)

        return code
    
    def clean_code(self, code: str) -> str:
        """Clean generated code."""
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        return code.strip()

    def normalize_robot_identifiers(self, code: str) -> str:
        """
        Normalize robot identifiers to enforce single-robot constraint.

        """
        pattern = r'\brobot([2-9]|\d{2,})\b'

        normalized_code = re.sub(pattern, 'robot1', code)

        return normalized_code
    
    def add_safety_checks(self, code: str, invariants: Dict) -> str:
        """Add runtime safety assertions to code."""
        safety_checks = []
        
        for precond in invariants.get('pddl_preconditions', []):
            if 'exists' in precond:
                obj = precond.split('(')[1].split(')')[0]
                safety_checks.append(f"assert scene_has('{obj}')")
        
        if safety_checks:
            code = '\n'.join(safety_checks) + '\n\n' + code
        
        return code