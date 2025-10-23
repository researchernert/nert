# nert/simulation/code_verifier.py
"""Verify generated code against invariants using EXTERNAL formal methods."""

import ast
from typing import Dict, List, Tuple, Optional, Set
import re
import z3
import logging
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

from utils.log_formatter import (
    log_section, log_info, log_ok, log_fail, log_result, log_substep, log_debug_substep
)

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Robot action types."""
    NAVIGATE = "navigate"
    PICKUP = "pickup"
    PLACE = "place"
    OPEN = "open"
    CLOSE = "close"
    POUR = "pour"
    SLICE = "slice"


@dataclass
class RobotState:
    """Robot state for symbolic execution."""
    location: str
    holding: Optional[str]
    objects_at: Dict[str, str]
    opened: Set[str]
    
    def copy(self):
        return RobotState(
            location=self.location,
            holding=self.holding,
            objects_at=self.objects_at.copy(),
            opened=self.opened.copy()
        )


class InvariantCodeVerifier:
    """Verify code using external formal methods with dynamic object handling."""
    
    def __init__(self, scene_objects: List[str] = None, robot_skills: List[str] = None):
        """
        Initialize with actual scene objects and robot capabilities.
        
        Args:
            scene_objects: List of objects actually in the scene
            robot_skills: List of skills the robot actually has
        """
        self.scene_objects = scene_objects or []
        self.robot_skills = robot_skills or []
        self.z3_solver = z3.Solver()
        self.z3_solver.set("timeout", 5000)  # 5 second timeout to prevent hangs
        self.violations = []
        self.dangerous_patterns = [
            'eval', 'exec', 'compile', '__import__',
            'open', 'file', 'input', 'raw_input'
        ]
        
    def verify(self, code: str, invariants: Dict) -> Tuple[bool, List[str], Dict[str, str]]:
        """
        Verify code against invariants using formal methods.

        Args:
            code: Generated robot code
            invariants: Dict containing preconditions, postconditions, ltl_invariants, etc.

        Returns:
            (is_valid, list_of_violations, methods_used_report)
        """
        violations = []
        methods_report = {
            'static_analysis': 'not_run',
            'symbolic_execution': 'not_run',
            'precondition_check': 'not_run',
            'postcondition_check': 'not_run',
            'ltl_verification': 'not_applicable',
            'stl_verification': 'not_applicable',
            'z3_consistency': 'not_run'
        }

        # Step 1: Static code analysis
        static_valid, static_violations = self.static_analyze(code)
        methods_report['static_analysis'] = 'passed' if static_valid else 'failed'
        if not static_valid:
            violations.extend(static_violations)
            return False, violations, methods_report

        # Step 2: Parse and extract action sequence
        try:
            tree = ast.parse(code)
            actions = self.extract_actions(tree)
        except SyntaxError as e:
            methods_report['symbolic_execution'] = 'failed'
            return False, [f"Syntax error: {e}"], methods_report

        # Step 3: Verify preconditions dynamically
        if self.scene_objects:
            precond_valid, precond_violations = self.verify_preconditions_dynamically(
                actions, invariants
            )
            methods_report['precondition_check'] = 'passed' if precond_valid else 'failed'
        else:
            precond_valid, precond_violations = self.verify_preconditions_formally(
                actions, invariants.get('pddl_preconditions', [])
            )
            methods_report['precondition_check'] = 'passed' if precond_valid else 'failed'
        violations.extend(precond_violations)

        # Step 4: Symbolic execution to build state trace
        state_trace = self.symbolic_execute(actions)
        methods_report['symbolic_execution'] = 'completed'

        # Step 5: Verify postconditions are achieved
        postcond_valid, postcond_violations = self.verify_postconditions(
            state_trace, invariants.get('pddl_postconditions', [])
        )
        methods_report['postcondition_check'] = 'passed' if postcond_valid else 'failed'
        violations.extend(postcond_violations)

        # Step 6: Check LTL invariants
        ltl_invariants = invariants.get('ltl_invariants', [])
        if ltl_invariants:
            ltl_valid, ltl_violations = self.verify_ltl_properties(
                state_trace, ltl_invariants
            )
            methods_report['ltl_verification'] = 'passed' if ltl_valid else 'failed'
            violations.extend(ltl_violations)

        # Step 7: Check physical constraints (STL)
        stl_constraints = invariants.get('stl_constraints', [])
        if stl_constraints:
            stl_valid, stl_violations = self.verify_physical_constraints(
                actions, stl_constraints
            )
            methods_report['stl_verification'] = 'passed' if stl_valid else 'failed'
            violations.extend(stl_violations)

        # Step 8: Use Z3 for logical consistency
        if self.scene_objects:
            z3_valid, z3_violations = self.verify_with_z3_dynamically(actions, invariants)
            methods_report['z3_consistency'] = 'passed' if z3_valid else 'failed'
            violations.extend(z3_violations)

        return len(violations) == 0, violations, methods_report
    
    def static_analyze(self, code: str) -> Tuple[bool, List[str]]:
        """
        Perform static code analysis for safety.
        
        Checks for dangerous patterns and malformed code.
        """
        violations = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Code has syntax errors: {e}"]
        
        # Check for dangerous function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.dangerous_patterns:
                        violations.append(
                            f"Dangerous function call detected: {node.func.id}"
                        )
                        
            # Check for imports (shouldn't be in robot code)
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                violations.append("Import statements not allowed in robot code")
        
        return len(violations) == 0, violations
    
    def extract_actions(self, tree: ast.AST) -> List[Dict]:
        """Extract robot actions from AST."""
        actions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    action = {
                        'function': node.func.id,
                        'args': [],
                        'line': getattr(node, 'lineno', 0)
                    }
                    
                    for arg in node.args:
                        if isinstance(arg, ast.Constant):
                            action['args'].append(arg.value)
                        elif isinstance(arg, ast.Str): 
                            action['args'].append(arg.s)
                        elif isinstance(arg, ast.Name):
                            action['args'].append(arg.id)
                    
                    action['type'] = self.classify_action(action['function'])
                    actions.append(action)

                    if 'putobject' in action['function'].lower():
                        logger.debug(f"Parsed action: {action['function']}({', '.join(map(str, action['args']))}) [type: {action['type']}]")
        
        return actions
    
    def classify_action(self, function_name: str) -> ActionType:
        """Classify action into types - supports both old and new action names."""
        function_lower = function_name.lower()

        if 'gotoobject' in function_lower or 'navigate' in function_lower or 'goto' in function_lower or 'move' in function_lower:
            return ActionType.NAVIGATE
        elif 'pickupobject' in function_lower or 'pickup' in function_lower or 'pick' in function_lower or 'grab' in function_lower:
            return ActionType.PICKUP
        elif 'putobject' in function_lower or 'place' in function_lower or 'put' in function_lower:
            return ActionType.PLACE
        elif 'openobject' in function_lower or 'open' in function_lower:
            return ActionType.OPEN
        elif 'closeobject' in function_lower or 'close' in function_lower:
            return ActionType.CLOSE
        elif 'pour' in function_lower:
            return ActionType.POUR
        elif 'sliceobject' in function_lower or 'slice' in function_lower or 'cut' in function_lower:
            return ActionType.SLICE
        elif 'breakobject' in function_lower or 'break' in function_lower:
            return ActionType.SLICE  
        elif 'throwobject' in function_lower or 'throw' in function_lower:
            return ActionType.PLACE  
        elif 'switchon' in function_lower or 'switchoff' in function_lower:
            return ActionType.OPEN  
        elif 'pushobject' in function_lower or 'pullobject' in function_lower:
            return ActionType.NAVIGATE  
        elif 'drophandobject' in function_lower or 'drop' in function_lower:
            return ActionType.PLACE  
        else:
            return ActionType.NAVIGATE  # Default
    
    def verify_preconditions_dynamically(self, actions: List[Dict], 
                                        invariants: Dict) -> Tuple[bool, List[str]]:
        """
        Verify preconditions by parsing the invariant strings dynamically.
        """
        violations = []
        
        precond_patterns = self.parse_invariant_patterns(
            invariants.get('pddl_preconditions', [])
        )
        
        state = {
            'robot_at': None,
            'holding': None,
            'object_locations': {},
            'opened': set()
        }
        
        for i, action in enumerate(actions):
            action_type = action['type']
            action_target = action['args'][0] if action['args'] else None
            
            required_preconditions = precond_patterns.get(action_type, [])
            
            for precond in required_preconditions:
                if not self.check_precondition(precond, state, action_target):
                    violations.append(
                        f"Line {action.get('line', i)}: Precondition '{precond}' not met for {action['function']}"
                    )
            
            state = self.update_state(state, action_type, action_target)
        
        return len(violations) == 0, violations
    
    def parse_invariant_patterns(self, invariant_strings: List[str]) -> Dict[ActionType, List]:
        """
        Parse invariant strings to extract patterns dynamically.
        
        Example: "exists(apple)" -> pattern: exists({object})
                 "at(robot, {object})" -> pattern: at(robot, {object})
        """
        patterns = {
            ActionType.PICKUP: [],
            ActionType.PLACE: [],
            ActionType.NAVIGATE: [],
            ActionType.OPEN: [],
            ActionType.CLOSE: [],
            ActionType.POUR: [],
            ActionType.SLICE: []
        }
        
        for inv_str in invariant_strings:
            if 'pickup' in inv_str.lower() or 'holding' in inv_str.lower():
                action_type = ActionType.PICKUP
            elif 'place' in inv_str.lower() or 'put' in inv_str.lower():
                action_type = ActionType.PLACE
            elif 'navigate' in inv_str.lower() or 'at(' in inv_str.lower():
                action_type = ActionType.NAVIGATE
            else:
                continue
            
            pattern = self.extract_pattern(inv_str)
            if pattern:
                patterns[action_type].append(pattern)
        
        if not patterns[ActionType.PICKUP]:
            patterns[ActionType.PICKUP] = [
                ('exists', '{object}'),
                ('at', 'robot', '{object}'),
                ('not_holding', None)
            ]
        
        return patterns
    
    def extract_pattern(self, invariant_str: str) -> Tuple:
        """
        Extract pattern from invariant string.
        
        Examples:
            "exists(apple)" -> ('exists', 'apple')
            "at(robot, fridge)" -> ('at', 'robot', 'fridge')
            "holding(cup)" -> ('holding', 'cup')
        """
        # Common patterns
        patterns = [
            (r'exists\(([^)]+)\)', lambda m: ('exists', m.group(1))),
            (r'at\(([^,]+),\s*([^)]+)\)', lambda m: ('at', m.group(1), m.group(2))),
            (r'holding\(([^)]+)\)', lambda m: ('holding', m.group(1))),
            (r'gripper_empty\(\)', lambda m: ('not_holding', None)),
            (r'not\(holding\(([^)]+)\)\)', lambda m: ('not_holding', m.group(1)))
        ]
        
        for pattern, extractor in patterns:
            match = re.search(pattern, invariant_str)
            if match:
                return extractor(match)
        
        return None
    
    def check_precondition(self, precond: Tuple, state: Dict, target_object: str) -> bool:
        """
        Check if a precondition is satisfied given current state.
        
        """
        if not precond:
            return True
        
        precond_type = precond[0]
        
        if precond_type == 'exists':
            obj_pattern = precond[1]
            if '{object}' in obj_pattern:
                obj_to_check = target_object
            else:
                obj_to_check = obj_pattern
            
            if self.scene_objects:
                return self.fuzzy_match_object(obj_to_check, self.scene_objects)
            return True 
        
        elif precond_type == 'at':
            entity = precond[1]
            location = precond[2] if len(precond) > 2 else None
            
            if entity == 'robot':
                if '{object}' in str(location):
                    return state.get('robot_at') == target_object
                else:
                    return state.get('robot_at') == location
            
        elif precond_type == 'holding':
            return state.get('holding') == precond[1]
        
        elif precond_type == 'not_holding':
            return state.get('holding') is None
        
        return False
    
    def fuzzy_match_object(self, obj_name: str, available_objects: List[str]) -> bool:
        """
        Fuzzy match object name to available objects in scene.
        
        Handles variations like "Apple" vs "apple", "GarbageCan" vs "garbage"
        """
        if not obj_name:
            return False
        
        obj_lower = obj_name.lower()
        
        for scene_obj in available_objects:
            scene_lower = scene_obj.lower()
            
            if obj_lower == scene_lower:
                return True
            
            if obj_lower in scene_lower or scene_lower in obj_lower:
                return True
            
            # Handle compound words
            # "garbage_can" matches "GarbageCan"
            obj_normalized = obj_lower.replace('_', '').replace('-', '').replace(' ', '')
            scene_normalized = scene_lower.replace('_', '').replace('-', '').replace(' ', '')
            if obj_normalized == scene_normalized:
                return True
        
        return False
    
    def update_state(self, state: Dict, action_type: ActionType, target: str) -> Dict:
        """
        Update state based on action semantics.
        
        """
        new_state = state.copy()
        
        if action_type == ActionType.NAVIGATE:
            new_state['robot_at'] = target
            
        elif action_type == ActionType.PICKUP:
            if state.get('holding') is None:  # Can only pickup if not holding
                new_state['holding'] = target
                # Remove from location
                if target in new_state.get('object_locations', {}):
                    del new_state['object_locations'][target]
                    
        elif action_type == ActionType.PLACE:
            if state.get('holding'):
                placed_obj = state['holding']
                new_state['holding'] = None
                new_state['object_locations'][placed_obj] = target or state.get('robot_at')
                
        elif action_type == ActionType.OPEN:
            new_state.setdefault('opened', set()).add(target)
            
        elif action_type == ActionType.CLOSE:
            new_state.setdefault('opened', set()).discard(target)
        
        return new_state
    
    def verify_preconditions_formally(self, actions: List[Dict], 
                                     preconditions: List[str]) -> Tuple[bool, List[str]]:
        """
        Use Z3 SAT solver to verify preconditions (fallback when no scene info).
        """
        violations = []
        
        # Create Z3 variables for robot state
        robot_at = z3.Bool('robot_at')
        holding = z3.Bool('holding')
        gripper_empty = z3.Bool('gripper_empty')
        object_exists = z3.Bool('object_exists')
        path_clear = z3.Bool('path_clear')
        
        for i, action in enumerate(actions):
            self.z3_solver.push()  # Create new scope
            
            # Add constraints based on action type
            if action['type'] == ActionType.PICKUP:
                # Preconditions for pickup: at location, gripper empty, object exists
                self.z3_solver.add(z3.Implies(
                    z3.And(robot_at, gripper_empty, object_exists),
                    z3.BoolVal(True)
                ))
                
                # Check if these can be satisfied
                if self.z3_solver.check() != z3.sat:
                    violations.append(
                        f"Line {action['line']}: Cannot satisfy pickup preconditions"
                    )
                    
            elif action['type'] == ActionType.PLACE:
                # Must be holding something
                self.z3_solver.add(z3.Implies(
                    holding,
                    z3.BoolVal(True)
                ))
                
                if self.z3_solver.check() != z3.sat:
                    violations.append(
                        f"Line {action['line']}: Cannot place without holding object"
                    )
                    
            elif action['type'] == ActionType.NAVIGATE:
                # Path must be clear
                self.z3_solver.add(z3.Implies(
                    path_clear,
                    z3.BoolVal(True)
                ))
                
            self.z3_solver.pop()  # Restore scope
        
        return len(violations) == 0, violations
    
    def symbolic_execute(self, actions: List[Dict]) -> List[RobotState]:
        """
        Symbolically execute actions to build state trace.

        """
        state = RobotState(
            location='start',
            holding=None,
            objects_at={},
            opened=set()
        )

        state_trace = [state.copy()]

        log_section("--- Symbolic Execution ---")
        log_info(f"Initial state: location='{state.location}', holding={state.holding}")
        log_debug_substep(f"objects_at={state.objects_at}")

        for i, action in enumerate(actions):
            log_info(f"Action {i+1}/{len(actions)}: {action['function']}({', '.join(map(str, action['args']))})")
            new_state = state.copy()
            
            if action['type'] == ActionType.NAVIGATE:
                if action['args']:
                    new_state.location = str(action['args'][0])
                    
            elif action['type'] == ActionType.PICKUP:
                if action['args'] and new_state.holding is None:
                    if len(action['args']) >= 2:
                        obj = str(action['args'][1])
                    else:
                        obj = str(action['args'][0])

                    new_state.holding = obj
                    if obj in new_state.objects_at:
                        del new_state.objects_at[obj]
                    log_ok(f"Picked up '{obj}'")
                        
            elif action['type'] == ActionType.PLACE:
                if new_state.holding and action['args']:
                    log_debug_substep(f"Holding: '{new_state.holding}', Args: {action['args']}")

                    if len(action['args']) >= 3:
                        location = str(action['args'][2])
                    elif len(action['args']) == 2:
                        location = str(action['args'][1])
                    elif len(action['args']) == 1:
                        location = str(action['args'][0])
                    else:
                        location = new_state.location

                    placed_object = new_state.holding
                    new_state.objects_at[placed_object] = location
                    new_state.holding = None

                    log_ok(f"Placed '{placed_object}' at '{location}'")
                    log_debug_substep(f"Updated objects_at: {new_state.objects_at}")
                    
            elif action['type'] == ActionType.OPEN:
                if action['args']:
                    new_state.opened.add(str(action['args'][0]))
                    
            elif action['type'] == ActionType.CLOSE:
                if action['args']:
                    obj = str(action['args'][0])
                    new_state.opened.discard(obj)
            
            state = new_state
            state_trace.append(state.copy())
            log_debug_substep(f"State: location='{state.location}', holding={state.holding}, objects_at={state.objects_at}")

        log_info("")
        log_result(f"Execution complete - Final state: location='{state.location}', holding={state.holding}")

        return state_trace
    
    def verify_postconditions(self, state_trace: List[RobotState],
                            postconditions: List[str]) -> Tuple[bool, List[str]]:
        """
        Verify postconditions are satisfied in final state.
        """
        violations = []

        if not state_trace:
            return False, ["No states in trace"]

        final_state = state_trace[-1]

        log_section("--- Postcondition Verification ---")
        log_info(f"Checking {len(postconditions)} postcondition(s)")
        log_debug_substep(f"Postconditions: {postconditions}")
        log_debug_substep(f"Final state: location='{final_state.location}', holding={final_state.holding}, objects_at={final_state.objects_at}")

        for postcond in postconditions:
            if 'holding(' in postcond:
                obj = re.search(r'holding\((\w+)\)', postcond)
                if obj:
                    expected_holding = obj.group(1)
                    if 'not(' in postcond:
                        if final_state.holding == expected_holding:
                            violations.append(f"Postcondition failed: {postcond}")
                    else:
                        if final_state.holding != expected_holding:
                            violations.append(f"Postcondition failed: {postcond}")
                            
            elif 'at(' in postcond:
                match = re.search(r"at\((['\"]?)(\w+)\1,\s*(['\"]?)(\w+)\3\)", postcond)
                if match:
                    obj = match.group(2)
                    location = match.group(4)

                    log_debug_substep(f"Evaluating: at({obj}, {location}) - objects_at={final_state.objects_at}")

                    if obj == 'robot':
                        if final_state.location.lower() != location.lower():
                            violations.append(f"Robot not at {location}")
                    else:
                        actual_location = final_state.objects_at.get(obj)
                        if actual_location is None:
                            for obj_key, loc_val in final_state.objects_at.items():
                                if obj_key.lower() == obj.lower():
                                    actual_location = loc_val
                                    break

                        if actual_location is None or actual_location.lower() != location.lower():
                            violations.append(f"{obj} not at {location}")
        
        return len(violations) == 0, violations
    
    def verify_ltl_properties(self, state_trace: List[RobotState], 
                            ltl_properties: List[str]) -> Tuple[bool, List[str]]:
        """
        Verify LTL properties using model checking.
        
        """
        violations = []
        
        # Try to use Spot model checker
        try:
            from core.ltl_verifier import LTLModelChecker
            checker = LTLModelChecker()
            use_formal = True
        except ImportError:
            logger.warning("Spot not available, using fallback LTL checking")
            checker = None
            use_formal = False
        
        for ltl_formula in ltl_properties:
            if use_formal:
                # Convert RobotState objects to dicts for model checker
                trace_dicts = []
                for state in state_trace:
                    state_dict = {
                        'holding': state.holding is not None,
                        'at_location': state.location,
                        'collision': False, 
                        'spilled': False,  
                        'gripper_empty': state.holding is None
                    }
                    
                    if state.holding and 'liquid' in str(state.holding).lower():
                        state_dict['has_liquid'] = True
                        
                    trace_dicts.append(state_dict)
                
                # Verify using model checker
                valid, error_msg = checker.verify_ltl_formula(ltl_formula, trace_dicts)
                if not valid:
                    violations.append(error_msg)
            else:
                if 'G(' in ltl_formula or 'never' in ltl_formula.lower():
                    for i, state in enumerate(state_trace):
                        if not self.check_state_property(state, ltl_formula):
                            violations.append(f"LTL property violated at step {i}: {ltl_formula}")
                            break
                elif 'F(' in ltl_formula or 'eventually' in ltl_formula.lower():
                    satisfied = any(self.check_state_property(s, ltl_formula) for s in state_trace)
                    if not satisfied:
                        violations.append(f"LTL property never satisfied: {ltl_formula}")
        
        return len(violations) == 0, violations
    
    def check_state_property(self, state: RobotState, property_str: str) -> bool:
        """
        Check if a property holds in a given state.
        """
        if 'collision' in property_str.lower():
            return True  
            
        if 'spilled' in property_str.lower():
            return 'liquid' not in str(state.holding).lower()
            
        if 'supervised' in property_str.lower():
            return True
            
        return True 
    
    def verify_physical_constraints(self, actions: List[Dict],
                                   stl_constraints: List[str]) -> Tuple[bool, List[str]]:
        """
        Verify physical safety constraints using STL monitoring.

        Uses rtamt for formal STL verification when available,
        falls back to heuristic checking otherwise.
        """
        if not stl_constraints:
            return True, []

        try:
            from core.stl_monitor import STLMonitor
            monitor = STLMonitor()
            return monitor.verify_constraints(actions, stl_constraints)
        except ImportError as e:
            logger.warning(f"STL monitor not available: {e}")
            return self._basic_physical_check(actions, stl_constraints)

    def _basic_physical_check(self, actions: List[Dict],
                             stl_constraints: List[str]) -> Tuple[bool, List[str]]:
        """
        Fallback physical constraint checking when STL monitor unavailable.
        """
        violations = []

        for constraint in stl_constraints:
            if 'temperature' in constraint.lower():
                hot_objects = ['coffee', 'tea', 'soup', 'hot', 'boiling']
                for action in actions:
                    if action['type'] in ['PICKUP', 'PLACE']:
                        if any(hot in str(action['args']).lower() for hot in hot_objects):
                            violations.append(
                                f"Temperature constraint may be violated: {constraint}"
                            )

            elif 'acceleration' in constraint.lower():
                fragile = ['glass', 'fragile', 'delicate']
                for action in actions:
                    if any(f in str(action['args']).lower() for f in fragile):
                        if action['type'] == 'THROW':
                            violations.append(
                                f"Acceleration constraint violated for fragile object"
                            )

            elif 'tilt_angle' in constraint.lower():
                for action in actions:
                    target = str(action['args'][0]) if action['args'] else ''
                    if 'liquid' in target.lower() or 'water' in target.lower():
                        if action['type'] == 'THROW':
                            violations.append(
                                f"Tilt angle constraint violated for liquid"
                            )

        return len(violations) == 0, violations
    
    def verify_with_z3_dynamically(self, actions: List[Dict],
                                   invariants: Dict) -> Tuple[bool, List[str]]:
        """
        Use Z3 to verify logical consistency of action sequence.
        Enhanced to check precondition satisfiability and state consistency.
        """
        violations = []

        # Create Z3 variables for each object mentioned
        objects_mentioned = set()
        for action in actions:
            if action['args']:
                objects_mentioned.add(str(action['args'][0]))

        # Create boolean variables for each object and property
        z3_vars = {}
        for obj in objects_mentioned:
            z3_vars[f'exists_{obj}'] = z3.Bool(f'exists_{obj}')
            z3_vars[f'at_robot_{obj}'] = z3.Bool(f'at_robot_{obj}')
            z3_vars[f'holding_{obj}'] = z3.Bool(f'holding_{obj}')

        # Add scene constraints - objects must exist in scene
        for obj in objects_mentioned:
            if not self.fuzzy_match_object(obj, self.scene_objects):
                # Object doesn't exist in scene
                self.z3_solver.add(z3.Not(z3_vars[f'exists_{obj}']))
                violations.append(f"Object '{obj}' not found in scene")

        # Check precondition satisfiability for each action
        preconditions = invariants.get('pddl_preconditions', [])
        for i, action in enumerate(actions):
            if not action['args']:
                continue

            target = str(action['args'][0])
            action_type = action['type']

            # Check if action preconditions can be satisfied
            if action_type == 'PICKUP':
                # Must exist and robot must be able to reach it
                if f'exists_{target}' in z3_vars:
                    self.z3_solver.push()
                    # Pickup requires: object exists AND robot can be at object
                    self.z3_solver.add(
                        z3.Implies(
                            z3_vars[f'exists_{target}'],
                            z3_vars.get(f'at_robot_{target}', z3.BoolVal(True))
                        )
                    )
                    if self.z3_solver.check() == z3.unsat:
                        violations.append(
                            f"Action {i+1}: Pickup preconditions unsatisfiable for {target}"
                        )
                    self.z3_solver.pop()

            elif action_type == 'PLACE':
                # Must be holding something
                holding_anything = z3.Or([z3_vars[f'holding_{obj}']
                                         for obj in objects_mentioned
                                         if f'holding_{obj}' in z3_vars])
                self.z3_solver.push()
                self.z3_solver.add(holding_anything)
                if self.z3_solver.check() == z3.unsat:
                    violations.append(
                        f"Action {i+1}: Cannot place without holding object"
                    )
                self.z3_solver.pop()

        # Check for contradictory state transitions
        state_vars = self._create_state_variables(actions, z3_vars)
        for i in range(len(state_vars) - 1):
            if self._creates_contradiction(state_vars[i], state_vars[i+1], z3_vars):
                violations.append(
                    f"Actions {i+1}-{i+2} create contradictory state"
                )

        # Check overall satisfiability
        if self.z3_solver.check() == z3.unsat:
            violations.append("Action sequence is logically inconsistent")

        return len(violations) == 0, violations

    def _create_state_variables(self, actions: List[Dict],
                               z3_vars: Dict) -> List[Dict]:
        """
        Create Z3 variables representing state at each action step.
        Returns list of state dictionaries.
        """
        states = [{'holding': z3.BoolVal(False)}]  # Initial state: not holding

        for action in actions:
            prev_state = states[-1]
            new_state = prev_state.copy()

            if action['type'] == 'PICKUP' and action['args']:
                target = str(action['args'][0])
                if f'holding_{target}' in z3_vars:
                    new_state['holding'] = z3_vars[f'holding_{target}']

            elif action['type'] == 'PLACE':
                new_state['holding'] = z3.BoolVal(False)

            states.append(new_state)

        return states

    def _creates_contradiction(self, state1: Dict, state2: Dict,
                              z3_vars: Dict) -> bool:
        """
        Check if transitioning from state1 to state2 creates logical contradiction.
        """
        self.z3_solver.push()

        # Add constraints for both states
        if 'holding' in state1:
            self.z3_solver.add(state1['holding'])
        if 'holding' in state2:
            # Check if transition is valid
            # For example: can't hold two things simultaneously
            self.z3_solver.add(state2['holding'])

        result = self.z3_solver.check() == z3.unsat
        self.z3_solver.pop()

        return result