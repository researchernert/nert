# nert/core/invariant_generator.py
"""Hybrid invariant generation combining formal methods and LLM."""

from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass
import yaml
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class Invariants:
    """Container for all types of invariants."""
    pddl_preconditions: List[str]
    pddl_postconditions: List[str]
    ltl_invariants: List[str]
    stl_constraints: List[str]
    llm_contextual: str  


class HybridInvariantGenerator:
    """
    Generate safety specifications using PDDL-inspired notation.

    This generator creates PDDL-style predicates (preconditions, postconditions)
    that constrain code generation and serve as verification targets.

    Note: This produces PDDL-style specifications for constraint expression,
    not complete PDDL domain files for automated planning. The specifications
    guide the LLM's code generation and provide formal targets for post-hoc
    verification of generated code.

    The specifications combine:
    - PDDL-style preconditions/postconditions (first-order logic predicates)
    - Linear Temporal Logic (LTL) properties for temporal constraints
    - Signal Temporal Logic (STL) for physical safety bounds
    - LLM-generated contextual safety considerations
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.load_templates()
    
    def load_templates(self):
        """Load PDDL and LTL templates."""
        current_dir = Path(__file__).parent
        template_path = current_dir / 'invariant_templates.yaml'
        
        try:
            with open(template_path, 'r') as f:
                self.templates = yaml.safe_load(f)
                logger.info(f"Loaded templates from {template_path}")
        except FileNotFoundError:
            logger.warning(f"Template file not found at {template_path}, using defaults")
            self.templates = self.get_default_templates()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML: {e}")
            self.templates = self.get_default_templates()
    
    def get_default_templates(self):
        """Return default templates if file not found."""
        return {
            'pddl': {
                'pickup': {
                    'preconditions': ['exists({object})', 'at(robot, {object})', 'gripper_empty()'],
                    'postconditions': ['holding({object})', 'not(at({object}, location))']
                },
                'place': {
                    'preconditions': ['holding({object})', 'at(robot, {destination})'],
                    'postconditions': ['at({object}, {destination})', 'not(holding({object}))']
                },
                'navigate': {
                    'preconditions': ['path_clear(robot, {destination})'],
                    'postconditions': ['at(robot, {destination})']
                }
            },
            'ltl_templates': {
                'safety': ['G(¬collision)', 'G(¬spilled)'],
                'liveness': ['F(task_complete)', 'F(at_goal)'],
                'fairness': ['G(request → F(response))']
            }
        }
    
    def parse_task_components(self, task: str, scene_objects: List[str] = None) -> Dict:
        """
        Parses task using LLM to identify ALL actions, objects, and destinations.

        Using LLM-based extraction allows for handling
        complex multi-object tasks.
        """
        prompt = f"""
Extract ALL components from this robot task.

Task: "{task}"

Available scene objects: {scene_objects if scene_objects else 'Not specified'}

Identify:
1. ALL objects to be manipulated (e.g., apple, plate, knife, cup, bread)
2. ALL destination locations (e.g., fridge, countertop, drawer, cabinet)
3. ALL actions required (e.g., pickup, place, navigate, open, close)

Rules:
- Extract every object mentioned, not just the first one
- List actions in order they appear in the task

Return ONLY valid JSON in this exact format:
{{{{
    "objects": ["Object1", "Object2", ...],
    "destinations": ["Dest1", "Dest2", ...],
    "actions": ["action1", "action2", ...]
}}}}

Do not include any explanation, just the JSON.
"""

        try:
            response = self.llm.call(prompt, max_tokens=500)

            if "GEMINI_SAFETY_BLOCK:" in response or "GEMINI_RECITATION_BLOCK:" in response:
                logger.warning("Gemini blocked response - using fallback")
                return self._fallback_keyword_parse(task)

            response_clean = response.strip()
            if "```json" in response_clean:
                response_clean = response_clean.split("```json")[1].split("```")[0]
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0]

            components = json.loads(response_clean.strip())

            if not isinstance(components.get('objects'), list):
                components['objects'] = []
            if not isinstance(components.get('destinations'), list):
                components['destinations'] = []
            if not isinstance(components.get('actions'), list):
                components['actions'] = []

            if scene_objects:
                validated_objects = []
                for obj in components['objects']:
                    for scene_obj in scene_objects:
                        if (obj.lower() in scene_obj.lower() or
                            scene_obj.lower() in obj.lower() or
                            obj.lower().replace(' ', '') == scene_obj.lower().replace(' ', '')):
                            validated_objects.append(scene_obj)
                            break
                    else:
                        validated_objects.append(obj)

                components['objects'] = validated_objects

            logger.info(f"LLM extracted {len(components['objects'])} objects, "
                       f"{len(components['destinations'])} destinations")

            return components

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Response was: {response[:200]}")
            return self._fallback_keyword_parse(task)
        except Exception as e:
            logger.warning(f"LLM component extraction failed: {e}")
            return self._fallback_keyword_parse(task)

    def _fallback_keyword_parse(self, task: str) -> Dict:
        """
        Fallback to keyword-based parsing if LLM extraction fails.
        This is the original implementation kept as backup.
        """
        components = {
            'actions': [],
            'objects': [],
            'destinations': []
        }

        task_lower = task.lower()

        action_keywords = ['pickup', 'pick up', 'place', 'put', 'navigate', 'go to', 'move',
                           'open', 'close', 'slice', 'pour']
        for keyword in action_keywords:
            if keyword in task_lower:
                components['actions'].append(keyword.replace(' ', '_'))

        object_keywords = ['apple', 'cup', 'knife', 'book', 'plate', 'bottle', 'bread',
                           'bowl', 'pan', 'pot', 'fork', 'spoon']
        for obj in object_keywords:
            if obj in task_lower:
                components['objects'].append(obj)

        destination_keywords = ['fridge', 'table', 'chair', 'shelf', 'counter', 'countertop',
                               'floor', 'sink', 'drawer', 'cabinet', 'toaster', 'microwave']
        for dest in destination_keywords:
            if dest in task_lower:
                components['destinations'].append(dest)

        logger.info(f"Fallback extraction: {len(components['objects'])} objects, "
                   f"{len(components['destinations'])} destinations")

        return components
    
    def extract_pddl_conditions(self, task: str, scene_objects: Optional[List[str]] = None,
                                 robot_skills: Optional[List[str]] = None) -> Dict:
        """
        Generate PDDL-style pre/postconditions for ALL object-destination pairs.

        Uses LLM to extract components, then applies templates for formal correctness.
        """
        components = self.parse_task_components(task, scene_objects)

        preconditions = []
        postconditions = []

        objects = components.get('objects', [])
        destinations = components.get('destinations', [])
        actions = components.get('actions', [])

        if not objects or not destinations:
            logger.warning("No objects or destinations extracted from task")
            return {'preconditions': [], 'postconditions': []}

        object_dest_pairs = []
        for i, obj in enumerate(objects):
            if i < len(destinations):
                dest = destinations[i]
            elif destinations:
                dest = destinations[-1]  
            else:
                continue
            object_dest_pairs.append((obj, dest))

        for obj, dest in object_dest_pairs:
            action_type = self._infer_action_type(actions, obj, dest)

            template = self.templates.get('pddl', {}).get(action_type, {})

            if template:
                for precond in template.get('preconditions', []):
                    instantiated = precond.replace('{object}', obj).replace('{destination}', dest)
                    if '{' not in instantiated:
                        preconditions.append(instantiated)
            else:
                preconditions.extend([
                    f"exists({obj})",
                    f"at(robot, {obj})",
                    "gripper_empty()"
                ])

            if template:
                for postcond in template.get('postconditions', []):
                    instantiated = postcond.replace('{object}', obj).replace('{destination}', dest)
                    if '{' not in instantiated:
                        postconditions.append(instantiated)
            else:
                postconditions.extend([
                    f"at({obj}, {dest})",
                    f"not(holding({obj}))"
                ])

        if objects:
            postconditions.append("gripper_empty()")

        logger.info(f"Generated PDDL conditions for {len(object_dest_pairs)} object-destination pairs")

        return {
            'preconditions': preconditions,
            'postconditions': postconditions
        }

    def _infer_action_type(self, actions: List[str], obj: str, dest: str) -> str:
        """
        Infer which PDDL template to use based on extracted actions.

        Returns: 'pickup', 'place', or 'navigate'
        """
        actions_lower = [a.lower() for a in actions]

        if any(a in actions_lower for a in ['place', 'put', 'drop']):
            return 'place'
        elif any(a in actions_lower for a in ['pickup', 'pick', 'grab']):
            return 'pickup'
        elif any(a in actions_lower for a in ['navigate', 'goto', 'go', 'move']):
            return 'navigate'

        return 'place'
    
    def generate_ltl_properties(self, task: str, scene_objects: Optional[List[str]] = None,
                                 robot_skills: Optional[List[str]] = None) -> List[str]:
        """
        Generate LTL formulas informed by both task keywords and LLM semantic analysis.

        Combines template-based safety properties with context-aware properties
        extracted from LLM understanding of the task.
        """
        ltl_properties = []
        task_lower = task.lower()

        # ALWAYS include universal safety properties
        ltl_properties.append("G(!collision)")  # Never collide
        ltl_properties.append("G(!spilled)")    # Never spill

        # Use LLM to identify semantic concepts in the task
        semantic_concepts = self._extract_semantic_concepts(task)

        # Context-aware properties based on semantic analysis
        if 'dangerous' in semantic_concepts or 'hot' in task_lower or 'sharp' in task_lower:
            ltl_properties.append("G(dangerous_object -> !human_contact)")

        if 'vulnerable_person' in semantic_concepts or 'child' in task_lower or 'baby' in task_lower:
            ltl_properties.append("G(child_present -> supervised)")

        if 'fragile' in semantic_concepts or 'glass' in task_lower or 'delicate' in task_lower:
            ltl_properties.append("G(handling_fragile -> gentle_movement)")

        # Task completion properties
        if 'put' in task_lower or 'place' in task_lower:
            ltl_properties.append("F(object_at_destination)")

        if 'pickup' in task_lower or 'pick' in task_lower:
            ltl_properties.append("F(holding)")

        # Precedence properties
        if 'then' in task_lower or 'after' in task_lower:
            ltl_properties.append("!Y U X")  # Not Y until X

        # Scene-aware properties
        if scene_objects:
            dangerous_objects = ['Knife', 'Pot', 'Pan', 'StoveBurner', 'Microwave']
            if any(obj in scene_objects for obj in dangerous_objects):
                ltl_properties.append("G(dangerous_object -> !human_contact)")

        return ltl_properties

    def _extract_semantic_concepts(self, task: str) -> List[str]:
        """
        Use LLM to extract high-level semantic concepts from task.

        Returns concepts like: 'dangerous', 'vulnerable_person', 'fragile', 'liquid'
        """
        prompt = f"""
Analyze this robot task and identify safety-relevant concepts.

Task: "{task}"

Which of these concepts are present?
- dangerous: Task involves hot, sharp, or hazardous objects
- vulnerable_person: Task involves children, elderly, or people needing protection
- fragile: Task involves breakable or delicate objects
- liquid: Task involves liquids that could spill
- heavy: Task involves heavy objects that could cause injury
- electrical: Task involves electrical appliances

Return ONLY a comma-separated list of applicable concepts, or "none" if none apply.
Example: "dangerous, fragile" or "vulnerable_person, liquid" or "none"
"""

        try:
            response = self.llm.call(prompt, max_tokens=50)

            if "GEMINI_SAFETY_BLOCK:" in response or "GEMINI_RECITATION_BLOCK:" in response:
                return []

            concepts = [c.strip().lower() for c in response.split(',')]
            return [c for c in concepts if c != 'none']
        except Exception as e:
            logger.debug(f"Semantic concept extraction failed: {e}")
            return []
    
    def generate_stl_constraints(self, task: str, scene_objects: Optional[List[str]] = None,
                                  robot_skills: Optional[List[str]] = None) -> List[str]:
        """
        Generate STL constraints informed by LLM semantic analysis and scene context.

        Uses both keyword matching and LLM understanding to generate appropriate
        physical safety constraints.
        """
        stl_constraints = []
        task_lower = task.lower()

        semantic_concepts = self._extract_semantic_concepts(task)

        hot_objects = ['Pot', 'Pan', 'Kettle', 'Microwave', 'StoveBurner', 'Toaster', 'CoffeeMachine']
        fragile_objects = ['WineBottle', 'Bottle', 'Vase', 'Egg', 'Plate', 'Bowl', 'Cup', 'Mug']
        liquid_objects = ['Cup', 'Mug', 'Bowl', 'Pot', 'Pan', 'Bottle', 'WineBottle']
        sharp_objects = ['Knife', 'ButterKnife']

        has_hot = (scene_objects and any(obj in scene_objects for obj in hot_objects)) or \
                  'hot' in task_lower or 'boiling' in task_lower
        has_fragile = (scene_objects and any(obj in scene_objects for obj in fragile_objects)) or \
                      'fragile' in task_lower or 'glass' in task_lower
        has_liquid = (scene_objects and any(obj in scene_objects for obj in liquid_objects)) or \
                     'liquid' in task_lower or 'water' in task_lower or 'pour' in task_lower
        has_sharp = (scene_objects and any(obj in scene_objects for obj in sharp_objects)) or \
                    'knife' in task_lower or 'sharp' in task_lower

        if has_hot or 'dangerous' in semantic_concepts:
            stl_constraints.append("globally[0,10](temperature > 60 implies distance_to_human > 1.0)")

        if 'fast' not in task_lower and 'quick' not in task_lower:
            stl_constraints.append("globally[0,10](velocity < 1.0)")

        if has_fragile or 'fragile' in semantic_concepts:
            stl_constraints.append("globally[0,10](acceleration < 0.5)")

        if has_liquid or 'liquid' in semantic_concepts:
            stl_constraints.append("globally[0,10](tilt_angle < 15.0)")

        if has_sharp or 'dangerous' in semantic_concepts:
            stl_constraints.append("globally[0,10](distance_to_human > 0.5)")

        if has_hot and has_liquid:
            stl_constraints.append("globally[0,10]((temperature > 60 and tilt_angle < 10) or temperature <= 60)")

        if 'vulnerable_person' in semantic_concepts:
            stl_constraints.append("globally[0,10](distance_to_vulnerable > 2.0)")

        return stl_constraints
    
    def _get_scene_objects_from_cache(self, floor_plan: int) -> List[str]:
        """Get objects for a specific floor plan from scene cache."""
        try:
            cache_path = Path(__file__).parent.parent / 'data' / 'scene_cache.json'
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
                    scene_key = f"floorplan_{floor_plan}"
                    if scene_key in cache:
                        return cache[scene_key].get('objects', [])
        except Exception as e:
            logger.warning(f"Could not load scene cache: {e}")

        return ['Apple', 'Table', 'Fridge', 'GarbageCan', 'Cup', 'Plate', 'Bowl']

    def _get_default_robot_skills(self) -> List[str]:
        """Get default AI2-THOR robot skills."""
        return [
            'GoToObject', 'PickupObject', 'PutObject',
            'OpenObject', 'CloseObject', 'SliceObject',
            'BreakObject', 'SwitchOn', 'SwitchOff',
            'ThrowObject', 'PushObject', 'PullObject',
            'DropHandObject'
        ]

    def generate_llm_contextual(self, task: str, scene_objects: List[str] = None,
                                robot_skills: List[str] = None) -> str:
        """Use LLM to generate context-specific safety constraints using neurosymbolic prompt.

        Returns the full structured LLM response without parsing to preserve all information:
        - Goal
        - Initial Conditions
        - Invariants
        - Action Sequence with preconditions and postconditions
        """

        # Import the invariant generation prompt from prompts.py
        try:
            from core.prompts import INVARIANT_GENERATION_PROMPT
        except ImportError:
            logger.warning("Could not import INVARIANT_GENERATION_PROMPT, using fallback")
            return self._generate_fallback_response(task)

        if scene_objects is None:
            scene_objects = self.current_scene_objects if hasattr(self, 'current_scene_objects') else []
        if robot_skills is None:
            robot_skills = self.current_robot_skills if hasattr(self, 'current_robot_skills') else []

        prompt = INVARIANT_GENERATION_PROMPT.replace('{{task}}', task)
        prompt = prompt.replace('{{objects}}', ', '.join(scene_objects) if scene_objects else 'None available')
        prompt = prompt.replace('{{skills}}', ', '.join(robot_skills) if robot_skills else 'None available')

        try:
            response = self.llm.call(prompt)

            if "GEMINI_SAFETY_BLOCK:" in response or "GEMINI_RECITATION_BLOCK:" in response:
                logger.warning("Gemini blocked contextual generation - using fallback")
                return self._generate_fallback_response(task)

            return response

        except Exception as e:
            logger.error(f"LLM contextual generation failed: {e}")
            return self._generate_fallback_response(task)

    def _generate_fallback_response(self, task: str) -> str:
        """Generate a fallback structured response when LLM fails."""
        task_lower = task.lower()

        fallback = f"""## Task: {task}

### Goal:
Task completion with safety constraints satisfied

### Initial Conditions:
Robot is at starting position
All objects are in their initial locations
Gripper is empty

### Invariants:
* Robot must verify gripper is empty before picking up objects
* Only one object can be held at a time
* All actions must complete successfully before proceeding"""

        if 'knife' in task_lower or 'sharp' in task_lower:
            fallback += "\n* Never hold sharp objects while moving quickly"
            fallback += "\n* Maintain safe distance from humans when handling sharp objects"

        if 'hot' in task_lower or 'boiling' in task_lower:
            fallback += "\n* Hot objects must be handled with appropriate safety measures"

        if 'fragile' in task_lower or 'glass' in task_lower:
            fallback += "\n* Fragile objects require gentle handling with minimal acceleration"

        if 'liquid' in task_lower or 'water' in task_lower:
            fallback += "\n* Maintain upright orientation when carrying liquids"

        fallback += "\n\n### Action Sequence:\nNo specific actions generated (LLM unavailable)"

        return fallback

    def _generate_default_safety_constraints(self, task: str) -> List[str]:
        """Generate default safety constraints when LLM fails (legacy method for PDDL/LTL/STL)."""
        constraints = []
        task_lower = task.lower()

        if 'knife' in task_lower or 'sharp' in task_lower:
            constraints.append("Never hold sharp objects while moving quickly")
            constraints.append("Maintain safe distance from humans when handling sharp objects")

        if 'hot' in task_lower or 'boiling' in task_lower:
            constraints.append("Hot objects must be handled with appropriate safety measures")
            constraints.append("Never place hot objects near flammable materials")

        if 'fragile' in task_lower or 'glass' in task_lower:
            constraints.append("Fragile objects require gentle handling with minimal acceleration")
            constraints.append("Never stack heavy objects on fragile items")

        if 'liquid' in task_lower or 'water' in task_lower:
            constraints.append("Maintain upright orientation when carrying liquids")
            constraints.append("Keep liquids away from electronic devices")

        constraints.extend([
            "Robot must verify gripper is empty before picking up objects",
            "Only one object can be held at a time",
            "All actions must complete successfully before proceeding"
        ])

        return constraints
    
    def contradicts(self, pre: str, post: str) -> bool:
        """Check if precondition and postcondition contradict."""
        if 'not(' in post:
            negated = post.replace('not(', '').replace(')', '')
            if negated in pre:
                return True
        return False
    
    def resolve_conflicts(self, invariants: Invariants, conflicts: List[Tuple]) -> Invariants:
        """Resolve conflicts in invariants."""
        logger.warning(f"Found {len(conflicts)} conflicts, keeping formal constraints")
        return invariants
    
    def validate_consistency(self, invariants: Invariants) -> Invariants:
        """Check for conflicts between different types of invariants."""
        conflicts = []
        
        for pre in invariants.pddl_preconditions:
            for post in invariants.pddl_postconditions:
                if self.contradicts(pre, post):
                    conflicts.append((pre, post))
        
        if conflicts:
            logger.warning(f"Found {len(conflicts)} conflicts in invariants")
            invariants = self.resolve_conflicts(invariants, conflicts)
        
        return invariants
    
    def generate(self, task: str, confidence_score: float = None,
                 floor_plan: int = 1, scene_objects: Optional[List[str]] = None,
                 robot_skills: Optional[List[str]] = None) -> Invariants:
        """
        Generate complete set of invariants using hybrid LLM-informed template approach.

        Strategy:
        1. Use LLM to extract ALL task components (objects, destinations, actions)
        2. Use templates to generate formally correct PDDL specifications
        3. Use LLM semantic analysis to inform LTL/STL property generation
        4. Generate comprehensive contextual response for semantic understanding

        This hybrid approach ensures both formal correctness and semantic richness.
        """
        if scene_objects is None:
            scene_objects = self._get_scene_objects_from_cache(floor_plan)

        if robot_skills is None:
            robot_skills = self._get_default_robot_skills()

        self.current_scene_objects = scene_objects
        self.current_robot_skills = robot_skills

        logger.info(f"Generating invariants for task with {len(scene_objects)} scene objects")

        pddl = self.extract_pddl_conditions(task, scene_objects, robot_skills)
        logger.info(f"Generated {len(pddl['preconditions'])} preconditions, "
                   f"{len(pddl['postconditions'])} postconditions")

        ltl = self.generate_ltl_properties(task, scene_objects, robot_skills)
        logger.info(f"Generated {len(ltl)} LTL properties")

        stl = self.generate_stl_constraints(task, scene_objects, robot_skills)
        logger.info(f"Generated {len(stl)} STL constraints")

        llm_contextual = self.generate_llm_contextual(task, scene_objects, robot_skills)

        if confidence_score and confidence_score < 0.5:
            logger.info("Low confidence - adding stricter safety constraints")
            ltl.append("G(human_confirmation_required)")
            stl.append("globally[0,10](velocity < 0.1)")

        invariants = Invariants(
            pddl_preconditions=pddl['preconditions'],
            pddl_postconditions=pddl['postconditions'],
            ltl_invariants=ltl,
            stl_constraints=stl,
            llm_contextual=llm_contextual
        )

        return self.validate_consistency(invariants)