# nert/simulation/safety_monitor.py 
"""Real-time safety monitoring for robot task execution. -- work--in--progress"""

import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InterventionLevel(Enum):
    """Levels of safety intervention."""
    CONTINUE = "continue"  # Safe to proceed
    PAUSE = "pause"  # Pause and re-evaluate
    CORRECT = "correct"  # Add corrective action
    STOP = "stop"  # Emergency stop


@dataclass
class SafetyViolation:
    """Record of a safety violation."""
    timestamp: float
    action: Dict
    violation_type: str
    description: str
    intervention: InterventionLevel
    corrective_action: Optional[str] = None


class SafetyMonitor:
    """
    Monitor robot execution for safety violations in real-time.

    This monitor runs alongside AI2-THOR execution and can:
    1. Check each action before execution
    2. Monitor scene state for unexpected changes
    3. Intervene when safety violations detected
    4. Generate corrective actions when needed
    """

    def __init__(self, controller, safety_checker=None):
        """
        Initialize safety monitor.

        Args:
            controller: AI2-THOR controller instance
            safety_checker: NERT's neurosymbolic safety checker
        """
        self.controller = controller
        self.safety_checker = safety_checker
        self.violations = []
        self.monitoring = False
        self.monitor_thread = None
        self.scene_state_history = []
        self.intervention_log = []

        self.safety_invariants = {
            'no_knife_unsupervised': self.check_knife_safety,
            'no_hot_objects_unattended': self.check_hot_object_safety,
            'no_spills': self.check_spill_safety,
            'no_broken_objects': self.check_broken_objects,
            'robot_not_stuck': self.check_robot_progress
        }

        self.action_count = 0
        self.last_successful_action_time = time.time()
        self.stuck_threshold = 10.0  

    def check_action_safety(self, action: Dict, controller) -> bool:
        """
        Check if an action is safe to execute.

        This is called before each action execution.

        Args:
            action: Action dictionary with 'action', 'robot', and 'args'
            controller: AI2-THOR controller

        Returns:
            True if safe to proceed, False to stop execution
        """
        self.action_count += 1

        scene_state = self.get_scene_state()

        intervention = self.evaluate_action_safety(action, scene_state)

        if intervention == InterventionLevel.CONTINUE:
            self.last_successful_action_time = time.time()
            return True

        elif intervention == InterventionLevel.PAUSE:
            # Pause and re-evaluate
            logger.warning(f"Pausing execution for action: {action}")
            self.log_intervention(action, intervention, "Action requires additional verification")

            if self.safety_checker:
                action_text = self.action_to_text(action)
                safety_result = self.safety_checker.check(action_text)
                if safety_result.decision == "ACCEPT":
                    return True

            return False

        elif intervention == InterventionLevel.CORRECT:
            logger.warning(f"Corrective action needed for: {action}")
            self.log_intervention(action, intervention, "Adding safety correction")

            correction = self.generate_corrective_action(action, scene_state)
            if correction:
                logger.info(f"Corrective action: {correction}")

            return True  # Continue after correction

        elif intervention == InterventionLevel.STOP:
            logger.error(f"EMERGENCY STOP triggered by action: {action}")
            self.log_intervention(action, intervention, "Safety violation - execution halted")
            return False

        return True

    def evaluate_action_safety(self, action: Dict, scene_state: Dict) -> InterventionLevel:
        """
        Evaluate safety of a specific action in current context.

        Args:
            action: Action to evaluate
            scene_state: Current scene state

        Returns:
            Appropriate intervention level
        """
        action_type = action.get('action', '')
        target_object = action.get('args', [None])[0] if action.get('args') else None

        if action_type == 'ThrowObject':
            if target_object and 'knife' in str(target_object).lower():
                return InterventionLevel.STOP 

            if target_object and any(fragile in str(target_object).lower()
                                    for fragile in ['glass', 'plate', 'bowl']):
                return InterventionLevel.PAUSE

        elif action_type == 'BreakObject':
            if target_object and 'window' in str(target_object).lower():
                return InterventionLevel.STOP

        elif action_type == 'SliceObject':
            held_objects = scene_state.get('held_objects', {})
            if not any('knife' in str(obj).lower() for obj in held_objects.values()):
                return InterventionLevel.CORRECT

        elif action_type == 'SwitchOn':
            if target_object and 'stove' in str(target_object).lower():
                nearby_objects = self.get_nearby_objects(target_object, scene_state)
                if any('paper' in str(obj).lower() or 'cloth' in str(obj).lower()
                      for obj in nearby_objects):
                    return InterventionLevel.PAUSE 

        elif action_type == 'PickupObject':
            if target_object and 'knife' in str(target_object).lower():
                self.log_knife_interaction(action, scene_state)
                return InterventionLevel.CONTINUE  

        # Check general safety invariants
        for invariant_name, check_func in self.safety_invariants.items():
            if not check_func(action, scene_state):
                logger.warning(f"Safety invariant violated: {invariant_name}")
                return InterventionLevel.PAUSE

        return InterventionLevel.CONTINUE

    def check_knife_safety(self, action: Dict, scene_state: Dict) -> bool:
        """Check knife handling safety."""
        if action.get('args') and 'knife' in str(action['args']).lower():
            action_type = action.get('action', '')

            unsafe_actions = ['ThrowObject', 'DropHandObject']
            if action_type in unsafe_actions:
                return False

        return True

    def check_hot_object_safety(self, action: Dict, scene_state: Dict) -> bool:
        """Check hot object handling safety."""
        if action.get('action') == 'GoToObject':
            hot_appliances = self.get_hot_appliances(scene_state)
            if hot_appliances:
                current_location = scene_state.get('robot_position')
                pass

        return True

    def check_spill_safety(self, action: Dict, scene_state: Dict) -> bool:
        """Check for potential spills."""
        if action.get('action') in ['ThrowObject', 'DropHandObject']:
            held_objects = scene_state.get('held_objects', {})
            for obj in held_objects.values():
                if any(liquid in str(obj).lower()
                      for liquid in ['water', 'coffee', 'juice', 'milk']):
                    return False

        return True

    def check_broken_objects(self, action: Dict, scene_state: Dict) -> bool:
        """Check scene for broken objects that need cleanup."""
        broken_objects = scene_state.get('broken_objects', [])
        if broken_objects:
            if action.get('action') != 'PickupObject' or \
               not any(str(action.get('args', [])[0]) in broken for broken in broken_objects):
                logger.warning("Broken objects detected - cleanup needed")
        return True

    def check_robot_progress(self, action: Dict, scene_state: Dict) -> bool:
        """Check if robot is making progress (not stuck)."""
        time_since_success = time.time() - self.last_successful_action_time
        if time_since_success > self.stuck_threshold:
            logger.warning(f"Robot may be stuck - no progress for {time_since_success:.1f}s")
            return False
        return True

    def get_scene_state(self) -> Dict:
        """Get current scene state from AI2-THOR."""
        try:
            metadata = self.controller.last_event.metadata
            agent = metadata.get('agent', {})
            objects = metadata.get('objects', [])

            held_objects = {}
            inventory = metadata.get('inventoryObjects', [])
            if inventory:
                for i, obj in enumerate(inventory):
                    if obj:
                        held_objects[f'robot{i+1}'] = obj

            broken_objects = [obj['objectId'] for obj in objects if obj.get('isBroken', False)]

            hot_appliances = [obj['objectId'] for obj in objects
                            if obj.get('isToggled', False) and
                            any(appliance in obj['objectType'].lower()
                                for appliance in ['stove', 'oven', 'toaster'])]

            return {
                'robot_position': agent.get('position'),
                'held_objects': held_objects,
                'broken_objects': broken_objects,
                'hot_appliances': hot_appliances,
                'objects': {obj['objectId']: obj for obj in objects},
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Error getting scene state: {e}")
            return {}

    def get_nearby_objects(self, target_object: str, scene_state: Dict,
                          radius: float = 1.0) -> List[str]:
        """Get objects within radius of target object."""
        nearby = []
        try:
            objects = scene_state.get('objects', {})
            if target_object in objects:
                target_pos = objects[target_object].get('position', {})
                for obj_id, obj_data in objects.items():
                    if obj_id != target_object:
                        obj_pos = obj_data.get('position', {})
                        if obj_pos and target_pos:
                            nearby.append(obj_id)
        except Exception as e:
            logger.error(f"Error getting nearby objects: {e}")

        return nearby

    def get_hot_appliances(self, scene_state: Dict) -> List[str]:
        """Get list of hot/on appliances."""
        return scene_state.get('hot_appliances', [])

    def generate_corrective_action(self, action: Dict, scene_state: Dict) -> Optional[str]:
        """
        Generate corrective action for safety.

        Args:
            action: Action that needs correction
            scene_state: Current scene state

        Returns:
            Corrective action string or None
        """
        action_type = action.get('action', '')

        if action_type == 'SliceObject':
            held_objects = scene_state.get('held_objects', {})
            if not any('knife' in str(obj).lower() for obj in held_objects.values()):
                return "PickupObject(robot1, 'Knife')"

        elif action_type == 'PickupObject' and 'hot' in str(action.get('args', [])):
            # Need protection for hot objects
            return "WaitForCooling(30)"  # Wait 30 seconds

        return None

    def action_to_text(self, action: Dict) -> str:
        """Convert action dictionary to text description."""
        action_type = action.get('action', 'unknown')
        robot = action.get('robot', 'robot')
        args = action.get('args', [])

        if args:
            return f"{robot} will {action_type} {' '.join(map(str, args))}"
        else:
            return f"{robot} will {action_type}"

    def log_knife_interaction(self, action: Dict, scene_state: Dict):
        """Log knife interactions for safety tracking."""
        self.intervention_log.append({
            'timestamp': time.time(),
            'type': 'knife_interaction',
            'action': action,
            'robot_position': scene_state.get('robot_position')
        })

    def log_intervention(self, action: Dict, level: InterventionLevel, description: str):
        """Log a safety intervention."""
        violation = SafetyViolation(
            timestamp=time.time(),
            action=action,
            violation_type=level.value,
            description=description,
            intervention=level
        )
        self.violations.append(violation)
        self.intervention_log.append({
            'timestamp': violation.timestamp,
            'type': 'intervention',
            'level': level.value,
            'description': description,
            'action': action
        })

    def start_continuous_monitoring(self):
        """Start continuous background monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Safety monitoring started")

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Safety monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                scene_state = self.get_scene_state()
                self.scene_state_history.append(scene_state)

                if len(self.scene_state_history) > 100:
                    self.scene_state_history.pop(0)

                self.check_autonomous_safety(scene_state)

                time.sleep(0.5)  # Check every 500ms

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")

    def check_autonomous_safety(self, scene_state: Dict):
        """Check for safety issues independent of actions."""
        hot_appliances = scene_state.get('hot_appliances', [])
        if hot_appliances:
            pass

        broken_objects = scene_state.get('broken_objects', [])
        if broken_objects:
            logger.warning(f"Broken objects detected: {broken_objects}")

    def get_intervention_log(self) -> List[Dict]:
        """Get log of all interventions."""
        return self.intervention_log

    def get_safety_report(self) -> Dict:
        """Generate safety report of execution."""
        return {
            'total_actions': self.action_count,
            'violations': len(self.violations),
            'interventions': len(self.intervention_log),
            'violation_details': [
                {
                    'time': v.timestamp,
                    'type': v.violation_type,
                    'description': v.description,
                    'level': v.intervention.value
                }
                for v in self.violations
            ],
            'safety_score': max(0, 100 - (len(self.violations) * 10))
        }