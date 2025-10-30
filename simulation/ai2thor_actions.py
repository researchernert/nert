# nert/simulation/ai2thor_actions.py
"""AI2-THOR action implementations adapted from SMART-LLM."""

import math
import re
import time
import threading
from typing import List, Dict, Tuple, Optional
from collections import deque
import numpy as np
from scipy.spatial import distance
import logging

logger = logging.getLogger(__name__)


class AI2THORActionExecutor:
    """Execute robot actions in AI2-THOR environment."""

    def __init__(self, controller, enable_video: bool = True, session_id: str = None):
        """
        Initialize executor with AI2-THOR controller.

        Args:
            controller: AI2-THOR controller instance
            enable_video: Whether to enable video recording
            session_id: Session ID for video recording
        """
        self.controller = controller
        self.action_queue = deque()
        self.task_over = False
        self.reachable_positions = []
        self.robots = []
        self.execution_thread = None
        self.monitor_callback = None  

        self.enable_video = enable_video
        self.video_recorder = None
        if enable_video and session_id:
            try:
                from simulation.video_recorder import VideoRecorder
                self.video_recorder = VideoRecorder(controller, session_id)
                logger.info(f"Video recording enabled for session {session_id}")
            except Exception as e:
                logger.warning(f"Could not initialize video recorder: {e}")
                self.video_recorder = None

        # Get reachable positions from scene
        self._initialize_scene()

    def step_render(self, **kwargs):
        """Central wrapper that always enforces renderImage=True for third-party camera frames."""
        kwargs["renderImage"] = True
        return self.controller.step(**kwargs)

    def _initialize_scene(self):
        """Initialize scene data from AI2-THOR."""
        try:
            event = self.step_render(action="GetReachablePositions")
            if event.metadata.get("actionReturn"):
                self.reachable_positions = [
                    (p["x"], p["y"], p["z"])
                    for p in event.metadata["actionReturn"]
                ]
            else:
                logger.warning("Could not get reachable positions from scene")
                self.reachable_positions = []

        except Exception as e:
            logger.error(f"Error initializing scene: {e}")
            self.reachable_positions = []

    def set_monitor_callback(self, callback):
        """Set callback for safety monitoring during execution."""
        self.monitor_callback = callback

    def _capture_frame(self, action_name: str, success: bool = True):
        """Helper method to capture video frame if recording is enabled."""
        if self.video_recorder:
            try:
                # Update camera to follow agent before capturing
                self.video_recorder._update_overhead_follow()
                # Capture the frame
                self.video_recorder.capture_frame(action_name, success)
            except Exception as e:
                logger.debug(f"Frame capture failed (non-critical): {e}")

    def _fallback_navigation(self, obj_id: str, obj_data: Dict, obj_name: str) -> bool:
        """
        Fallback navigation when ObjectNavExpertAction fails.

        Attempts to get close to the object using position-based teleportation
        to the nearest reachable position.

        Args:
            obj_id: AI2-THOR object ID
            obj_data: Full object metadata dictionary
            obj_name: Human-readable object name for logging

        Returns:
            True if successfully navigated near object, False otherwise
        """
        try:
            obj_pos = obj_data['position']

            reachable_event = self.step_render(action="GetReachablePositions")
            reachable = reachable_event.metadata.get('actionReturn', [])

            if not reachable:
                logger.error("No reachable positions available for fallback navigation")
                return False

            min_dist = float('inf')
            closest_pos = None

            for pos in reachable:
                dist = ((pos['x'] - obj_pos['x'])**2 +
                       (pos['z'] - obj_pos['z'])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_pos = pos

            # Use 1.5m threshold for better interaction range
            if closest_pos and min_dist < 1.5:
                event = self.step_render(
                    action="Teleport",
                    position=closest_pos,
                    forceAction=True
                )

                if event.metadata.get("lastActionSuccess"):
                    logger.info(f"Fallback navigation successful (teleported {min_dist:.2f}m from {obj_name})")

                    try:
                        look_event = self.step_render(
                            action="LookAtObject",
                            objectId=obj_id,
                            forceAction=True
                        )
                        if look_event.metadata.get("lastActionSuccess"):
                            logger.info(f"Oriented towards {obj_name}")
                    except Exception as e:
                        logger.debug(f"Could not orient towards object: {e}")

                    self._capture_frame(f"GoToObject({obj_name}) - Fallback", True)
                    return True
                else:
                    logger.error(f"Teleport failed: {event.metadata.get('errorMessage')}")
                    return False

            logger.error(f"Fallback navigation failed - closest reachable position is {min_dist:.2f}m away (threshold: 1.5m)")
            self._capture_frame(f"GoToObject({obj_name}) - Failed", False)
            return False

        except Exception as e:
            logger.error(f"Fallback navigation error: {e}")
            return False

    def _navigate_to_position(self, target_pos: Dict) -> bool:
        """Simple position-based navigation fallback."""
        try:
            reachable = self.step_render(action="GetReachablePositions").metadata.get('actionReturn', [])

            if not reachable:
                return False

            min_dist = float('inf')
            closest_pos = None

            for pos in reachable:
                dist = ((pos['x'] - target_pos['x'])**2 +
                       (pos['z'] - target_pos['z'])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_pos = pos

            if closest_pos and min_dist < 1.5:  # Within 1.5 meters
                event = self.step_render(
                    action="Teleport",
                    position=closest_pos,
                    forceAction=True
                )
                return event.metadata.get("lastActionSuccess", False)

            return False

        except Exception as e:
            logger.error(f"Position navigation failed: {e}")
            return False

    def execute_action_sequence(self, actions: List[str], robots: List[Dict] = None) -> Dict:
        """
        Execute a sequence of robot actions.

        Args:
            actions: List of action strings like "GoToObject(robot1, 'Apple')"
            robots: List of robot configurations

        Returns:
            Execution result dictionary
        """
        if robots is None:
            robots = [{'name': 'robot1', 'skills': ['all']}]

        self.robots = robots
        results = {
            'success': True,
            'executed_actions': [],
            'failed_action': None,
            'error': None,
            'screenshots': []
        }

        try:
            # Capture initial state for recording
            if self.video_recorder:
                self._capture_frame("Initial State", True)

            for action_str in actions:
                action_str_stripped = action_str.strip()
                if action_str_stripped.startswith('assert') or action_str_stripped.startswith('#') or not action_str_stripped:
                    continue

                parsed = self._parse_action(action_str)
                if parsed:
                    self.action_queue.append(parsed)
                else:
                    results['success'] = False
                    results['error'] = f"Could not parse action: {action_str}"
                    return results

            # Execute actions sequentially
            while self.action_queue:
                action = self.action_queue.popleft()

                # Safety monitoring callback
                if self.monitor_callback:
                    if not self.monitor_callback(action, self.controller):
                        results['success'] = False
                        results['error'] = f"Safety monitor stopped execution at: {action}"
                        break

                # Execute the action
                success = self._execute_single_action(action)

                if success:
                    results['executed_actions'].append(action)
                    # Capture screenshot after action
                    if self.controller.last_event.frame is not None:
                        results['screenshots'].append(self.controller.last_event.frame)
                else:
                    results['success'] = False
                    results['failed_action'] = action
                    results['error'] = f"Action failed: {action}"
                    break

                # Small delay between actions for stability
                time.sleep(0.1)

        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            logger.error(f"Error executing action sequence: {e}")

        # Finalize video recording 
        if self.video_recorder:
            try:
                video_results = self.video_recorder.finalize()
                results['video'] = video_results
                logger.info(f"Video recording complete: {video_results}")
            except Exception as e:
                logger.warning(f"Could not finalize video: {e}")

        return results

    def _parse_action(self, action_str: str) -> Optional[Dict]:
        """
        Parse action string into structured format.

        Example: "GoToObject(robot1, 'Apple')" ->
                 {'action': 'GoToObject', 'robot': 'robot1', 'args': ['Apple']}
        """
        pattern = r'(\w+)\((robot\d+)(?:,\s*[\'"]([^\'"]*)[\'"]\s*(?:,\s*[\'"]([^\'"]*)[\'"])?)?\)'
        match = re.match(pattern, action_str.strip())

        if not match:
            pattern2 = r'(\w+)\((robot\d+)\)'
            match = re.match(pattern2, action_str.strip())
            if match:
                return {
                    'action': match.group(1),
                    'robot': match.group(2),
                    'args': []
                }
            logger.warning(f"Could not parse action: {action_str}")
            return None

        action_name = match.group(1)
        robot_name = match.group(2)
        args = [arg for arg in match.groups()[2:] if arg is not None]

        return {
            'action': action_name,
            'robot': robot_name,
            'args': args
        }

    def _execute_single_action(self, action: Dict) -> bool:
        """
        Execute a single parsed action.

        Args:
            action: Parsed action dictionary

        Returns:
            True if successful, False otherwise
        """
        action_name = action['action']
        robot_name = action['robot']
        args = action['args']

        agent_id = int(robot_name[-1]) - 1

        try:
            if action_name == 'GoToObject':
                return self.GoToObject(robot_name, args[0] if args else None)
            elif action_name == 'PickupObject':
                return self.PickupObject(robot_name, args[0] if args else None)
            elif action_name == 'PutObject':
                return self.PutObject(robot_name, args[0] if args else None,
                                     args[1] if len(args) > 1 else None)
            elif action_name == 'OpenObject':
                return self.OpenObject(robot_name, args[0] if args else None)
            elif action_name == 'CloseObject':
                return self.CloseObject(robot_name, args[0] if args else None)
            elif action_name == 'SwitchOn':
                return self.SwitchOn(robot_name, args[0] if args else None)
            elif action_name == 'SwitchOff':
                return self.SwitchOff(robot_name, args[0] if args else None)
            elif action_name == 'BreakObject':
                return self.BreakObject(robot_name, args[0] if args else None)
            elif action_name == 'SliceObject':
                return self.SliceObject(robot_name, args[0] if args else None)
            elif action_name == 'ThrowObject':
                return self.ThrowObject(robot_name, args[0] if args else None)
            elif action_name == 'PushObject':
                return self.PushObject(robot_name, args[0] if args else None)
            elif action_name == 'PullObject':
                return self.PullObject(robot_name, args[0] if args else None)
            elif action_name == 'DropHandObject':
                return self.DropHandObject(robot_name)
            else:
                logger.warning(f"Unknown action: {action_name}")
                return False

        except Exception as e:
            logger.error(f"Error executing {action_name}: {e}")
            return False

    def GoToObject(self, robot: str, dest_obj: str) -> bool:
        """Navigate robot to object with improved error handling."""
        if not dest_obj:
            logger.error("GoToObject: No destination object specified")
            return False

        logger.info(f"Going to {dest_obj}")

        try:
            objects = self.controller.last_event.metadata.get("objects", [])

            dest_obj_id = None
            dest_obj_data = None

            for obj in objects:
                if re.search(dest_obj, obj["objectId"], re.IGNORECASE):
                    dest_obj_id = obj["objectId"]
                    dest_obj_data = obj
                    break

            if not dest_obj_id:
                logger.warning(f"Object '{dest_obj}' not found in scene")
                self._capture_frame(f"GoToObject({dest_obj}) - Not Found", False)
                return False

            event = self.step_render(
                action="ObjectNavExpertAction",
                objectId=dest_obj_id
            )

            next_action = event.metadata.get('actionReturn')

            if next_action is None:
                logger.warning(f"No navigation path to {dest_obj} (may be in closed container)")

                if dest_obj_data.get('position'):
                    success = self._navigate_to_position(dest_obj_data['position'])
                    if success:
                        self._capture_frame(f"GoToObject({dest_obj})", True)
                        return True

                self._capture_frame(f"GoToObject({dest_obj}) - No Path", False)
                return False

            event = self.step_render(action=next_action, forceAction=True)

            if event.metadata.get("lastActionSuccess"):
                logger.info(f"Reached {dest_obj}")
                self._capture_frame(f"GoToObject({dest_obj})", True)
                return True
            else:
                error_msg = event.metadata.get('errorMessage', 'Unknown error')
                logger.warning(f"Navigation failed: {error_msg}")
                self._capture_frame(f"GoToObject({dest_obj}) - Failed", False)
                return False

        except Exception as e:
            logger.error(f"Error in GoToObject: {e}")
            return False

    def PickupObject(self, robot: str, pick_obj: str) -> bool:
        """Pick up an object."""
        if not pick_obj:
            logger.error("PickupObject: No object specified")
            return False

        try:
            objects = self.controller.last_event.metadata.get("objects", [])

            pick_obj_id = None
            for obj in objects:
                if re.search(pick_obj, obj["objectId"], re.IGNORECASE):
                    pick_obj_id = obj["objectId"]
                    break

            if not pick_obj_id:
                logger.warning(f"Object '{pick_obj}' not found")
                return False

            event = self.step_render(
                action="PickupObject",
                objectId=pick_obj_id,
                forceAction=True
            )

            success = event.metadata.get("lastActionSuccess", False)
            if success:
                logger.info(f"Picked up {pick_obj}")
                self._capture_frame(f"PickupObject({pick_obj})", True)
            else:
                logger.warning(f"Failed to pickup {pick_obj}: {event.metadata.get('errorMessage')}")
                self._capture_frame(f"PickupObject({pick_obj})", False)

            return success

        except Exception as e:
            logger.error(f"Error in PickupObject: {e}")
            return False

    def PutObject(self, robot: str, put_obj: str, receptacle: str) -> bool:
        """Put object on/in receptacle."""
        if not receptacle:
            logger.error("PutObject: No receptacle specified")
            return False

        try:
            objects = self.controller.last_event.metadata.get("objects", [])

            recp_obj_id = None
            min_distance = float('inf')

            for obj in objects:
                if re.search(receptacle, obj["objectId"], re.IGNORECASE):
                    dist = obj.get("distance", float('inf'))
                    if dist < min_distance:
                        recp_obj_id = obj["objectId"]
                        min_distance = dist

            if not recp_obj_id:
                logger.warning(f"Receptacle '{receptacle}' not found")
                return False

            event = self.step_render(
                action="PutObject",
                objectId=recp_obj_id,
                forceAction=True
            )

            success = event.metadata.get("lastActionSuccess", False)
            if success:
                logger.info(f"Put object on {receptacle}")
                self._capture_frame(f"PutObject({put_obj}, {receptacle})", True)
            else:
                logger.warning(f"Failed to put on {receptacle}: {event.metadata.get('errorMessage')}")
                self._capture_frame(f"PutObject({put_obj}, {receptacle})", False)

            return success

        except Exception as e:
            logger.error(f"Error in PutObject: {e}")
            return False

    def OpenObject(self, robot: str, obj_name: str) -> bool:
        """Open an object (door, drawer, etc.)."""
        if not obj_name:
            return False

        try:
            objects = self.controller.last_event.metadata.get("objects", [])

            target_id = None
            for obj in objects:
                if re.search(obj_name, obj["objectId"], re.IGNORECASE):
                    if obj.get("openable", False): 
                        target_id = obj["objectId"]
                        break

            if not target_id:
                logger.warning(f"Openable object '{obj_name}' not found")
                return False

            event = self.step_render(
                action="OpenObject",
                objectId=target_id,
                forceAction=True
            )

            success = event.metadata.get("lastActionSuccess", False)
            if success:
                logger.info(f"Opened {obj_name}")
                self._capture_frame(f"OpenObject({obj_name})", True)
            else:
                logger.warning(f"Failed to open {obj_name}: {event.metadata.get('errorMessage')}")
                self._capture_frame(f"OpenObject({obj_name})", False)

            return success

        except Exception as e:
            logger.error(f"Error in OpenObject: {e}")
            return False

    def CloseObject(self, robot: str, obj_name: str) -> bool:
        """Close an object."""
        if not obj_name:
            return False

        try:
            objects = self.controller.last_event.metadata.get("objects", [])

            target_id = None
            for obj in objects:
                if re.search(obj_name, obj["objectId"], re.IGNORECASE):
                    if obj.get("openable", False):
                        target_id = obj["objectId"]
                        break

            if not target_id:
                logger.warning(f"Closeable object '{obj_name}' not found")
                return False

            event = self.step_render(
                action="CloseObject",
                objectId=target_id,
                forceAction=True
            )

            success = event.metadata.get("lastActionSuccess", False)
            if success:
                logger.info(f"Closed {obj_name}")
                self._capture_frame(f"CloseObject({obj_name})", True)
            else:
                logger.warning(f"Failed to close {obj_name}: {event.metadata.get('errorMessage')}")
                self._capture_frame(f"CloseObject({obj_name})", False)

            return success

        except Exception as e:
            logger.error(f"Error in CloseObject: {e}")
            return False

    def SwitchOn(self, robot: str, obj_name: str) -> bool:
        """Turn on an appliance."""
        if not obj_name:
            return False

        try:
            objects = self.controller.last_event.metadata.get("objects", [])

            target_id = None
            for obj in objects:
                if re.search(obj_name, obj["objectId"], re.IGNORECASE):
                    if obj.get("toggleable", False): 
                        target_id = obj["objectId"]
                        break

            if not target_id:
                logger.warning(f"Toggleable object '{obj_name}' not found")
                return False

            event = self.step_render(
                action="ToggleObjectOn",
                objectId=target_id,
                forceAction=True
            )

            success = event.metadata.get("lastActionSuccess", False)
            if success:
                logger.info(f"Switched on {obj_name}")
                self._capture_frame(f"SwitchOn({obj_name})", True)
            else:
                logger.warning(f"Failed to switch on {obj_name}: {event.metadata.get('errorMessage')}")
                self._capture_frame(f"SwitchOn({obj_name})", False)

            return success

        except Exception as e:
            logger.error(f"Error in SwitchOn: {e}")
            return False

    def SwitchOff(self, robot: str, obj_name: str) -> bool:
        """Turn off an appliance."""
        if not obj_name:
            return False

        try:
            objects = self.controller.last_event.metadata.get("objects", [])

            target_id = None
            for obj in objects:
                if re.search(obj_name, obj["objectId"], re.IGNORECASE):
                    if obj.get("toggleable", False):
                        target_id = obj["objectId"]
                        break

            if not target_id:
                logger.warning(f"Toggleable object '{obj_name}' not found")
                return False

            event = self.step_render(
                action="ToggleObjectOff",
                objectId=target_id,
                forceAction=True
            )

            success = event.metadata.get("lastActionSuccess", False)
            if success:
                logger.info(f"Switched off {obj_name}")
                self._capture_frame(f"SwitchOff({obj_name})", True)
            else:
                logger.warning(f"Failed to switch off {obj_name}: {event.metadata.get('errorMessage')}")
                self._capture_frame(f"SwitchOff({obj_name})", False)

            return success

        except Exception as e:
            logger.error(f"Error in SwitchOff: {e}")
            return False

    def BreakObject(self, robot: str, obj_name: str) -> bool:
        """Break an object."""
        if not obj_name:
            return False

        try:
            objects = self.controller.last_event.metadata.get("objects", [])

            target_id = None
            for obj in objects:
                if re.search(obj_name, obj["objectId"], re.IGNORECASE):
                    if obj.get("breakable", False): 
                        target_id = obj["objectId"]
                        break

            if not target_id:
                logger.warning(f"Breakable object '{obj_name}' not found")
                return False

            event = self.step_render(
                action="BreakObject",
                objectId=target_id,
                forceAction=True
            )

            success = event.metadata.get("lastActionSuccess", False)
            if success:
                logger.info(f"Broke {obj_name}")
                self._capture_frame(f"BreakObject({obj_name})", True)
            else:
                logger.warning(f"Failed to break {obj_name}: {event.metadata.get('errorMessage')}")
                self._capture_frame(f"BreakObject({obj_name})", False)

            return success

        except Exception as e:
            logger.error(f"Error in BreakObject: {e}")
            return False

    def SliceObject(self, robot: str, obj_name: str) -> bool:
        """Slice an object."""
        if not obj_name:
            return False

        try:
            objects = self.controller.last_event.metadata.get("objects", [])

            target_id = None
            for obj in objects:
                if re.search(obj_name, obj["objectId"], re.IGNORECASE):
                    if obj.get("sliceable", False):
                        target_id = obj["objectId"]
                        break

            if not target_id:
                logger.warning(f"Sliceable object '{obj_name}' not found")
                return False

            event = self.step_render(
                action="SliceObject",
                objectId=target_id,
                forceAction=True
            )

            success = event.metadata.get("lastActionSuccess", False)
            if success:
                logger.info(f"Sliced {obj_name}")
                self._capture_frame(f"SliceObject({obj_name})", True)
            else:
                logger.warning(f"Failed to slice {obj_name}: {event.metadata.get('errorMessage')}")
                self._capture_frame(f"SliceObject({obj_name})", False)

            return success

        except Exception as e:
            logger.error(f"Error in SliceObject: {e}")
            return False

    def ThrowObject(self, robot: str, obj_name: str) -> bool:
        """Throw an object."""
        result = self.DropHandObject(robot)
        self._capture_frame(f"ThrowObject({obj_name})", result)
        return result

    def PushObject(self, robot: str, obj_name: str) -> bool:
        """Push an object."""
        if not obj_name:
            return False

        try:
            objects = self.controller.last_event.metadata.get("objects", [])

            target_id = None
            for obj in objects:
                if re.search(obj_name, obj["objectId"], re.IGNORECASE):
                    if obj.get("moveable", False) or obj.get("pickupable", False):
                        target_id = obj["objectId"]
                        break

            if not target_id:
                logger.warning(f"Pushable object '{obj_name}' not found")
                return False

            event = self.step_render(
                action="PushObject",
                objectId=target_id,
                forceAction=True
            )

            success = event.metadata.get("lastActionSuccess", False)
            if success:
                logger.info(f"Pushed {obj_name}")
                self._capture_frame(f"PushObject({obj_name})", True)
            else:
                logger.warning(f"Failed to push {obj_name}: {event.metadata.get('errorMessage')}")
                self._capture_frame(f"PushObject({obj_name})", False)

            return success

        except Exception as e:
            logger.error(f"Error in PushObject: {e}")
            return False

    def PullObject(self, robot: str, obj_name: str) -> bool:
        """Pull an object."""
        if not obj_name:
            return False

        try:
            objects = self.controller.last_event.metadata.get("objects", [])

            target_id = None
            for obj in objects:
                if re.search(obj_name, obj["objectId"], re.IGNORECASE):
                    if obj.get("moveable", False) or obj.get("pickupable", False):
                        target_id = obj["objectId"]
                        break

            if not target_id:
                logger.warning(f"Pullable object '{obj_name}' not found")
                return False

            event = self.step_render(
                action="PullObject",
                objectId=target_id,
                forceAction=True
            )

            success = event.metadata.get("lastActionSuccess", False)
            if success:
                logger.info(f"Pulled {obj_name}")
                self._capture_frame(f"PullObject({obj_name})", True)
            else:
                logger.warning(f"Failed to pull {obj_name}: {event.metadata.get('errorMessage')}")
                self._capture_frame(f"PullObject({obj_name})", False)

            return success

        except Exception as e:
            logger.error(f"Error in PullObject: {e}")
            return False

    def DropHandObject(self, robot: str) -> bool:
        """Drop currently held object."""
        try:
            event = self.step_render(
                action="DropHandObject",
                forceAction=True
            )

            success = event.metadata.get("lastActionSuccess", False)
            if success:
                logger.info("Dropped held object")
                self._capture_frame("DropHandObject()", True)
            else:
                logger.warning(f"Failed to drop object: {event.metadata.get('errorMessage')}")
                self._capture_frame("DropHandObject()", False)

            return success

        except Exception as e:
            logger.error(f"Error in DropHandObject: {e}")
            return False

    def get_scene_objects(self) -> List[str]:
        """Get list of all objects currently in the scene."""
        try:
            objects = self.controller.last_event.metadata.get("objects", [])
            return [obj["objectType"] for obj in objects]
        except Exception as e:
            logger.error(f"Error getting scene objects: {e}")
            return []

    def get_robot_state(self) -> Dict:
        """Get current robot state including position and held object."""
        try:
            metadata = self.controller.last_event.metadata
            agent = metadata.get("agent", {})

            return {
                'position': agent.get('position'),
                'rotation': agent.get('rotation'),
                'held_object': metadata.get('inventoryObjects', [None])[0] if metadata.get('inventoryObjects') else None,
                'reachable_positions': len(self.reachable_positions)
            }
        except Exception as e:
            logger.error(f"Error getting robot state: {e}")
            return {}