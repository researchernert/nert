# nert/simulation/task_executor.py
"""Execute tasks in AI2-THOR simulator."""

import traceback
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Execute generated code in simulation or mock mode."""
    
    def __init__(self, use_ai2thor: bool = False):
        self.use_ai2thor = use_ai2thor
        if use_ai2thor:
            self.init_ai2thor()
        else:
            self.controller = None
    
    def init_ai2thor(self):
        """Initialize AI2-THOR if available."""
        try:
            from ai2thor.controller import Controller
            self.controller = Controller(
                scene="FloorPlan1",
                width=1280,
                height=720,
                headless=True
            )
        except ImportError:
            print("AI2-THOR not available, using mock execution")
            self.controller = None
    
    def execute(self, code: str, dry_run: bool = False) -> Dict:
        """Execute code in simulator or mock mode."""
        result = {
            'success': False,
            'error': None,
            'trace': [],
            'final_state': None
        }
        
        if dry_run or not self.controller:
            result['success'] = True
            result['trace'] = ['Mock execution completed']
            result['final_state'] = {'mode': 'mock'}
            return result
        
        try:
            namespace = {
                'navigate_to': self.navigate_to,
                'pickup': self.pickup,
                'place': self.place,
                'open_object': self.open_object,
                'close_object': self.close_object,
                'scene_has': self.scene_has
            }
            
            exec(code, namespace)
            
            result['success'] = True
            result['final_state'] = self.get_state()
            
        except Exception as e:
            result['error'] = str(e)
            result['trace'] = traceback.format_exc().split('\n')
        
        return result
    
    def navigate_to(self, object_name: str):
        """Navigate robot to object."""
        if self.controller:
            for obj in self.controller.last_event.metadata["objects"]:
                if object_name.lower() in obj["objectType"].lower():
                    self.controller.step(
                        action="Teleport",
                        position=obj["position"]
                    )
                    return
            raise ValueError(f"Object not found: {object_name}")
    
    def pickup(self, object_name: str):
        """Pick up an object."""
        if self.controller:
            for obj in self.controller.last_event.metadata["objects"]:
                if object_name.lower() in obj["objectType"].lower():
                    self.controller.step(
                        action="PickupObject",
                        objectId=obj["objectId"]
                    )
                    return
            raise ValueError(f"Cannot pickup: {object_name}")
    
    def place(self, object_name: str, location: str):
        """Place object at location."""
        if self.controller:
            held_object = self.controller.last_event.metadata["agent"].get("heldObject")
            if not held_object:
                raise ValueError(f"Place failed: robot is not holding any object")

            receptacle_obj = None
            for obj in self.controller.last_event.metadata["objects"]:
                if location.lower() in obj["objectType"].lower():
                    receptacle_obj = obj
                    break

            if not receptacle_obj:
                raise ValueError(f"Place failed: receptacle '{location}' not found")

            if receptacle_obj.get("openable") and not receptacle_obj.get("isOpen", False):
                logger.info(f"Opening {location} before placing")
                self.controller.step(
                    action="OpenObject",
                    objectId=receptacle_obj["objectId"]
                )

            self.controller.step(
                action="PutObject",
                objectId=held_object["objectId"],
                receptacleObjectId=receptacle_obj["objectId"],
                forceAction=False
            )
            return
    
    def open_object(self, object_name: str):
        """Open an object."""
        if self.controller:
            for obj in self.controller.last_event.metadata["objects"]:
                if object_name.lower() in obj["objectType"].lower():
                    self.controller.step(
                        action="OpenObject",
                        objectId=obj["objectId"]
                    )
                    return
    
    def close_object(self, object_name: str):
        """Close an object."""
        if self.controller:
            for obj in self.controller.last_event.metadata["objects"]:
                if object_name.lower() in obj["objectType"].lower():
                    self.controller.step(
                        action="CloseObject",
                        objectId=obj["objectId"]
                    )
                    return
    
    def scene_has(self, object_name: str) -> bool:
        """Check if object exists in scene."""
        if self.controller:
            for obj in self.controller.last_event.metadata["objects"]:
                if object_name.lower() in obj["objectType"].lower():
                    return True
        return False
    
    def get_state(self) -> Dict:
        """Get current simulator state."""
        if self.controller:
            return {
                'agent_position': self.controller.last_event.metadata["agent"]["position"],
                'holding': self.controller.last_event.metadata["agent"].get("heldObject"),
                'objects': [obj["objectType"] for obj in self.controller.last_event.metadata["objects"]]
            }
        return {'mode': 'mock'}