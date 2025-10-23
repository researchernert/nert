# nert/core/grounding_verifier.py
"""Semantic grounding verification with hybrid LLM-assisted matching."""

import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import re
import logging
from models.embedding_cache import get_encoder

logger = logging.getLogger(__name__)


class SemanticGroundingVerifier:    
    def __init__(self, scene_cache_path: str = "data/scene_cache.json", 
                 llm_client=None):
        self.encoder = get_encoder()  
        self.llm = llm_client  
        self.load_scene_cache(scene_cache_path)

        self.ai2thor_available = self.check_ai2thor_available()

        self.mapping_cache = {}
    
    def load_scene_cache(self, path: str):
        """Load pre-computed scene data."""
        cache_file = Path(path)
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.scene_cache = json.load(f)
        else:
            self.scene_cache = self.generate_default_cache()
    
    def generate_default_cache(self) -> Dict:
        """Generate default scene cache with common objects."""
        return {
            "floorplan_1": {
                "objects": ["Table", "Chair", "Apple", "Fridge", "Microwave", 
                           "Sink", "Cabinet", "Knife", "Plate", "Cup", "Bowl",
                           "GarbageCan", "CounterTop", "Toaster", "CoffeeMachine"],
                "robot_skills": ["GoToObject", "PickupObject", "PutObject", 
                                "OpenObject", "CloseObject", "SwitchOn", "SwitchOff"]
            }
        }
    
    def check_ai2thor_available(self) -> bool:
        """Check if AI2-THOR is available without initializing controller."""
        try:
            import ai2thor
            return True
        except ImportError:
            return False
    
    def semantic_match(self, query: str, candidates: List[str], 
                      threshold: float = 0.7) -> List[Tuple[str, float]]:
        query_embedding = self.encoder.encode(query)
        matches = []
        
        for candidate in candidates:
            candidate_embedding = self.encoder.encode(candidate)
            similarity = np.dot(query_embedding, candidate_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
            )
            
            if similarity > threshold:
                matches.append((candidate, similarity))
        
        if matches:
            return sorted(matches, key=lambda x: x[1], reverse=True)
        
        if self.llm and threshold < 0.8:  
            llm_match = self.llm_assisted_match(query, candidates)
            if llm_match:
                matches.append((llm_match, 0.75))  # Give LLM matches moderate confidence
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def llm_assisted_match(self, query: str, candidates: List[str]) -> Optional[str]:
        """
        Use LLM to understand semantic relationships, constrained to actual candidates.

        Falls back to fuzzy matching if LLM response is not in candidate list.
        """
        cache_key = f"{query}|{','.join(sorted(candidates))}"
        if cache_key in self.mapping_cache:
            return self.mapping_cache[cache_key]
        
        if not self.llm:
            return None
        
        prompt = f"""
        The user mentioned: "{query}"
        
        Available objects in the scene are: {candidates}
        
        Which object from the scene list does "{query}" most likely refer to?
        Consider synonyms, descriptions, and common references.
        
        Rules:
        - Return ONLY an object name from the provided list
        - If no reasonable match exists, return "NONE"
        - Do not invent objects not in the list
        - Only extract a person or an entity if they are actively part of completing the task and not just because they are in the scene
        
        Examples:
        - "red fruit" might refer to "Apple"
        - "cooling device" might refer to "Fridge"
        - "trash" might refer to "GarbageCan"
        
        Your answer (single word from the list or NONE):
        """
        
        try:
            response = self.llm.call(prompt).strip()
            
            if response == "NONE":
                result = None
            elif response in candidates:
                result = response
            else:
                result = self.fuzzy_match_best(response, candidates)
            
            self.mapping_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"LLM matching failed: {e}")
            return None
    
    def fuzzy_match_best(self, query: str, candidates: List[str]) -> Optional[str]:
        """
        Find best fuzzy match using string similarity.
        
        Fallback for when embeddings and LLM don't work.
        """
        query_lower = query.lower().replace('_', '').replace('-', '').replace(' ', '')
        
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            cand_lower = candidate.lower().replace('_', '').replace('-', '').replace(' ', '')
            
            if query_lower == cand_lower:
                return candidate
            
            if query_lower in cand_lower or cand_lower in query_lower:
                score = len(query_lower) / max(len(query_lower), len(cand_lower))
                if score > best_score:
                    best_score = score
                    best_match = candidate
        
        return best_match if best_score > 0.5 else None
    
    def extract_required_objects(self, task: str, invariants: Dict, scene_objects: Optional[List[str]] = None) -> List[str]:
        """
        Extract objects mentioned in task and invariants.

        Two-phase extraction:
        1. Pure linguistic extraction from task text (scene-agnostic)
        2. PDDL invariants extraction (LLM-generated, context-aware)

        Both sources are combined to ensure comprehensive requirement extraction.
        Verification phase handles grounding against actual scene availability.
        """
        objects = []

        import re
        task_lower = task.lower()  

        try:
            import spacy
            if not hasattr(self, '_nlp'):
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                    self._nlp = None

            if self._nlp:
                doc = self._nlp(task)
                for chunk in doc.noun_chunks:
                    chunk_text = chunk.text.lower().strip()
                    chunk_text = re.sub(r'\b(the|a|an)\b\s*', '', chunk_text).strip()
                    if chunk_text and len(chunk_text) > 2:
                        objects.append(chunk_text)
            else:
                raise ImportError("spaCy not available")

        except ImportError:
            logger.info("Using fallback heuristic noun-phrase extraction (spaCy not available)")

            task_words = re.findall(r'\b\w+\b', task_lower)

            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'from', 'with',
                'by', 'for', 'of', 'as', 'is', 'was', 'are', 'were', 'be', 'been',
                'place', 'put', 'get', 'take', 'move', 'bring', 'make', 'set', 'then',
                'it', 'its', 'that', 'this', 'these', 'those', 'there', 'here'
            }

            current_phrase = []
            for word in task_words:
                if word in stop_words:
                    if current_phrase:
                        phrase = ' '.join(current_phrase)
                        if len(phrase) > 2:
                            objects.append(phrase)
                        current_phrase = []
                else:
                    current_phrase.append(word)

            if current_phrase:
                phrase = ' '.join(current_phrase)
                if len(phrase) > 2:
                    objects.append(phrase)


        if invariants:
            pddl_predicates = ['at', 'not', 'exists', 'holding', 'gripper_empty',
                              'clear', 'on', 'and', 'or', 'forall', 'when',
                              'implies', 'empty', 'full', 'near', 'above', 'below']

            for inv_type in ['pddl_preconditions', 'pddl_postconditions']:
                if inv_type in invariants:
                    for condition in invariants[inv_type]:
                        exists_matches = re.findall(r'exists\(([^)]+)\)', condition)
                        for match in exists_matches:
                            obj = match.strip()
                            if obj and obj not in pddl_predicates:
                                objects.append(obj)

                        at_matches = re.findall(r'at\(([^,]+),\s*([^)]+)\)', condition)
                        for match in at_matches:
                            for item in match:
                                item = item.strip()
                                if item and not item.isupper() and item not in pddl_predicates:
                                    if item != 'robot':
                                        objects.append(item)

                        holding_matches = re.findall(r'holding\(([^)]+)\)', condition)
                        for match in holding_matches:
                            obj = match.strip()
                            if obj and not obj.isupper() and obj not in pddl_predicates:
                                objects.append(obj)

                        on_matches = re.findall(r'on\(([^,]+),\s*([^)]+)\)', condition)
                        for match in on_matches:
                            for item in match:
                                item = item.strip()
                                if item and not item.isupper() and item not in pddl_predicates:
                                    objects.append(item)
        
        if self.llm and ('it' in task_lower or 'that' in task_lower or 'there' in task_lower):
            indirect_objects = self.resolve_pronouns(task, objects)
            objects.extend(indirect_objects)

        compound_objects = []
        simple_words = []

        for obj in objects:
            if any(c.isupper() for c in obj[1:]): 
                compound_objects.append(obj)
            else:
                simple_words.append(obj)

        words_to_remove = set()

        for compound in compound_objects:
            compound_words = []
            current_word = []
            for char in compound:
                if char.isupper() and current_word:
                    compound_words.append(''.join(current_word).lower())
                    current_word = [char.lower()]
                else:
                    current_word.append(char.lower())
            if current_word:
                compound_words.append(''.join(current_word))

            all_words_present = all(word in [w.lower() for w in simple_words] for word in compound_words)

            if all_words_present:
                for word in compound_words:
                    words_to_remove.add(word.lower())

        filtered_simple_words = [w for w in simple_words if w.lower() not in words_to_remove]

        filtered_objects = filtered_simple_words + compound_objects

        seen = set()
        deduplicated = []
        for obj in filtered_objects:
            obj_lower = obj.lower()
            if obj_lower not in seen:
                seen.add(obj_lower)
                deduplicated.append(obj)

        return deduplicated

    def extract_required_skills(self, task: str, invariants: Dict) -> List[str]:
        skills = []
        task_lower = task.lower()

        skill_mapping = {
            'go': 'navigate',
            'move': 'navigate',
            'navigate': 'navigate',
            'gotoobject': 'navigate',
            'pick': 'pickup',
            'pickup': 'pickup',
            'pickupobject': 'pickup',
            'grab': 'pickup',
            'get': 'pickup',
            'place': 'place',
            'put': 'place',
            'putobject': 'place',
            'drop': 'place',
            'open': 'open',
            'openobject': 'open',
            'close': 'close',
            'closeobject': 'close',
            'slice': 'slice',
            'sliceobject': 'slice',
            'cut': 'slice',
            'pour': 'pour',
            'switch': 'toggle',
            'switchon': 'toggle',
            'switchoff': 'toggle',
            'turn': 'toggle',
            'break': 'break',
            'breakobject': 'break',
            'throw': 'throw',
            'throwobject': 'throw',
            'push': 'push',
            'pushobject': 'push',
            'pull': 'pull',
            'pullobject': 'pull',
            'drophand': 'drop',
            'drophandobject': 'drop'
        }

        for keyword, skill in skill_mapping.items():
            if keyword in task_lower:
                if skill not in skills:
                    skills.append(skill)

        if invariants:
            for inv_type in ['pddl_preconditions', 'pddl_postconditions']:
                if inv_type in invariants:
                    for condition in invariants[inv_type]:
                        condition_lower = condition.lower()
                        for keyword, skill in skill_mapping.items():
                            if keyword in condition_lower:
                                if skill not in skills:
                                    skills.append(skill)

            if 'llm_contextual' in invariants and invariants['llm_contextual']:
                llm_response = invariants['llm_contextual']

                if isinstance(llm_response, str):
                    llm_lower = llm_response.lower()

                    import re
                    action_pattern = r'(\w+)\s*\('
                    action_matches = re.findall(action_pattern, llm_response)

                    for action in action_matches:
                        action_lower = action.lower()
                        for keyword, skill in skill_mapping.items():
                            if keyword in action_lower:
                                if skill not in skills:
                                    skills.append(skill)
                                break

        return skills

    def resolve_pronouns(self, task: str, context_objects: List[str]) -> List[str]:
        """
        Resolve pronouns and indirect references in the task.
        
        LLM helps understand context but output is validated.
        """
        if not self.llm:
            return []
        
        prompt = f"""
        Task: "{task}"
        Objects already mentioned: {context_objects}
        
        Are there any pronouns (it, that, this, there) that refer to objects?
        If so, what objects do they likely refer to?
        
        Return a comma-separated list of object names, or "NONE" if no pronouns.
        Be conservative - only include clear references.
        """
        
        try:
            response = self.llm.call(prompt).strip()
            if response == "NONE":
                return []
            
            objects = [obj.strip() for obj in response.split(',')]
            return [obj for obj in objects if obj] 
            
        except Exception as e:
            logger.warning(f"Pronoun resolution failed: {e}")
            return []
    
    def verify(self, task: str, invariants: Dict, floor_plan: int = 1,
               confidence_threshold: float = 0.5,
               scene_objects: Optional[List[str]] = None,
               robot_skills: Optional[List[str]] = None) -> Tuple[bool, Dict]:

        data_source_used = 'unknown'

        if scene_objects is not None and robot_skills is not None:
            logger.info(f"Using provided scene objects ({len(scene_objects)} objects) and skills ({len(robot_skills)} skills)")
            data_source_used = 'frontend'
        elif scene_objects is not None:
            logger.info(f"Using provided scene objects ({len(scene_objects)} objects), fetching skills")
            if robot_skills is None:
                robot_skills = self.get_live_robot_skills()
            data_source_used = 'frontend'
        else:
            logger.warning(f"Scene objects not provided by frontend, falling back to cache for FloorPlan{floor_plan}")

            scene_key = f"floorplan_{floor_plan}"
            scene_data = self.scene_cache.get(scene_key, {})
            scene_objects = scene_data.get('objects', [])
            robot_skills = scene_data.get('robot_skills', []) if robot_skills is None else robot_skills
            data_source_used = 'cache'

            if scene_objects:
                logger.info(f"Using cached data for {scene_key}: {len(scene_objects)} objects")
            else:
                logger.warning(f"No data found for {scene_key} in cache")
                scene_objects = self.scene_cache.get('floorplan_1', {}).get('objects', [])
                robot_skills = self.scene_cache.get('floorplan_1', {}).get('robot_skills', [])
                logger.warning(f"Using FloorPlan1 defaults: {len(scene_objects)} objects")
                data_source_used = 'cache_fallback'

        required_objects = self.extract_required_objects(task, invariants, scene_objects)
        required_skills = self.extract_required_skills(task, invariants)

        verification_results = {}
        all_found = True

        for req_obj in required_objects:
            matches = self.semantic_match(req_obj, scene_objects)

            if matches:
                best_match = matches[0]
                verification_results[req_obj] = {
                    'type': 'object',
                    'found': True,
                    'matched_to': best_match[0],
                    'confidence': best_match[1],
                    'method': 'hybrid_match',
                    'accessible': True  
                }

                if best_match[1] < confidence_threshold:
                    verification_results[req_obj]['found'] = False
                    verification_results[req_obj]['reason'] = f'Low confidence: {best_match[1]:.2f}'
                    all_found = False
            else:
                verification_results[req_obj] = {
                    'type': 'object',
                    'found': False,
                    'reason': 'No match in scene',
                    'available_objects': scene_objects[:10]
                }
                all_found = False

        for req_skill in required_skills:
            skill_available = self._skill_match(req_skill, robot_skills)

            verification_results[f'skill_{req_skill}'] = {
                'type': 'skill',
                'found': skill_available,
                'required_skill': req_skill,
                'available_skills': robot_skills if not skill_available else None
            }

            if not skill_available:
                all_found = False

        verification_results['_metadata'] = {
            'floor_plan': floor_plan,
            'scene_key': f'floorplan_{floor_plan}',
            'total_objects': len(required_objects),
            'total_skills': len(required_skills),
            'scene_objects_count': len(scene_objects) if scene_objects else 0,
            'available_skills_count': len(robot_skills) if robot_skills else 0,
            'data_source': data_source_used
        }

        return all_found, verification_results

    def _skill_match(self, required_skill: str, available_skills: List[str]) -> bool:
        """
        Check if required skill matches any available skill.
        Uses fuzzy matching to handle naming variations.
        """
        required_lower = required_skill.lower()

        for available in available_skills:
            available_lower = available.lower()

            if required_lower == available_lower:
                return True

            if required_lower in available_lower or available_lower in required_lower:
                return True

            variations = {
                'navigate': ['goto', 'gotoobject', 'move'],
                'pickup': ['pickupobject', 'pick', 'grab'],
                'place': ['putobject', 'put', 'drop'],
                'open': ['openobject'],
                'close': ['closeobject'],
                'slice': ['sliceobject', 'cut'],
                'toggle': ['switchon', 'switchoff']
            }

            if required_lower in variations:
                if any(var in available_lower for var in variations[required_lower]):
                    return True

        return False

    def get_live_robot_skills(self) -> List[str]:
        """Get robot skills from live AI2-THOR instance."""
        return [
            'GoToObject', 'PickupObject', 'PutObject',
            'OpenObject', 'CloseObject', 'SwitchOn', 'SwitchOff',
            'SliceObject', 'BreakObject', 'ThrowObject', 'PushObject',
            'PullObject', 'DropHandObject'
        ]
    
    def get_live_scene_objects(self, floor_plan: int) -> List[str]:
        try:
            from ai2thor.controller import Controller

            logger.info(f"Creating temporary controller for FloorPlan{floor_plan}")

            temp_controller = Controller(
                scene=f"FloorPlan{floor_plan}",
                width=300,
                height=300,
                headless=True
            )

            objects = [obj['objectType'] for obj in temp_controller.last_event.metadata['objects']]
            unique_objects = sorted(list(set(objects)))

            temp_controller.stop()

            logger.info(f"Got {len(unique_objects)} objects from FloorPlan{floor_plan}")
            return unique_objects

        except Exception as e:
            logger.error(f"Error getting objects from FloorPlan{floor_plan}: {e}")
            return []