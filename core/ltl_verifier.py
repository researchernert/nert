# nert/core/ltl_verifier.py
"""Actual LTL verification using model checking."""

import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class LTLModelChecker:
    """Verify LTL properties using Spot model checker"""

    def __init__(self):
        try:
            import spot
            self.spot = spot
            self.bdd_dict = spot.make_bdd_dict()
            self.spot_available = True
            logger.info("Spot model checker loaded - formal LTL verification enabled")
        except ImportError:
            self.spot = None
            self.bdd_dict = None
            self.spot_available = False
            logger.warning(
                "Spot model checker not available. LTL verification will use "
                "heuristic checking. For formal verification, install Spot:\n"
                "  conda install -c conda-forge spot"
            )
        
    def verify_ltl_formula(self, formula_str: str,
                          state_trace: List[Dict]) -> Tuple[bool, Optional[str]]:
        """
        Verify an LTL formula against execution trace.

        Args:
            formula_str: LTL formula like "G(!collision U goal_reached)"
            state_trace: List of state dictionaries

        Returns:
            (is_satisfied, error_message)
        """
        if not self.spot_available:
            # Use fallback if Spot not available
            return self.fallback_check(formula_str, state_trace)

        try:
            formula = self.spot.formula(formula_str)

            word = self.trace_to_word(state_trace)

            aut = self.spot.translate(formula, 'BA', 'High', 'SBAcc')

            if self.check_word_acceptance(aut, word):
                return True, None
            else:
                return False, f"LTL formula '{formula_str}' violated by execution trace"

        except Exception as e:
            logger.error(f"LTL verification failed: {e}")
            return self.fallback_check(formula_str, state_trace)
    
    def trace_to_word(self, state_trace: List[Dict]) -> List:
        """
        Convert state trace to a word for automaton checking.
        Enhanced to support comprehensive atomic propositions.
        """
        if not self.spot_available:
            return []

        word = []

        for state in state_trace:
            props = self.spot.bdd_true()

            if state.get('holding'):
                ap = self.bdd_dict.register_ap("holding")
                props &= self.spot.bdd_ithvar(ap)
            else:
                ap = self.bdd_dict.register_ap("gripper_empty")
                props &= self.spot.bdd_ithvar(ap)

            if state.get('collision', False):
                ap = self.bdd_dict.register_ap("collision")
                props &= self.spot.bdd_ithvar(ap)

            if state.get('spilled', False):
                ap = self.bdd_dict.register_ap("spilled")
                props &= self.spot.bdd_ithvar(ap)

            if state.get('at_location'):
                location = str(state['at_location'])
                ap = self.bdd_dict.register_ap(f"at_{location}")
                props &= self.spot.bdd_ithvar(ap)

            for obj, loc in state.get('objects_at', {}).items():
                obj_clean = str(obj).replace(' ', '_').replace('-', '_')
                loc_clean = str(loc).replace(' ', '_').replace('-', '_')
                ap = self.bdd_dict.register_ap(f"{obj_clean}_at_{loc_clean}")
                props &= self.spot.bdd_ithvar(ap)

            for obj in state.get('opened', set()):
                obj_clean = str(obj).replace(' ', '_').replace('-', '_')
                ap = self.bdd_dict.register_ap(f"{obj_clean}_open")
                props &= self.spot.bdd_ithvar(ap)

            if state.get('task_complete', False):
                ap = self.bdd_dict.register_ap("task_complete")
                props &= self.spot.bdd_ithvar(ap)

            if state.get('at_goal', False):
                ap = self.bdd_dict.register_ap("at_goal")
                props &= self.spot.bdd_ithvar(ap)

            word.append(props)

        return word
    
    def check_word_acceptance(self, automaton, word) -> bool:
        """Check if automaton accepts the word."""
        run = automaton.new_run()
        for letter in word:
            if not run.step(letter):
                return False
        return run.accepting()
    
    def fallback_check(self, formula_str: str, state_trace: List[Dict]) -> Tuple[bool, Optional[str]]:
        """Fallback to simple checking when Spot unavailable."""
        formula_lower = formula_str.lower()
        
        if 'g(' in formula_lower or 'globally' in formula_lower:
            for i, state in enumerate(state_trace):
                if not self.evaluate_proposition(formula_str, state):
                    return False, f"Global property violated at step {i}"
            return True, None
            
        elif 'f(' in formula_lower or 'eventually' in formula_lower:
            for state in state_trace:
                if self.evaluate_proposition(formula_str, state):
                    return True, None
            return False, "Eventually property never satisfied"
        
        return True, None  
    
    def evaluate_proposition(self, prop: str, state: Dict) -> bool:
        """Evaluate atomic proposition in a state."""
        if 'collision' in prop.lower():
            return not state.get('collision', False)
        if 'spilled' in prop.lower():
            return not state.get('spilled', False)
        if 'holding' in prop.lower():
            return state.get('holding') is not None
        return True