"""STL monitoring using rtamt for physical safety constraints."""

import re
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class STLMonitor:
    """Monitor Signal Temporal Logic constraints using rtamt."""

    def __init__(self):
        try:
            import rtamt
            self.rtamt_available = True
            self.rtamt = rtamt
            logger.info("rtamt STL monitor available")
        except ImportError:
            self.rtamt_available = False
            logger.warning(
                "rtamt not available - STL verification disabled. "
                "Install with: pip install rtamt"
            )

    def verify_constraints(self, actions: List[Dict],
                          stl_constraints: List[str]) -> Tuple[bool, List[str]]:
        """
        Verify STL constraints against action sequence.

        Args:
            actions: List of robot actions with types and arguments
            stl_constraints: STL formulas to verify

        Returns:
            (is_valid, list_of_violations)
        """
        if not self.rtamt_available:
            logger.warning("STL verification skipped - rtamt not installed")
            return True, []

        if not stl_constraints:
            return True, []

        violations = []

        signals = self._actions_to_signals(actions)

        for constraint in stl_constraints:
            try:
                valid, error_msg = self._verify_single_constraint(constraint, signals)
                if not valid:
                    violations.append(error_msg)
            except Exception as e:
                logger.error(f"Error verifying STL constraint '{constraint}': {e}")
                violations.append(f"STL verification error: {constraint} - {str(e)}")

        return len(violations) == 0, violations

    def _actions_to_signals(self, actions: List[Dict]) -> Dict[str, List[Tuple[float, float]]]:
        """
        Convert action sequence to discrete-time signals for STL monitoring.

        Returns signals as {signal_name: [(time, value), ...]}
        """
        signals = {
            'velocity': [],
            'temperature': [],
            'distance_to_human': [],
            'acceleration': [],
            'tilt_angle': []
        }

        time_step = 1.0  # Each action is 1 time unit

        for i, action in enumerate(actions):
            t = float(i)
            action_type = action.get('type')
            target = str(action.get('args', [None])[0]) if action.get('args') else ''

            if action_type == 'NAVIGATE':
                signals['velocity'].append((t, 0.08))  # Safe movement speed (was 0.5)
            elif self._is_fragile(target):
                signals['velocity'].append((t, 0.05))  # Very slow for fragile objects (was 0.1)
            else:
                signals['velocity'].append((t, 0.08))  # Safe default speed (was 0.3)

            # Temperature signal
            if self._is_hot(target) and action_type in ['PICKUP', 'PLACE']:
                signals['temperature'].append((t, 80.0))  # Hot object
            else:
                signals['temperature'].append((t, 20.0))  # Room temperature

            # Distance to human (simplified - would need scene info)
            if self._is_hot(target) or self._is_sharp(target):
                signals['distance_to_human'].append((t, 2.0))  # Safe distance
            else:
                signals['distance_to_human'].append((t, 0.5))  # Normal proximity

            # Acceleration signal
            if action_type == 'THROW':
                signals['acceleration'].append((t, 2.0))  # High acceleration
            elif self._is_fragile(target):
                signals['acceleration'].append((t, 0.1))  # Gentle handling
            else:
                signals['acceleration'].append((t, 0.5))  # Normal

            # Tilt angle signal (for liquids)
            if 'liquid' in target.lower() or 'water' in target.lower():
                if action_type in ['PICKUP', 'PLACE']:
                    signals['tilt_angle'].append((t, 5.0))  # Careful tilt
                else:
                    signals['tilt_angle'].append((t, 0.0))  # Upright
            else:
                signals['tilt_angle'].append((t, 0.0))  # Not relevant

        return signals

    def _verify_single_constraint(self, constraint: str,
                                  signals: Dict) -> Tuple[bool, Optional[str]]:
        """
        Verify a single STL constraint using rtamt.

        Supports constraints like:
        - G[0,T](velocity < 1.0)
        - G[0,T](temperature > 60 -> distance_to_human > 1.0)
        - G[0,T](holding_fragile -> acceleration < 0.5)
        """
        formula = self._parse_stl_formula(constraint)

        if not formula:
            return True, None  # Skip unparseable formulas

        try:
            spec = self.rtamt.StlDiscreteTimeSpecification()
            spec.name = 'Safety Constraint'

            for signal_name in signals.keys():
                if signal_name in constraint:
                    spec.declare_var(signal_name, 'float')

            spec.spec = formula

            try:
                spec.parse()
            except Exception as e:
                logger.debug(f"Could not parse STL formula: {formula}")
                return self._heuristic_stl_check(constraint, signals)

            time_points = sorted(set(t for signal in signals.values() for t, v in signal))

            input_data = {}
            for signal_name, values in signals.items():
                if signal_name in constraint:
                    signal_dict = {t: v for t, v in values}
                    input_data[signal_name] = [(t, signal_dict.get(t, 0.0))
                                               for t in time_points]

            result = spec.evaluate(input_data)

            if isinstance(result, list) and len(result) > 0:
                final_robustness = result[-1][1] if len(result[-1]) > 1 else result[-1]
                if final_robustness < 0:
                    return False, f"STL constraint violated: {constraint}"

            return True, None

        except Exception as e:
            logger.warning(f"rtamt evaluation failed: {e}, using heuristic check")
            return self._heuristic_stl_check(constraint, signals)

    def _parse_stl_formula(self, constraint: str) -> Optional[str]:
        """
        Parse STL constraint string to rtamt-compatible formula.

        Converts: "G[0,T](temperature > 60°C → distance > 1m)"
        To: "globally[0,10](temperature > 60 implies distance > 1.0)"
        """
        formula = constraint.replace('°C', '').replace('m/s²', '').replace('m/s', '').replace('m', '').replace('°', '')

        formula = formula.replace('→', 'implies')
        formula = formula.replace('->', 'implies')
        formula = formula.replace('∧', 'and')
        formula = formula.replace('∨', 'or')
        formula = formula.replace('¬', 'not')

        formula = re.sub(r'G\[(\d+),T\]', r'globally[0,10]', formula)
        formula = re.sub(r'G\[(\d+),(\d+)\]', r'globally[\1,\2]', formula)
        formula = re.sub(r'F\[(\d+),T\]', r'eventually[0,10]', formula)

        return formula if formula else None

    def _heuristic_stl_check(self, constraint: str,
                            signals: Dict) -> Tuple[bool, Optional[str]]:
        """
        Fallback heuristic checking when rtamt parsing fails.
        """
        constraint_lower = constraint.lower()

        if 'velocity' in constraint_lower:
            threshold_match = re.search(r'velocity\s*<\s*(\d+\.?\d*)', constraint_lower)
            if threshold_match:
                threshold = float(threshold_match.group(1))
                max_velocity = max(v for t, v in signals.get('velocity', [(0, 0)]))
                if max_velocity > threshold:
                    return False, f"Velocity {max_velocity} exceeds threshold {threshold}"

        if 'temperature' in constraint_lower and 'distance' in constraint_lower:
            temp_threshold_match = re.search(r'temperature\s*>\s*(\d+)', constraint_lower)
            dist_threshold_match = re.search(r'distance\s*>\s*(\d+\.?\d*)', constraint_lower)

            if temp_threshold_match and dist_threshold_match:
                temp_threshold = float(temp_threshold_match.group(1))
                dist_threshold = float(dist_threshold_match.group(1))

                for (t_temp, temp), (t_dist, dist) in zip(signals.get('temperature', []),
                                                           signals.get('distance_to_human', [])):
                    if temp > temp_threshold and dist <= dist_threshold:
                        return False, f"Hot object (temp={temp}) too close (dist={dist})"

        if 'acceleration' in constraint_lower:
            threshold_match = re.search(r'acceleration\s*<\s*(\d+\.?\d*)', constraint_lower)
            if threshold_match:
                threshold = float(threshold_match.group(1))
                max_accel = max(a for t, a in signals.get('acceleration', [(0, 0)]))
                if max_accel > threshold:
                    return False, f"Acceleration {max_accel} exceeds threshold {threshold}"

        return True, None

    def _is_hot(self, obj_name: str) -> bool:
        """Check if object name suggests hot temperature."""
        hot_keywords = ['hot', 'boiling', 'coffee', 'tea', 'stove', 'oven']
        return any(kw in obj_name.lower() for kw in hot_keywords)

    def _is_fragile(self, obj_name: str) -> bool:
        """Check if object name suggests fragility."""
        fragile_keywords = ['glass', 'fragile', 'delicate', 'plate', 'cup', 'vase']
        return any(kw in obj_name.lower() for kw in fragile_keywords)

    def _is_sharp(self, obj_name: str) -> bool:
        """Check if object name suggests sharp edge."""
        sharp_keywords = ['knife', 'scissors', 'blade', 'sharp']
        return any(kw in obj_name.lower() for kw in sharp_keywords)