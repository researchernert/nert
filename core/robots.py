# nert/core/robots.py
"""Robot configurations and capabilities."""

ROBOT_CONFIGURATIONS = [
    {
        'id': 'robot_basic',
        'name': 'BasicManipulator',
        'skills': [
            'navigate_to',
            'pickup',
            'place',
            'open_object',
            'close_object'
        ],
        'constraints': {
            'max_weight': 5.0,  # kg
            'reach': 1.0,  # meters
            'speed': 0.5  # m/s
        }
    },
    {
        'id': 'robot_advanced',
        'name': 'AdvancedAssistant',
        'skills': [
            'navigate_to',
            'pickup',
            'place',
            'open_object',
            'close_object',
            'pour',
            'slice',
            'switch_on',
            'switch_off',
            'push',
            'pull'
        ],
        'constraints': {
            'max_weight': 10.0,
            'reach': 1.2,
            'speed': 1.0,
            'precision': 'high'
        }
    },
    {
        'id': 'robot_kitchen',
        'name': 'KitchenHelper',
        'skills': [
            'navigate_to',
            'pickup',
            'place',
            'open_object',
            'close_object',
            'pour',
            'slice',
            'mix',
            'cook',
            'clean'
        ],
        'constraints': {
            'max_weight': 8.0,
            'reach': 1.0,
            'speed': 0.7,
            'temperature_safe': True
        }
    }
]

def get_robot_skills(robot_id='robot_basic'):
    """Get skills for a specific robot."""
    for robot in ROBOT_CONFIGURATIONS:
        if robot['id'] == robot_id:
            return robot['skills']
    return ROBOT_CONFIGURATIONS[0]['skills']

def get_robot_constraints(robot_id='robot_basic'):
    """Get constraints for a specific robot."""
    for robot in ROBOT_CONFIGURATIONS:
        if robot['id'] == robot_id:
            return robot['constraints']
    return ROBOT_CONFIGURATIONS[0]['constraints']