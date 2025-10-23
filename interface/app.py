import os
import sys
import time
import json
import yaml
import uuid
import logging
import webbrowser
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from threading import Thread
from typing import Dict, List

if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

llm_client = None
safety_checker = None
invariant_generator = None
grounding_verifier = None
code_generator = None
code_verifier = None
task_executor = None
ai2thor_controller = None
config = None

system_ready = False
ai2thor_ready = False
initialization_error = None

def print_banner():
    print("\n" + "="*55)
    print("🤖  NERT - Neurosymbolic Robot Safety System")
    print("="*55)
    print("Initializing your AI safety assistant...")
    print()

def print_progress(step, total, message, success=True, time_taken=None):
    status = "✓" if success else "✗"
    time_str = f" ({time_taken:.1f}s)" if time_taken else ""
    print(f"[{step}/{total}] {message}...{' '*20} {status}{time_str}")

def init_components():
    global llm_client, safety_checker, invariant_generator
    global grounding_verifier, code_generator, code_verifier, task_executor
    global ai2thor_controller, config, system_ready, ai2thor_ready, initialization_error

    start_time = time.time()

    try:

        step_start = time.time()
        config_path = Path(__file__).parent.parent / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print_progress(1, 6, "Loading configuration", True, time.time() - step_start)

        step_start = time.time()
        from utils.llm_client import LLMClient
        llm_client = LLMClient(config['llm']['model'])

        try:

            test_response = llm_client.call("Say hello", max_tokens=20)
            if not test_response or len(test_response.strip()) == 0:
                raise Exception("API returned empty response")
        except Exception as e:
            print_progress(2, 6, "LLM client setup", False)
            print(f"    API validation error: {str(e)}")
            raise Exception(f"LLM API issue: {str(e)}. Check your API key in .env file")

        print_progress(2, 6, f"LLM client ready ({config['llm']['model']})", True, time.time() - step_start)

        step_start = time.time()
        from core.safety_checker import NeurosymbolicSafetyChecker
        safety_checker = NeurosymbolicSafetyChecker(llm_client, config.get('neural', {}))

        if hasattr(safety_checker, 'confidence_estimator') and safety_checker.confidence_estimator:

            try:
                safety_checker.confidence_estimator.retrieve_similar_examples("test task", k=1)
            except:
                pass

        print_progress(3, 6, "Neurosymbolic safety checker loaded", True, time.time() - step_start)

        step_start = time.time()
        from core.invariant_generator import HybridInvariantGenerator
        from core.grounding_verifier import SemanticGroundingVerifier

        invariant_generator = HybridInvariantGenerator(llm_client)
        grounding_verifier = SemanticGroundingVerifier(llm_client=llm_client)

        if hasattr(grounding_verifier, 'encoder'):

            try:
                grounding_verifier.encoder.encode(["warm up"])
            except:
                pass

        print_progress(4, 6, "Verification engine ready", True, time.time() - step_start)

        step_start = time.time()
        from simulation.code_generator import SafetyConstrainedCodeGenerator
        from simulation.task_executor import TaskExecutor

        code_generator = SafetyConstrainedCodeGenerator(llm_client)
        task_executor = TaskExecutor(use_ai2thor=config.get('ai2thor', {}).get('enabled', False))

        print_progress(5, 6, "Code generation ready", True, time.time() - step_start)

        step_start = time.time()
        try:

            ai2thor_controller = init_ai2thor_controller()
            if ai2thor_controller:
                ai2thor_ready = True
                print_progress(6, 7, "AI2-THOR controller ready", True, time.time() - step_start)
            else:
                ai2thor_ready = False
                print_progress(6, 7, "AI2-THOR controller not available", False, time.time() - step_start)
        except Exception as e:
            ai2thor_ready = False
            print_progress(6, 7, f"AI2-THOR controller failed: {str(e)}", False, time.time() - step_start)

        step_start = time.time()

        log_dir = Path(__file__).parent.parent / 'data' / 'logs' / 'sessions'
        log_dir.mkdir(parents=True, exist_ok=True)

        print_progress(7, 7, "System initialization complete", True, time.time() - step_start)

        total_time = time.time() - start_time
        print()
        print("="*55)
        print(f" All systems ready! ({total_time:.1f}s total)")
        print(" Starting web interface...")
        print("="*55)

        system_ready = True

    except Exception as e:
        initialization_error = str(e)
        system_ready = False
        import sys
        import traceback
        print("="*70, file=sys.stderr, flush=True)
        print(f"INITIALIZATION FAILED: {str(e)}", file=sys.stderr, flush=True)
        print("="*70, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        print("="*70, file=sys.stderr, flush=True)
        print_progress("X", 6, f"Initialization failed: {str(e)}", False)
        print()
        print("Startup failed. Common fixes:")
        print("   • Check environment variables for valid API keys")
        print("   • Ensure config.yaml exists")
        print("   • Check logs above for detailed traceback")
        print()

def init_ai2thor_controller():

    if os.name == 'nt':
        print("[AI2-THOR] AI2-THOR simulation is not available on Windows")
        print("[AI2-THOR] Recommendation: Use Linux with X11 display support for AI2-THOR simulation")
        print("[AI2-THOR] Note: All other NERT features (safety checking, planning, code generation) work normally on Windows")
        return None

    try:

        print("[AI2-THOR] Attempting to import ai2thor...")
        import ai2thor
        from ai2thor.controller import Controller
        print(f"[AI2-THOR] AI2-THOR version: {ai2thor.__version__}")

        print("[AI2-THOR] Creating controller with minimal parameters...")
        controller = Controller(
            height=1000,
            width=1000
        )
        print("[AI2-THOR] ✓ Controller created successfully")

        print("[AI2-THOR] Resetting to FloorPlan1 (Kitchen scene)...")
        controller.reset("FloorPlan1")
        print("[AI2-THOR] ✓ Scene reset completed")

        scene_metadata = controller.last_event.metadata
        scene_name = scene_metadata.get('sceneName', 'Unknown')
        object_count = len(scene_metadata.get('objects', []))

        print(f"[AI2-THOR]  Initialization successful!")
        print(f"[AI2-THOR]    Scene: {scene_name}")
        print(f"[AI2-THOR]    Objects in scene: {object_count}")
        print(f"[AI2-THOR]    Window size: 1000x1000")

        return controller

    except ImportError as e:
        print(f"[AI2-THOR] Import failed: {e}")
        print(f"[AI2-THOR] Install with: pip install ai2thor")
        return None
    except Exception as e:
        print(f"[AI2-THOR] Initialization failed: {e}")
        print(f"[AI2-THOR] Error type: {type(e).__name__}")
        print(f"[AI2-THOR] Recommendation: Ensure you're running on Linux with X11 display support")
        return None

def launch_browser():
    time.sleep(2)
    try:
        webbrowser.open('http://localhost:5000')
        print("Browser opened automatically")
    except Exception as e:
        print(f"Could not auto-open browser: {e}")
        print("Please manually open: http://localhost:5000")

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5000", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

_init_started = False

def _kickoff_init():
    global _init_started
    if not _init_started:
        _init_started = True
        Thread(target=init_components, daemon=True).start()

_kickoff_init()

@app.route('/')
def index():
    if not system_ready:
        return render_template('error.html',
                             error="System not ready",
                             details=initialization_error), 503
    return render_template('index.html')

@app.route('/health')
def health():
    try:
        _kickoff_init()
    except Exception:
        pass

    return jsonify({
        'status': 'ok',
        'ready': bool(system_ready),
        'ai2thor_ready': bool(ai2thor_ready),
        'error': initialization_error
    }), 200

@app.route('/status')
def status():
    return jsonify({
        'ready': system_ready,
        'ai2thor_ready': ai2thor_ready,
        'error': initialization_error,
        'components': {
            'llm_client': llm_client is not None,
            'safety_checker': safety_checker is not None,
            'neural_model': (safety_checker.has_confidence if safety_checker else False),
            'ai2thor_controller': ai2thor_controller is not None
        }
    })

@app.route('/ai2thor/status')
def ai2thor_status():
    if not ai2thor_controller:
        return jsonify({
            'connected': False,
            'error': 'AI2-THOR controller not initialized'
        }), 503

    try:

        scene_info = ai2thor_controller.last_event.metadata
        return jsonify({
            'connected': True,
            'scene': scene_info.get('sceneName', 'Unknown'),
            'objects_count': len(scene_info.get('objects', [])),
            'agent_position': scene_info.get('agent', {}).get('position')
        })
    except Exception as e:
        return jsonify({
            'connected': False,
            'error': f'AI2-THOR communication error: {str(e)}'
        }), 500

@app.route('/process', methods=['POST'])
def process_task():
    if not system_ready:
        return jsonify({
            'error': 'System not ready',
            'details': initialization_error,
            'final_status': 'SYSTEM_ERROR'
        }), 503

    start_time = time.time()
    data = request.get_json()

    if not data:
        return jsonify({
            'error': 'No JSON data provided',
            'final_status': 'REQUEST_ERROR'
        }), 400

    task_text = data.get('task', '').strip()
    scene_description = data.get('scene', '').strip()
    selected_model = data.get('model', 'gpt-4o')
    floor_plan = data.get('floor_plan', 1)
    disabled_skills = data.get('disabled_skills', [])
    mode = data.get('mode', 'nert')

    scene_objects_from_request = data.get('scene_objects', None)

    if not task_text:
        return jsonify({
            'error': 'No task provided',
            'final_status': 'REQUEST_ERROR'
        }), 400

    MAX_TASK_LENGTH = 1000
    if len(task_text) > MAX_TASK_LENGTH:
        return jsonify({
            'error': f'Task description too long. Maximum {MAX_TASK_LENGTH} characters allowed. Current length: {len(task_text)} characters.',
            'final_status': 'REQUEST_ERROR'
        }), 400

    MAX_SCENE_LENGTH = 2000
    if scene_description and len(scene_description) > MAX_SCENE_LENGTH:
        return jsonify({
            'error': f'Scene description too long. Maximum {MAX_SCENE_LENGTH} characters allowed. Current length: {len(scene_description)} characters.',
            'final_status': 'REQUEST_ERROR'
        }), 400

    try:
        from core.llm_clients import LLMClientFactory
        dynamic_llm_client = LLMClientFactory.create_client(selected_model)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🤖 Using model: {selected_model}")
    except Exception as e:
        return jsonify({
            'error': f'Failed to initialize model {selected_model}: {str(e)}',
            'final_status': 'MODEL_ERROR'
        }), 400

    timestamp = datetime.now().strftime('%H:%M:%S')
    task_display = task_text[:50] + ('...' if len(task_text) > 50 else '')
    print(f"\n[{timestamp}] New task: \"{task_display}\"")

    session_id = str(uuid.uuid4())
    result = {
        'session_id': session_id,
        'timestamp': datetime.now().isoformat(),
        'task': task_text,
        'scene': scene_description,
        'model': selected_model,
        'floor_plan': floor_plan,
        'disabled_skills': disabled_skills,
        'mode': mode,
        'stages': {},
        'processing_times': {}
    }

    if mode == 'base_llm':

        logger.info("Running in BASE LLM mode (no NERT)")

        scene_objects = scene_objects_from_request
        if not scene_objects or len(scene_objects) == 0:

            try:
                cache_path = Path(__file__).parent.parent / 'data' / 'scene_cache.json'
                if cache_path.exists():
                    with open(cache_path, 'r') as f:
                        cache = json.load(f)
                        scene_key = f"floorplan_{floor_plan}"
                        if scene_key in cache:
                            scene_objects = cache[scene_key].get('objects', [])
            except Exception as e:
                logger.warning(f"Could not load scene cache: {e}")
                scene_objects = []

        all_skills = [
            'GoToObject', 'PickupObject', 'PutObject',
            'OpenObject', 'CloseObject', 'SliceObject',
            'BreakObject', 'SwitchOn', 'SwitchOff',
            'ThrowObject', 'PushObject', 'PullObject',
            'DropHandObject'
        ]
        robot_skills = [skill for skill in all_skills if skill not in disabled_skills]

        from core.base_llm_checker import run_base_llm_pipeline
        result = run_base_llm_pipeline(
            task=task_text,
            scene=scene_description,
            floor_plan=floor_plan,
            model=selected_model,
            objects=scene_objects,
            skills=robot_skills,
            llm_client=dynamic_llm_client,
            disabled_skills=disabled_skills
        )

        result['session_id'] = session_id
        result['timestamp'] = datetime.now().isoformat()
        result['task'] = task_text
        result['scene'] = scene_description
        result['model'] = selected_model
        result['floor_plan'] = floor_plan
        result['disabled_skills'] = disabled_skills

        save_session_log(result)

        return jsonify(result)

    try:

        stage_start = time.time()
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Checking safety guidelines...")

        try:

            from core.safety_checker import NeurosymbolicSafetyChecker
            dynamic_safety_checker = NeurosymbolicSafetyChecker(dynamic_llm_client, config.get('neural', {}))

            print(f"[{timestamp}] About to call safety_checker.check() with model {selected_model}: '{task_text[:100]}...'")
            safety_result = dynamic_safety_checker.check(task_text, scene_description)
            print(f"[{timestamp}] Safety checker returned: {safety_result}")
            stage_time = time.time() - stage_start
            result['processing_times']['safety'] = stage_time

            result['stages']['safety'] = {
                'passed': safety_result.decision == "ACCEPT",
                'decision': safety_result.decision,
                'symbolic': safety_result.symbolic_result,
                'symbolic_trace': safety_result.symbolic_trace,
                'neural_confidence': safety_result.neural_confidence,
                'explanation': safety_result.explanation,
                'nearest_neighbors': safety_result.nearest_neighbors[:3] if safety_result.nearest_neighbors else []
            }

            timestamp = datetime.now().strftime('%H:%M:%S')
            if safety_result.decision == "ACCEPT":
                print(f"[{timestamp}] Task approved - no safety concerns found ({stage_time:.1f}s)")
            else:
                print(f"[{timestamp}] Task rejected - safety concerns detected ({stage_time:.1f}s)")
                result['final_status'] = "REJECTED_SAFETY"
                save_session_log(result)
                return jsonify(convert_numpy_types(result))

        except Exception as e:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] Safety check failed: {str(e)}")
            print(f"[{timestamp}] Error details: {repr(e)}")
            if hasattr(e, '__traceback__'):
                print(f"[{timestamp}] Traceback: {traceback.format_exc()}")
            raise Exception(f"Safety verification failed: {str(e)}")

        stage_start = time.time()
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Planning execution steps...")

        try:

            from core.invariant_generator import HybridInvariantGenerator
            dynamic_invariant_generator = HybridInvariantGenerator(dynamic_llm_client)

            scene_objects = None
            robot_skills = None

            try:
                cache_path = Path(__file__).parent.parent / 'data' / 'scene_cache.json'
                if cache_path.exists():
                    with open(cache_path, 'r') as f:
                        cache = json.load(f)
                        scene_key = f"floorplan_{floor_plan}"
                        if scene_key in cache:
                            scene_objects = cache[scene_key].get('objects', [])
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]  Could not load scene cache: {e}")

            if not scene_objects:
                if ai2thor_controller:
                    try:
                        ai2thor_controller.reset(scene=f"FloorPlan{floor_plan}")
                        objects = ai2thor_controller.last_event.metadata['objects']
                        scene_objects = sorted(list(set([obj['objectType'] for obj in objects])))
                    except:
                        pass

            all_skills = [
                'GoToObject', 'PickupObject', 'PutObject',
                'OpenObject', 'CloseObject', 'SliceObject',
                'BreakObject', 'SwitchOn', 'SwitchOff',
                'ThrowObject', 'PushObject', 'PullObject',
                'DropHandObject'
            ]
            robot_skills = [skill for skill in all_skills if skill not in disabled_skills]

            invariants = dynamic_invariant_generator.generate(
                task_text,
                safety_result.neural_confidence,
                floor_plan=floor_plan,
                scene_objects=scene_objects,
                robot_skills=robot_skills
            )
            stage_time = time.time() - stage_start
            result['processing_times']['invariants'] = stage_time

            result['stages']['invariants'] = {
                'pddl_preconditions': invariants.pddl_preconditions,
                'pddl_postconditions': invariants.pddl_postconditions,
                'ltl_invariants': invariants.ltl_invariants,
                'stl_constraints': invariants.stl_constraints,
                'llm_contextual': invariants.llm_contextual
            }

            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] Execution plan ready ({stage_time:.1f}s)")

        except Exception as e:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] Planning failed: {str(e)}")
            raise Exception(f"Planning failed: {str(e)}")

        stage_start = time.time()
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Checking available resources...")

        try:

            from core.grounding_verifier import SemanticGroundingVerifier
            dynamic_grounding_verifier = SemanticGroundingVerifier(llm_client=dynamic_llm_client)

            grounding_passed, grounding_details = dynamic_grounding_verifier.verify(
                task_text,
                invariants.__dict__,
                floor_plan=floor_plan
            )
            stage_time = time.time() - stage_start
            result['processing_times']['grounding'] = stage_time

            result['stages']['grounding'] = {
                'passed': grounding_passed,
                'details': grounding_details
            }

            if grounding_passed:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] All resources available ({stage_time:.1f}s)")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Missing required resources ({stage_time:.1f}s)")
                result['final_status'] = "REJECTED_GROUNDING"
                save_session_log(result)
                return jsonify(convert_numpy_types(result))

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Resource check failed: {str(e)}")
            raise Exception(f"Resource verification failed: {str(e)}")

        stage_start = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating execution code...")

        try:

            from simulation.code_generator import SafetyConstrainedCodeGenerator
            dynamic_code_generator = SafetyConstrainedCodeGenerator(dynamic_llm_client)

            scene_objects = list(grounding_details.keys()) if grounding_details else []
            generated_code = dynamic_code_generator.generate(task_text, invariants.__dict__, scene_objects)
            stage_time = time.time() - stage_start
            result['processing_times']['code_generation'] = stage_time

            result['stages']['code_generation'] = {
                'code': generated_code
            }

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Code generated ({stage_time:.1f}s)")

            stage_start = time.time()
            # Create fresh code verifier per request to prevent Z3 solver state accumulation
            from simulation.code_verifier import InvariantCodeVerifier
            dynamic_code_verifier = InvariantCodeVerifier()

            valid, violations, methods_used = dynamic_code_verifier.verify(
                generated_code,
                invariants.__dict__
            )
            stage_time = time.time() - stage_start
            result['processing_times']['verification'] = stage_time

            result['stages']['verification'] = {
                'passed': valid,
                'violations': violations,
                'methods_used': methods_used
            }

            if valid:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Code verified ({stage_time:.1f}s)")

                methods_passed = [k for k, v in methods_used.items()
                                if v == 'passed' or v == 'completed']
                if methods_passed:
                    print(f"    Verification methods: {', '.join(methods_passed)}")

                if ai2thor_controller and ai2thor_ready:
                    stage_start = time.time()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🤖 Executing task in AI2-THOR...")

                    try:
                        execution_result = execute_task_in_ai2thor(generated_code, task_text, floor_plan)
                        stage_time = time.time() - stage_start
                        result['processing_times']['execution'] = stage_time

                        result['stages']['execution'] = {
                            'passed': execution_result.get('success', False),
                            'details': execution_result.get('details', ''),
                            'error': execution_result.get('error') if not execution_result.get('success', False) else None,
                            'ai2thor_screenshot': execution_result.get('screenshot_path'),
                            'video_folder': execution_result.get('video_folder')
                        }

                        if execution_result.get('success', False):
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Task executed successfully in AI2-THOR ({stage_time:.1f}s)")
                            result['final_status'] = "EXECUTED_SUCCESS"
                        else:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Task execution failed in AI2-THOR ({stage_time:.1f}s)")
                            result['final_status'] = "REJECTED_EXECUTION"

                    except Exception as e:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] AI2-THOR execution failed: {str(e)}")
                        result['stages']['execution'] = {
                            'passed': False,
                            'error': str(e)
                        }
                        result['final_status'] = "REJECTED_EXECUTION"
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]  AI2-THOR not available - code verified but not executed")
                    result['final_status'] = "VERIFIED_READY"
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Code verification failed ({stage_time:.1f}s)")
                result['final_status'] = "REJECTED_VERIFICATION"

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Code generation failed: {str(e)}")
            raise Exception(f"Code generation failed: {str(e)}")

        result['final_status'] = "VERIFIED_READY"

        total_time = time.time() - start_time
        result['total_processing_time'] = total_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 Task processing complete! ({total_time:.1f}s total)")

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing failed: {str(e)}")

        result['error'] = str(e)
        result['error_trace'] = error_trace.split('\n')
        result['final_status'] = "ERROR"
        result['total_processing_time'] = time.time() - start_time

    save_session_log(result)

    clean_result = convert_numpy_types(result)
    return jsonify(clean_result)

@app.route('/test')
def test_endpoint():
    return jsonify({
        'status': 'OK',
        'message': 'NERT API is working',
        'timestamp': datetime.now().isoformat(),
        'system_ready': system_ready
    })

@app.route('/models')
def get_available_models():
    try:
        from core.llm_clients import LLMClientFactory

        models = LLMClientFactory.get_available_models()
        api_keys = LLMClientFactory.check_api_keys()

        available_models = {}
        for model_id, model_info in models.items():
            provider = model_info['provider']
            if api_keys.get(provider, False):
                available_models[model_id] = model_info

        return jsonify({
            'models': available_models,
            'api_keys_available': api_keys,
            'default_model': 'gpt-4o'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-llm')
def test_llm():
    if not system_ready or not llm_client:
        return jsonify({'error': 'System not ready'}), 503

    try:

        response = llm_client.call("Say 'Hello'", max_tokens=10)
        return jsonify({
            'status': 'LLM working',
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': f'LLM test failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/get_scene_objects')
def get_scene_objects():
    floor_plan = request.args.get('floor_plan', 1, type=int)

    try:

        if ai2thor_controller:

            ai2thor_controller.reset(scene=f"FloorPlan{floor_plan}")
            objects = ai2thor_controller.last_event.metadata['objects']

            object_types = sorted(list(set([obj['objectType'] for obj in objects])))

            return jsonify({
                'objects': object_types,
                'floor_plan': floor_plan
            })
        else:

            mock_objects = {
                'kitchen': ['Apple', 'Book', 'Bottle', 'Bowl', 'Bread', 'Cabinet',
                           'CoffeeMachine', 'CounterTop', 'Cup', 'Fridge', 'Knife',
                           'Microwave', 'Pan', 'Plate', 'Sink', 'Stove', 'Toaster'],
                'living': ['ArmChair', 'Book', 'CoffeeTable', 'FloorLamp', 'Laptop',
                          'Painting', 'RemoteControl', 'Sofa', 'Television'],
                'bedroom': ['Bed', 'Book', 'Desk', 'DeskLamp', 'Dresser', 'Mirror',
                           'Painting', 'Pillow'],
                'bathroom': ['Bathtub', 'Cabinet', 'Faucet', 'Mirror', 'Sink',
                            'SoapBar', 'Toilet', 'Towel']
            }

            if floor_plan <= 30:
                room_type = 'kitchen'
            elif floor_plan <= 230:
                room_type = 'living'
            elif floor_plan <= 330:
                room_type = 'bedroom'
            else:
                room_type = 'bathroom'

            return jsonify({
                'objects': mock_objects.get(room_type, []),
                'floor_plan': floor_plan
            })

    except Exception as e:
        logger.error(f"Error getting scene objects: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_scene_objects', methods=['POST'])
def update_scene_objects():
    data = request.json
    floor_plan = data.get('floor_plan', 1)
    removed_objects = data.get('removed_objects', [])

    try:

        return jsonify({
            'status': 'success',
            'floor_plan': floor_plan,
            'removed_objects': removed_objects,
            'message': f'Scene updated: {len(removed_objects)} objects marked for removal'
        })
    except Exception as e:
        logger.error(f"Error updating scene: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/process', '/status', '/test'],
        'final_status': 'NOT_FOUND'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'details': str(error),
        'final_status': 'SERVER_ERROR'
    }), 500

@app.route('/static/screenshots/<filename>')
def serve_screenshot(filename):
    screenshot_dir = Path(__file__).parent.parent / 'data' / 'screenshots'
    return send_from_directory(screenshot_dir, filename)

@app.route('/static/videos/<path:filepath>')
def serve_video(filepath):
    """Serve video files from data/videos directory with error handling"""
    try:
        videos_dir = Path(__file__).parent.parent / 'data' / 'videos'
        video_path = videos_dir / filepath

        # Check if file exists
        if not video_path.exists():
            logger.error(f"Video file not found: {filepath}")
            return jsonify({
                'error': 'Video file not found',
                'filepath': filepath
            }), 404

        # Check if file is actually a video
        if not filepath.endswith(('.mp4', '.webm', '.avi', '.mov')):
            logger.error(f"Invalid video format requested: {filepath}")
            return jsonify({
                'error': 'Invalid video format',
                'filepath': filepath
            }), 400

        return send_from_directory(videos_dir, filepath, mimetype='video/mp4')

    except Exception as e:
        logger.error(f"Error serving video {filepath}: {e}")
        return jsonify({
            'error': 'Failed to serve video',
            'details': str(e)
        }), 500

def execute_task_in_ai2thor(generated_code: str, task_text: str, floor_plan: int = 1) -> Dict:
    if not ai2thor_controller:
        return {
            'success': False,
            'error': 'AI2-THOR controller not available'
        }

    try:

        from simulation.ai2thor_actions import AI2THORActionExecutor
        from simulation.safety_monitor import SafetyMonitor

        session_id = str(uuid.uuid4())[:8]
        executor = AI2THORActionExecutor(ai2thor_controller, enable_video=True, session_id=session_id)

        ai2thor_controller.reset(scene=f"FloorPlan{floor_plan}")

        if safety_checker:
            monitor = SafetyMonitor(ai2thor_controller, safety_checker)
            executor.set_monitor_callback(monitor.check_action_safety)
        else:
            monitor = None

        initial_event = ai2thor_controller.last_event
        initial_objects = {obj['objectId']: obj for obj in initial_event.metadata.get('objects', [])}

        action_lines = [line.strip() for line in generated_code.strip().split('\n') if line.strip()]

        actions = []
        for line in action_lines:

            if '(' in line and ')' in line:

                action = line.split('#')[0].strip()
                if action:
                    actions.append(action)

        if not actions:
            return {
                'success': False,
                'error': 'No valid actions found in generated code'
            }

        print(f"[AI2-THOR] Executing {len(actions)} actions for task: {task_text[:50]}...")

        execution_result = executor.execute_action_sequence(actions)

        screenshot_path = save_ai2thor_screenshot()

        final_event = ai2thor_controller.last_event
        final_objects = {obj['objectId']: obj for obj in final_event.metadata.get('objects', [])}

        scene_changes = analyze_scene_changes(initial_objects, final_objects)

        video_info = execution_result.get('video', {})
        video_folder_info = None
        if video_info and 'output_folder' in video_info:
            folder_path = Path(video_info['output_folder'])
            if folder_path.exists():
                files = []
                for file in folder_path.rglob('*'):
                    if file.is_file():
                        rel_path = file.relative_to(folder_path)
                        file_size_mb = file.stat().st_size / (1024 * 1024)
                        files.append({
                            'name': str(rel_path),
                            'size_mb': round(file_size_mb, 2)
                        })

                video_folder_info = {
                    'folder_path': str(folder_path.resolve()),
                    'total_frames': video_info.get('total_frames', 0),
                    'duration': video_info.get('duration', 0),
                    'files': files
                }

        result = {
            'success': execution_result['success'],
            'details': execution_result.get('error') or f"Executed {len(execution_result['executed_actions'])}/{len(actions)} actions",
            'screenshot_path': screenshot_path,
            'executed_actions': execution_result['executed_actions'],
            'total_actions': len(actions),
            'scene_changes': scene_changes,
            'video_folder': video_folder_info 
        }

        if monitor:
            result['safety_interventions'] = monitor.get_intervention_log()

        return result

    except ImportError as e:
        return {
            'success': False,
            'error': f"Could not import AI2-THOR executor: {str(e)}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Execution failed: {str(e)}"
        }

def analyze_scene_changes(initial_objects: Dict, final_objects: Dict) -> Dict:
    changes = {
        'objects_moved': [],
        'objects_picked_up': [],
        'objects_placed': [],
        'state_changes': []
    }

    try:
        for obj_id, initial_obj in initial_objects.items():
            if obj_id in final_objects:
                final_obj = final_objects[obj_id]

                initial_pos = initial_obj.get('position', {})
                final_pos = final_obj.get('position', {})
                if initial_pos != final_pos:
                    changes['objects_moved'].append({
                        'object': obj_id,
                        'from': initial_pos,
                        'to': final_pos
                    })

                if initial_obj.get('isOpen') != final_obj.get('isOpen'):
                    changes['state_changes'].append({
                        'object': obj_id,
                        'change': 'opened' if final_obj.get('isOpen') else 'closed'
                    })

                if initial_obj.get('isToggled') != final_obj.get('isToggled'):
                    changes['state_changes'].append({
                        'object': obj_id,
                        'change': 'turned on' if final_obj.get('isToggled') else 'turned off'
                    })

    except Exception as e:
        logger.warning(f"Error analyzing scene changes: {e}")

    return changes

def parse_robot_code(code: str) -> List[Dict]:
    actions = []
    lines = code.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if 'GoToObject' in line:
            obj_name = extract_object_from_line(line)
            actions.append({'action': 'navigate', 'object': obj_name})
        elif 'PickupObject' in line:
            obj_name = extract_object_from_line(line)
            actions.append({'action': 'pickup', 'object': obj_name})
        elif 'PutObject' in line:
            obj_name = extract_object_from_line(line)
            actions.append({'action': 'place', 'object': obj_name})

    return actions

def extract_object_from_line(line: str) -> str:

    import re
    match = re.search(r'"([^"]*)"', line)
    if match:
        return match.group(1)
    return "Unknown"

def execute_ai2thor_action(step: Dict) -> Dict:
    try:
        action_type = step['action']
        obj_name = step.get('object', '')

        scene_objects = ai2thor_controller.last_event.metadata['objects']
        object_ids = [obj['objectId'] for obj in scene_objects]

        if action_type == 'navigate':

            target_obj_id = None
            for obj_id in object_ids:
                if obj_name.lower() in obj_id.lower():
                    target_obj_id = obj_id
                    break

            if not target_obj_id:
                return {
                    'success': False,
                    'error': f'Object "{obj_name}" not found in scene. Available: {object_ids[:5]}...'
                }

            event = ai2thor_controller.step(
                action="MoveAhead",
                moveMagnitude=0.25
            )

            success = event.metadata['lastActionSuccess']
            error_msg = event.metadata.get('errorMessage', '') if not success else None

            return {
                'success': success,
                'error': error_msg,
                'target_object': target_obj_id
            }

        elif action_type == 'pickup':

            target_obj_id = None
            for obj_id in object_ids:
                if obj_name.lower() in obj_id.lower():
                    target_obj_id = obj_id
                    break

            if not target_obj_id:
                return {
                    'success': False,
                    'error': f'Cannot pickup "{obj_name}" - object not found in scene'
                }

            event = ai2thor_controller.step(
                action="PickupObject",
                objectId=target_obj_id,
                forceAction=True
            )

            success = event.metadata['lastActionSuccess']
            error_msg = event.metadata.get('errorMessage', '') if not success else None

            return {
                'success': success,
                'error': error_msg,
                'object_id': target_obj_id
            }

        elif action_type == 'place':

            receptacle_obj_id = None
            for obj_id in object_ids:
                if obj_name.lower() in obj_id.lower():
                    receptacle_obj_id = obj_id
                    break

            if not receptacle_obj_id:
                return {
                    'success': False,
                    'error': f'Cannot place in "{obj_name}" - receptacle not found'
                }

            event = ai2thor_controller.step(
                action="PutObject",
                objectId=receptacle_obj_id,
                forceAction=True
            )

            success = event.metadata['lastActionSuccess']
            error_msg = event.metadata.get('errorMessage', '') if not success else None

            return {
                'success': success,
                'error': error_msg,
                'receptacle_id': receptacle_obj_id
            }

        else:
            return {
                'success': False,
                'error': f'Unknown action type: {action_type}. Supported: navigate, pickup, place'
            }

    except Exception as e:
        return {
            'success': False,
            'error': f'Action execution failed: {str(e)}',
            'exception_type': type(e).__name__
        }

def save_ai2thor_screenshot() -> str:
    try:

        screenshot_dir = Path(__file__).parent.parent / 'data' / 'screenshots'
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        screenshot_path = screenshot_dir / f'ai2thor_{timestamp}.png'

        frame = ai2thor_controller.last_event.frame

        from PIL import Image
        img = Image.fromarray(frame)

        img.save(screenshot_path, 'PNG', optimize=True)

        scene_name = ai2thor_controller.last_event.metadata.get('sceneName', 'Unknown')
        print(f"[AI2-THOR] Screenshot saved: {screenshot_path.name}")
        print(f"[AI2-THOR] Scene: {scene_name}, Size: {img.size}")

        return screenshot_path.name

    except Exception as e:
        print(f"[AI2-THOR] Screenshot failed: {e}")
        print(f"[AI2-THOR] Error type: {type(e).__name__}")
        return None

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_session_log(result: dict):
    try:
        session_dir = Path(__file__).parent.parent / 'data' / 'logs' / 'sessions'
        session_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{result['session_id']}.json"

        clean_result = convert_numpy_types(result)
        with open(session_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(clean_result, f, indent=2, default=str, ensure_ascii=False)

    except Exception as e:
        print(f"Could not save session log: {e}")

if __name__ == '__main__':
    try:
        print_banner()

        init_components()

        if system_ready:

            is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
            if not is_railway:
                browser_thread = Thread(target=launch_browser, daemon=True)
                browser_thread.start()

            port_display = os.environ.get('PORT', 5000)
            print(f"🚀 Server starting on port {port_display}")
            print("💡 Press Ctrl+C to stop")
            print()

            log = logging.getLogger('werkzeug')
            log.setLevel(logging.WARNING)

            port = int(os.environ.get('PORT', 5000))
            host = os.environ.get('HOST', '0.0.0.0')

            app.run(
                host=host,
                port=port,
                debug=False,
                use_reloader=False
            )
        else:
            print("Cannot start - initialization failed")
            print(f"Error: {initialization_error}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 NERT stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n Failed to start NERT: {e}")
        print("\n Troubleshooting:")
        print("   1. Check .env file has valid API keys")
        print("   2. Run: pip install -r requirements.txt")
        print("   3. Ensure config.yaml exists")
        sys.exit(1)