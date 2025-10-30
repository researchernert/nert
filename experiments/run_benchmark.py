# nert/experiments/run_benchmark.py - Updated with better error handling

import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import time
from tqdm import tqdm
import sys
import traceback

from core.safety_checker import NeurosymbolicSafetyChecker
from core.llm_clients import LLMClientFactory


class BenchmarkEvaluator:
    """Evaluate safety classifier on benchmark dataset (supports NERT and BASE_LLM modes)."""

    def __init__(self, config_path: str = "config.yaml", temperature: float = 0.0, mode: str = 'nert'):
        self.load_config(config_path)
        self.temperature = temperature
        self.mode = mode
        self.model_name = self.config['llm']['model']
        self.results = []
        self.debug_mode = True  
        print(f"Benchmark initialized with model: {self.model_name}, temperature: {temperature}, mode: {mode.upper()}")

        print("Pre-loading LLM client...")
        self.shared_llm_client = LLMClientFactory.create_client(self.model_name)
        print(f"LLM client loaded successfully: {self.model_name}")

        # Mode-specific initialization
        if mode == 'nert':
            print("Pre-loading neural confidence estimator...")
            self.shared_confidence_estimator = self._load_confidence_estimator()
            if self.shared_confidence_estimator:
                print("Confidence estimator loaded successfully and ready for parallel use")
            else:
                print("WARNING: Confidence estimator not available - neural analysis will be disabled")

            print("Pre-loading NERT safety checker...")
            self.shared_safety_checker = NeurosymbolicSafetyChecker(
                self.shared_llm_client,
                self.config.get('neural', {}),
                temperature=self.temperature,
                shared_confidence_estimator=self.shared_confidence_estimator
            )
            print("NERT safety checker loaded successfully and ready for parallel use")

        elif mode == 'base_llm':
            print("BASE_LLM mode: No pre-loading needed (uses stateless function)")
            self.shared_safety_checker = None  # Not used in BASE_LLM mode
            self.shared_confidence_estimator = None

        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes: ['nert', 'base_llm']")

    def load_config(self, path: str):
        """Load configuration."""
        import yaml
        config_file = Path(path)
        if not config_file.exists():
            config_file = Path('..') / path

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

    def _load_confidence_estimator(self):
        """
        Load confidence estimator once for shared use across all workers.
        """
        try:
            from models.contrastive.contrastive_retrieval import SafetyConfidenceEstimator
            estimator = SafetyConfidenceEstimator(
                self.config.get('neural', {}).get('model_path', 'models/trained_encoder.pt')
            )
            return estimator
        except Exception as e:
            try:
                error_msg = str(e)
            except UnicodeEncodeError:
                error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
            print(f"Could not load confidence estimator: {error_msg}")
            print("Benchmark will run without neural confidence analysis")
            return None

    def _normalize_result(self, raw_result, task_text: str, ground_truth: str, task_id: int, processing_time: float) -> Dict:
        """
        Convert raw classifier result (NERT or BASE_LLM) to standard evaluation format.
        """
        if self.mode == 'nert':
            prediction = "safe" if raw_result.decision == "ACCEPT" else "unsafe"

            return {
                'task_id': task_id,
                'task': task_text,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'processing_time': processing_time,
                'correct': prediction == ground_truth,
                'mode': 'nert',
                'decision': raw_result.decision,
                'confidence': raw_result.neural_confidence,
                'symbolic_result': raw_result.symbolic_result,
                'support_ratio': raw_result.support_ratio,
                'explanation': raw_result.explanation
            }

        elif self.mode == 'base_llm':
            prediction = raw_result['prediction']
            has_error = not raw_result.get('success', True)

            result = {
                'task_id': task_id,
                'task': task_text,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'processing_time': processing_time,
                'correct': prediction == ground_truth,
                'mode': 'base_llm',
                'has_error': has_error
            }

            if has_error:
                result['error'] = raw_result.get('error', 'Unknown error')
                result['error_type'] = raw_result.get('error_type', 'Exception')

            return result

    def evaluate_single_task(self, task_data: Dict) -> Dict:
        """Evaluate a single task (works for both NERT and BASE_LLM modes)."""
        start_time = time.time()

        task_text = task_data.get('task_prompt', '')
        scene_description = task_data.get('scene_description', '')
        ground_truth = task_data.get('ground_truth', task_data.get('safety_classification', 'unknown'))
        task_id = task_data.get('id', 0)

        if not task_text:
            print(f"Warning: Empty task_prompt in data: {task_data}")

        try:
            if self.debug_mode:
                print(f"\n=== DEBUGGING FIRST TASK ({self.mode.upper()} MODE) ===")
                print(f"Task: {task_text}")
                print(f"Scene: {scene_description}")
                print(f"Ground truth: {ground_truth}")
                print(f"Temperature: {self.temperature}")
                self.debug_mode = False

            if self.mode == 'nert':
                raw_result = self.shared_safety_checker.check(task_text, scene_description)

            elif self.mode == 'base_llm':
                from core.base_llm_checker import check_safety_base_llm
                raw_result = check_safety_base_llm(
                    task=task_text,
                    scene_description=scene_description,
                    llm_client=self.shared_llm_client,
                    temperature=self.temperature
                )

            processing_time = time.time() - start_time

            evaluation = self._normalize_result(
                raw_result=raw_result,
                task_text=task_text,
                ground_truth=ground_truth,
                task_id=task_id,
                processing_time=processing_time
            )

            if task_id == 0:
                print(f"First task evaluation successful:")
                print(f"  Prediction: {evaluation['prediction']}")
                print(f"  Ground truth: {evaluation['ground_truth']}")
                print(f"  Correct: {evaluation['correct']}")
                if self.mode == 'nert':
                    print(f"  Decision: {evaluation['decision']}")
                    print(f"  Confidence: {evaluation['confidence']:.2f}")
                    print(f"  Symbolic: {evaluation['symbolic_result']}")

            return evaluation

        except Exception as e:
            print(f"\n❌ Error in evaluate_single_task ({self.mode} mode): {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

            result = {
                'task_id': task_id,
                'task': task_text,
                'ground_truth': ground_truth,
                'prediction': 'error',
                'processing_time': time.time() - start_time,
                'correct': False,
                'mode': self.mode
            }

            if self.mode == 'base_llm':
                result['has_error'] = True
                result['error'] = str(e)
                result['error_type'] = type(e).__name__

            return result
    
    def run_parallel_evaluation(self, dataset_path: str, max_workers: int = 20):
        """Run evaluation in parallel using ThreadPoolExecutor."""
        df = pd.read_csv(dataset_path, encoding='utf-8')
        print(f"Loaded {len(df)} tasks from {dataset_path}")
        print(f"Columns: {df.columns.tolist()}")

        tasks = df.to_dict('records')
        results = []

        print("\n=== Testing first task ===")
        first_result = self.evaluate_single_task(tasks[0])
        results.append(first_result)
        print(f"First result: {first_result}")

        if first_result['prediction'] == 'error':
            print("\nFirst task failed - stopping to debug")
            self.results = results
            return results

        print(f"\nFirst task successful, continuing with {max_workers} workers...")
        remaining_tasks = tasks[1:]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.evaluate_single_task, task): task
                for task in remaining_tasks
            }

            error_count = 0
            with tqdm(total=len(remaining_tasks), desc="Evaluating tasks") as pbar:
                for future in as_completed(future_to_task):
                    try:
                        result = future.result(timeout=120) 
                        results.append(result)

                        if result['prediction'] == 'error':
                            error_count += 1

                    except TimeoutError:
                        print(f"\n Task timed out after 120 seconds")
                        error_count += 1
                        task = future_to_task[future]
                        result = {
                            'task_id': task.get('id', 0),
                            'task': task.get('task_prompt', ''),
                            'ground_truth': task.get('ground_truth', 'unknown'),
                            'prediction': 'error',
                            'processing_time': 120.0,
                            'correct': False,
                            'mode': self.mode
                        }
                        results.append(result)

                    except Exception as e:
                        print(f"\n Worker exception: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
                        error_count += 1

                    finally:
                        pbar.update(1)

            if error_count > 0:
                print(f"\nWARNING: {error_count} tasks encountered errors during evaluation")

        self.results = results
        print(f"\nEvaluated {len(results)} tasks")

        return results
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.results:
            print("No results to calculate metrics from")
            return {}

        df = pd.DataFrame(self.results)

        if 'has_error' in df.columns:
            error_count = len(df[df['has_error'] == True])
            if error_count > 0:
                print(f"WARNING: {error_count} tasks had errors")
                if self.mode == 'base_llm':
                    error_types = df[df['has_error'] == True]['error_type'].value_counts()
                    print(f"Error types: {dict(error_types)}")
            valid_df = df[df['has_error'] == False]
        else:
            error_count = len(df[df['prediction'] == 'error'])
            if error_count > 0:
                print(f"WARNING: {error_count} tasks had errors")
            valid_df = df[df['prediction'] != 'error']

        if len(valid_df) == 0:
            return {
                'total_tasks': len(df),
                'valid_tasks': 0,
                'error_count': error_count,
                'accuracy': 0.0,
                'safe_task_acceptance': 0.0,
                'unsafe_task_rejection': 0.0,
                'avg_processing_time': 0.0
            }

        accuracy = valid_df['correct'].mean()

        safe_tasks = valid_df[valid_df['ground_truth'] == 'safe']
        unsafe_tasks = valid_df[valid_df['ground_truth'] == 'unsafe']

        metrics = {
            'total_tasks': len(df),
            'valid_tasks': len(valid_df),
            'error_count': error_count,
            'accuracy': accuracy,
            'safe_task_acceptance': (safe_tasks['prediction'] == 'safe').mean() if len(safe_tasks) > 0 else 0,
            'unsafe_task_rejection': (unsafe_tasks['prediction'] == 'unsafe').mean() if len(unsafe_tasks) > 0 else 0,
            'avg_processing_time': valid_df['processing_time'].mean() if len(valid_df) > 0 else 0
        }

        if self.mode == 'nert' and 'confidence' in valid_df.columns:
            metrics['avg_confidence'] = valid_df['confidence'].mean()

        return metrics

    def run_with_selective_evaluation(self, dataset_path: str, max_workers: int = 20):
        """Run evaluation and generate selective prediction curves."""
        results = self.run_parallel_evaluation(dataset_path, max_workers)

        if self.mode == 'base_llm':
            valid_results = [r for r in results if not r.get('has_error', False)]
        else:
            valid_results = [r for r in results if r.get('prediction') != 'error']
        
        if len(valid_results) < 10:
            print("Not enough valid results for selective prediction curves")
            return results, None, {}

        try:
            from experiments.selective_prediction import SelectivePredictionEvaluator
            evaluator = SelectivePredictionEvaluator()
            
            tasks = [r['task'] for r in valid_results]
            ground_truth = [r['ground_truth'] for r in valid_results]
            predictions = [r['prediction'] for r in valid_results]
            confidences = [r['neural_confidence'] for r in valid_results]

            curves = evaluator.evaluate(tasks, ground_truth, predictions, confidences)
            
            output_dir = Path('results')
            curves.savefig(output_dir / 'selective_prediction_curves.png', dpi=150, bbox_inches='tight')
            
            metrics = self.calculate_selective_metrics(valid_results)
            
            return results, curves, metrics
        except ImportError:
            print("SelectivePredictionEvaluator not found, skipping curves")
            return results, None, {}

    def calculate_selective_metrics(self, results):
        """Calculate key selective prediction metrics."""
        if not results:
            return {}
            
        df = pd.DataFrame(results)
        
        df = df.sort_values('neural_confidence', ascending=False)
        
        metrics = {}
        
        coverages = []
        risks = []
        
        for k in range(1, len(df) + 1):
            top_k = df.iloc[:k]
            coverage = k / len(df)
            risk = (top_k['prediction'] != top_k['ground_truth']).mean()
            
            coverages.append(coverage)
            risks.append(risk)
        
        try:
            from sklearn.metrics import auc
            metrics['auc_risk'] = auc(coverages, risks)
        except ImportError:
            metrics['auc_risk'] = 0.0
        
        for cov, risk in zip(coverages, risks):
            if risk > 0.05:
                metrics['coverage_at_5_risk'] = coverages[coverages.index(cov) - 1] if cov > 0 else 0
                break
        else:
            metrics['coverage_at_5_risk'] = 1.0
        
        for cov, risk in zip(coverages, risks):
            if cov >= 0.9:
                metrics['risk_at_90_coverage'] = risk
                break
        else:
            metrics['risk_at_90_coverage'] = risks[-1] if risks else 0
        
        return metrics
    
    def save_results(self, output_dir: str = "results"):
        """Save evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.results)
        results_filename = f'benchmark_results_{self.mode}.csv'
        df.to_csv(output_path / results_filename, index=False, encoding='utf-8')

        metrics = self.calculate_metrics()
        metrics_filename = f'metrics_{self.mode}.json'
        with open(output_path / metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nResults saved to {output_path}")
        print(f"  Results CSV: {results_filename}")
        print(f"  Metrics JSON: {metrics_filename}")
        if metrics:
            if metrics.get('valid_tasks', 0) == 0:
                print(f"\nWARNING: All {metrics['total_tasks']} tasks failed - no valid results")
                print(f"Error count: {metrics['error_count']}")
            else:
                if 'error_count' in metrics and metrics['error_count'] > 0:
                    print(f"Errors: {metrics['error_count']} tasks failed")
                print(f"Accuracy: {metrics['accuracy']:.2%}")
                print(f"Safe task acceptance: {metrics['safe_task_acceptance']:.2%}")
                print(f"Unsafe task rejection: {metrics['unsafe_task_rejection']:.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='benchmark_621.csv')
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--temp', type=float, default=None,
                       help='Temperature for LLM (0.0=deterministic, 0.7=creative). If not specified, uses config.yaml value.')
    parser.add_argument('--mode', default='nert', choices=['nert', 'base_llm'],
                       help='Classification mode: nert (neurosymbolic) or base_llm (direct LLM)')
    parser.add_argument('--selective', action='store_true',
                       help='Generate selective prediction curves')

    args = parser.parse_args()

    print("Checking API key availability...")
    from core.llm_clients import LLMClientFactory
    api_keys = LLMClientFactory.check_api_keys()

    if not any(api_keys.values()):
        print("\nERROR: No API keys found!")
        print("\nMethod 1 (Recommended): Use .env file")
        print("  1. Copy .env.example to .env:")
        print("     cp .env.example .env")
        print("  2. Edit .env and add your API keys")
        print("\nMethod 2: Set environment variables manually")
        print("  Windows:   set OPENAI_API_KEY=your-api-key-here")
        print("  Linux/Mac: export OPENAI_API_KEY=your-api-key-here")
        print("\nSupported API keys:")
        print("  - OPENAI_API_KEY (for GPT models)")
        print("  - GOOGLE_API_KEY (for Gemini models)")
        print("  - ANTHROPIC_API_KEY (for Claude models)")
        sys.exit(1)

    print(f"API keys found: {[k for k, v in api_keys.items() if v]}")
    print()
    
    data_path = Path(args.data)
    if not data_path.exists():
        for potential_path in [
            Path('data/benchmark') / args.data,
            Path('../data/benchmark') / args.data,
            Path('benchmark_621.csv'),
        ]:
            if potential_path.exists():
                data_path = potential_path
                break
    
    if not data_path.exists():
        print(f"Cannot find data file: {args.data}")
        sys.exit(1)
    
    print(f"Using data file: {data_path}")

    if args.temp is not None:
        # User explicitly provided --temp (highest priority)
        temperature = args.temp
        print(f"Using temperature from command-line: {temperature}")
    else:
        # User didn't provide --temp, read from config.yaml
        import yaml
        config_file = Path(args.config)
        if not config_file.exists():
            config_file = Path('..') / args.config

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        temperature = config.get('llm', {}).get('temperature', 0.0)
        print(f"Using temperature from config.yaml: {temperature}")

    evaluator = BenchmarkEvaluator(args.config, temperature=temperature, mode=args.mode)

    if args.selective:
        print("Running evaluation with selective prediction analysis...")
        results, curves, metrics = evaluator.run_with_selective_evaluation(
            str(data_path), args.workers
        )
        if metrics:
            print("\nSelective prediction curves saved to results/")
    else:
        evaluator.run_parallel_evaluation(str(data_path), args.workers)
        evaluator.save_results()