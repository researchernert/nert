# nert/experiments/analyze_results.py
import pandas as pd
import json
from pathlib import Path
import sys

def analyze_results(df, mode_name):
    """Analyze results for a specific mode (NERT or BASE_LLM)."""
    print(f"\n{'='*60}")
    print(f"=== {mode_name.upper()} RESULTS SUMMARY ===")
    print(f"{'='*60}")

    print(f"\nTotal tasks: {len(df)}")

    if 'prediction' in df.columns:
        error_count = len(df[df['prediction'] == 'error'])
        if error_count > 0:
            print(f"⚠️  WARNING: {error_count} tasks had errors")
            df = df[df['prediction'] != 'error'] 
            print(f"Analyzing {len(df)} valid results\n")

    if len(df) == 0:
        print("No valid results to analyze")
        return

    print(f"Ground truth distribution:")
    print(df['ground_truth'].value_counts())
    print(f"\nPrediction distribution:")
    print(df['prediction'].value_counts())

    if 'symbolic_result' in df.columns:
        print(f"\nSymbolic result distribution:")
        print(df['symbolic_result'].value_counts())

    if 'confidence' in df.columns:
        print(f"\nNeural confidence stats:")
        print(f"  Mean: {df['confidence'].mean():.3f}")
        print(f"  Median: {df['confidence'].median():.3f}")
        print(f"  Std: {df['confidence'].std():.3f}")

    if 'support_ratio' in df.columns:
        print(f"\nSupport ratio stats:")
        print(f"  Mean: {df['support_ratio'].mean():.3f}")
        print(f"  Median: {df['support_ratio'].median():.3f}")

    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {df['correct'].mean():.2%}")
    print(f"{'='*60}")

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(df['ground_truth'], df['prediction'], labels=['safe', 'unsafe'])
    print(f"\nConfusion Matrix:")
    print("       Predicted")
    print("       safe  unsafe")
    print(f"safe   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"unsafe {cm[1,0]:4d}  {cm[1,1]:4d}")

    safe_tasks = df[df['ground_truth'] == 'safe']
    unsafe_tasks = df[df['ground_truth'] == 'unsafe']

    print(f"\nPerformance by Category:")
    if len(safe_tasks) > 0:
        safe_acc = (safe_tasks['prediction'] == 'safe').mean()
        print(f"  Safe task acceptance:   {safe_acc:.2%} ({int(safe_acc * len(safe_tasks))}/{len(safe_tasks)})")
    if len(unsafe_tasks) > 0:
        unsafe_acc = (unsafe_tasks['prediction'] == 'unsafe').mean()
        print(f"  Unsafe task rejection:  {unsafe_acc:.2%} ({int(unsafe_acc * len(unsafe_tasks))}/{len(unsafe_tasks)})")

    print(f"\nDetailed Classification Report:")
    print(classification_report(df['ground_truth'], df['prediction']))

    if 'processing_time' in df.columns:
        print(f"\nProcessing Time Stats:")
        print(f"  Mean: {df['processing_time'].mean():.2f}s")
        print(f"  Median: {df['processing_time'].median():.2f}s")
        print(f"  Total: {df['processing_time'].sum():.1f}s ({df['processing_time'].sum()/60:.1f}m)")

def compare_modes(nert_df, base_llm_df):
    """Compare NERT and BASE_LLM results side-by-side."""
    print(f"\n{'='*60}")
    print("=== MODE COMPARISON ===")
    print(f"{'='*60}")

    nert_valid = nert_df[nert_df['prediction'] != 'error'] if 'prediction' in nert_df.columns else nert_df
    base_valid = base_llm_df[base_llm_df['prediction'] != 'error'] if 'prediction' in base_llm_df.columns else base_llm_df

    print(f"\n{'Metric':<30} {'NERT':<15} {'BASE_LLM':<15} {'Δ':<10}")
    print("-" * 70)

    nert_acc = nert_valid['correct'].mean() if len(nert_valid) > 0 else 0
    base_acc = base_valid['correct'].mean() if len(base_valid) > 0 else 0
    delta_acc = nert_acc - base_acc

    print(f"{'Accuracy':<30} {nert_acc:>6.2%}{'':8} {base_acc:>6.2%}{'':8} {delta_acc:>+6.2%}")

    nert_safe = nert_valid[nert_valid['ground_truth'] == 'safe']
    base_safe = base_valid[base_valid['ground_truth'] == 'safe']

    if len(nert_safe) > 0 and len(base_safe) > 0:
        nert_safe_acc = (nert_safe['prediction'] == 'safe').mean()
        base_safe_acc = (base_safe['prediction'] == 'safe').mean()
        delta_safe = nert_safe_acc - base_safe_acc
        print(f"{'Safe task acceptance':<30} {nert_safe_acc:>6.2%}{'':8} {base_safe_acc:>6.2%}{'':8} {delta_safe:>+6.2%}")

    nert_unsafe = nert_valid[nert_valid['ground_truth'] == 'unsafe']
    base_unsafe = base_valid[base_valid['ground_truth'] == 'unsafe']

    if len(nert_unsafe) > 0 and len(base_unsafe) > 0:
        nert_unsafe_acc = (nert_unsafe['prediction'] == 'unsafe').mean()
        base_unsafe_acc = (base_unsafe['prediction'] == 'unsafe').mean()
        delta_unsafe = nert_unsafe_acc - base_unsafe_acc
        print(f"{'Unsafe task rejection':<30} {nert_unsafe_acc:>6.2%}{'':8} {base_unsafe_acc:>6.2%}{'':8} {delta_unsafe:>+6.2%}")

    if 'processing_time' in nert_valid.columns and 'processing_time' in base_valid.columns:
        nert_time = nert_valid['processing_time'].mean()
        base_time = base_valid['processing_time'].mean()
        speedup = base_time / nert_time if nert_time > 0 else 0
        print(f"{'Avg processing time':<30} {nert_time:>6.2f}s{'':6} {base_time:>6.2f}s{'':6} {speedup:>6.2f}x")

if __name__ == "__main__":
    results_dir = Path("results")

    nert_path = results_dir / "benchmark_results_nert.csv"
    base_llm_path = results_dir / "benchmark_results_base_llm.csv"

    legacy_path = results_dir / "benchmark_results.csv"

    nert_df = None
    base_llm_df = None

    if nert_path.exists():
        print(f"Loading NERT results from: {nert_path}")
        nert_df = pd.read_csv(nert_path, encoding='utf-8')
        analyze_results(nert_df, "NERT")

    if base_llm_path.exists():
        print(f"Loading BASE_LLM results from: {base_llm_path}")
        base_llm_df = pd.read_csv(base_llm_path, encoding='utf-8')
        analyze_results(base_llm_df, "BASE_LLM")

    if nert_df is None and base_llm_df is None and legacy_path.exists():
        print(f"Loading results from legacy file: {legacy_path}")
        df = pd.read_csv(legacy_path, encoding='utf-8')

        if 'mode' in df.columns:
            mode = df['mode'].iloc[0]
            analyze_results(df, mode)
        elif 'symbolic_result' in df.columns:
            analyze_results(df, "NERT (detected)")
        else:
            analyze_results(df, "BASE_LLM (detected)")

    if nert_df is not None and base_llm_df is not None:
        compare_modes(nert_df, base_llm_df)

    if nert_df is None and base_llm_df is None and not legacy_path.exists():
        print(f"❌ No results found in {results_dir}")
        print("\nExpected files:")
        print(f"  - {nert_path}")
        print(f"  - {base_llm_path}")
        print(f"  - {legacy_path} (legacy)")
        print("\nRun benchmark first: python run_benchmark.py --mode nert")
        sys.exit(1)