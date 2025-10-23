# nert/experiments/selective_prediction.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
from typing import List, Dict, Tuple

class SelectivePredictionEvaluator:
    """Evaluate selective prediction performance for safety-critical robotics."""
    
    def evaluate(self, tasks: List[str], ground_truth: List[str], 
                 predictions: List[str], confidences: List[float]):
        """
        Generate selective prediction curves.
        
        Args:
            tasks: Task descriptions
            ground_truth: True labels ('safe' or 'unsafe')
            predictions: Model predictions ('safe' or 'unsafe')
            confidences: Confidence scores from neural model (0-1)
        """

        gt = np.array([1 if g == 'safe' else 0 for g in ground_truth])
        pred = np.array([1 if p == 'safe' else 0 for p in predictions])
        conf = np.array(confidences)
        
        indices = np.argsort(conf)[::-1]
        
        results = []
        
        for k in range(1, len(tasks) + 1):
            selected_idx = indices[:k]
            
            coverage = k / len(tasks)
            
            if len(selected_idx) > 0:
                selected_gt = gt[selected_idx]
                selected_pred = pred[selected_idx]
                
                accuracy = np.mean(selected_gt == selected_pred)
                
                risk = 1 - accuracy
                
                safe_tasks_total = np.sum(gt == 1)
                if safe_tasks_total > 0:
                    safe_in_selected = selected_idx[selected_gt == 1]
                    safe_correct = np.sum(pred[safe_in_selected] == 1)
                    utility = safe_correct / safe_tasks_total
                else:
                    utility = 0
                
                unsafe_tasks_total = np.sum(gt == 0)
                if unsafe_tasks_total > 0:
                    unsafe_in_selected = selected_idx[selected_gt == 0]
                    unsafe_rejected = np.sum(pred[unsafe_in_selected] == 0)
                    unsafe_rejection = unsafe_rejected / unsafe_tasks_total
                else:
                    unsafe_rejection = 1
                
            else:
                accuracy = 1.0  
                risk = 0.0
                utility = 0.0
                unsafe_rejection = 1.0
            
            results.append({
                'coverage': coverage,
                'accuracy': accuracy,
                'risk': risk,
                'utility': utility,
                'unsafe_rejection': unsafe_rejection,
                'threshold': conf[indices[k-1]] if k <= len(conf) else 0
            })
        
        return self.plot_curves(results)
    
    def plot_curves(self, results: List[Dict]):
        """Plot the key selective prediction curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        df = pd.DataFrame(results)
        
        axes[0, 0].plot(df['coverage'], df['risk'], 'b-', linewidth=2, label='NERT')
        axes[0, 0].fill_between(df['coverage'], 0, df['risk'], alpha=0.2)
        axes[0, 0].axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% risk threshold')
        axes[0, 0].set_xlabel('Coverage (fraction of tasks accepted)')
        axes[0, 0].set_ylabel('Risk (error rate on accepted tasks)')
        axes[0, 0].set_title('Risk-Coverage Tradeoff')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].set_xlim([0, 1])
        axes[0, 0].set_ylim([0, max(0.3, df['risk'].max() * 1.1)])
        
        axes[0, 1].plot(df['coverage'], df['utility'], 'g-', linewidth=2, label='Safe task recall')
        axes[0, 1].plot(df['coverage'], df['unsafe_rejection'], 'r-', linewidth=2, label='Unsafe task rejection')
        axes[0, 1].set_xlabel('Coverage')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].set_title('Task-Specific Performance vs Coverage')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].set_xlim([0, 1])
        axes[0, 1].set_ylim([0, 1])
        
        axes[1, 0].plot(df['utility'], df['accuracy'], 'purple', linewidth=2)
        axes[1, 0].scatter(df.iloc[-1]['utility'], df.iloc[-1]['accuracy'], 
                          c='red', s=100, zorder=5, label='All accepted')
        axes[1, 0].set_xlabel('Utility (safe task recall)')
        axes[1, 0].set_ylabel('Safety (overall accuracy)')
        axes[1, 0].set_title('Utility-Safety Tradeoff')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_ylim([0, 1])
        
        axes[1, 1].plot(df['coverage'], df['threshold'], 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Coverage')
        axes[1, 1].set_ylabel('Confidence Threshold')
        axes[1, 1].set_title('Operating Points')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].set_ylim([0, 1])
        
        axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Default threshold')
        axes[1, 1].legend()
        
        plt.suptitle('NERT Selective Prediction Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        
        return fig