# src/explainable_ai/fairness_analysis.py
"""
Fairness & Bias Analysis Module
Unit V Concept: Fairness, Bias, and Explainability
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class FairnessAnalysis:
    """
    Fairness and Bias Analysis Implementation
    
    Implements:
    1. Demographic Parity
    2. Equal Opportunity
    3. Predictive Equality
    4. Bias Detection
    5. Fairness Metrics
    """
    
    def __init__(self):
        """Initialize fairness analysis"""
        self.metrics = {}
        
        logger.info("FairnessAnalysis initialized")
    
    def calculate_fairness_metrics(self, y_true, y_pred, sensitive_attrs, group_names=None):
        """
        Calculate fairness metrics for different groups
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attrs: Sensitive attribute values (e.g., gender, age group)
            group_names: Names of groups
            
        Returns:
            Dictionary of fairness metrics
        """
        unique_groups = np.unique(sensitive_attrs)
        
        if group_names is None:
            group_names = [f'Group_{g}' for g in unique_groups]
        
        results = {}
        
        print("\n" + "="*60)
        print("⚖️ FAIRNESS & BIAS ANALYSIS")
        print("="*60)
        
        # Overall metrics
        overall_acc = accuracy_score(y_true, y_pred)
        overall_prec = precision_score(y_true, y_pred, average='weighted')
        overall_rec = recall_score(y_true, y_pred, average='weighted')
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {overall_acc:.4f}")
        print(f"  Precision: {overall_prec:.4f}")
        print(f"  Recall: {overall_rec:.4f}")
        
        print("\n" + "-"*60)
        print("Group-wise Performance:")
        print("-"*60)
        
        group_metrics = {}
        
        for i, group in enumerate(unique_groups):
            group_mask = sensitive_attrs == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]
            
            if len(group_true) == 0:
                continue
            
            group_acc = accuracy_score(group_true, group_pred)
            group_prec = precision_score(group_true, group_pred, average='weighted')
            group_rec = recall_score(group_true, group_pred, average='weighted')
            
            group_metrics[group_names[i]] = {
                'accuracy': group_acc,
                'precision': group_prec,
                'recall': group_rec,
                'n_samples': len(group_true)
            }
            
            print(f"\n{group_names[i]} (n={len(group_true)}):")
            print(f"  Accuracy: {group_acc:.4f}")
            print(f"  Precision: {group_prec:.4f}")
            print(f"  Recall: {group_rec:.4f}")
        
        # Calculate fairness gaps
        print("\n" + "="*60)
        print("📊 FAIRNESS GAPS")
        print("="*60)
        
        fairness_gaps = {}
        
        for metric in ['accuracy', 'precision', 'recall']:
            values = [group_metrics[g][metric] for g in group_metrics]
            if values:
                max_val = max(values)
                min_val = min(values)
                gap = max_val - min_val
                fairness_gaps[metric] = gap
                print(f"{metric.capitalize()} Gap: {gap:.4f}")
        
        # Determine if model is fair
        print("\n" + "="*60)
        print("⚖️ FAIRNESS ASSESSMENT")
        print("="*60)
        
        max_gap = max(fairness_gaps.values()) if fairness_gaps else 0
        
        if max_gap < 0.05:
            print("✅ Model is FAIR - Small disparities across groups")
        elif max_gap < 0.1:
            print("⚠️ Model has MODERATE bias - Some disparities detected")
        else:
            print("❌ Model has SIGNIFICANT bias - Large disparities across groups")
        
        self.metrics = {
            'overall': {
                'accuracy': overall_acc,
                'precision': overall_prec,
                'recall': overall_rec
            },
            'group_metrics': group_metrics,
            'fairness_gaps': fairness_gaps
        }
        
        return self.metrics
    
    def calculate_bias_metrics(self, y_true, y_pred, sensitive_attrs):
        """
        Calculate bias metrics for binary classification
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attrs: Sensitive attribute values
            
        Returns:
            Dictionary of bias metrics
        """
        unique_groups = np.unique(sensitive_attrs)
        
        bias_metrics = {}
        
        print("\n" + "="*60)
        print("📊 BIAS METRICS")
        print("="*60)
        
        for group in unique_groups:
            group_mask = sensitive_attrs == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]
            
            # Confusion matrix for this group
            tn, fp, fn, tp = confusion_matrix(group_true, group_pred).ravel()
            
            # Bias metrics
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
            
            bias_metrics[group] = {
                'fpr': fpr,
                'fnr': fnr,
                'ppv': ppv,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
            
            print(f"\nGroup {group}:")
            print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
            print(f"  False Positive Rate: {fpr:.4f}")
            print(f"  False Negative Rate: {fnr:.4f}")
            print(f"  Positive Predictive Value: {ppv:.4f}")
        
        # Compare groups
        if len(unique_groups) == 2:
            g0, g1 = unique_groups
            
            # Equal Opportunity: Should have similar True Positive Rates
            fpr_diff = abs(bias_metrics[g0]['fpr'] - bias_metrics[g1]['fpr'])
            fnr_diff = abs(bias_metrics[g0]['fnr'] - bias_metrics[g1]['fnr'])
            
            print("\n" + "="*60)
            print("⚖️ BIAS COMPARISON")
            print("="*60)
            print(f"FPR Difference: {fpr_diff:.4f} (Lower is better)")
            print(f"FNR Difference: {fnr_diff:.4f} (Lower is better)")
        
        return bias_metrics
    
    def plot_fairness_metrics(self, save_path=None):
        """
        Plot fairness metrics visualization
        
        Args:
            save_path: Optional save path
        """
        if not self.metrics:
            print("No metrics available. Run calculate_fairness_metrics() first.")
            return
        
        group_metrics = self.metrics.get('group_metrics', {})
        
        if not group_metrics:
            return
        
        groups = list(group_metrics.keys())
        metrics_to_plot = ['accuracy', 'precision', 'recall']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [group_metrics[g][metric] for g in groups]
            
            axes[idx].bar(groups, values, color=['#3498db', '#2ecc71'])
            axes[idx].set_title(f'{metric.capitalize()} by Group')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_ylim(0, 1)
            axes[idx].grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.suptitle('Fairness Metrics Comparison Across Groups', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Fairness plot saved to: {save_path}")
        
        plt.show()
    
    def generate_fairness_report(self):
        """
        Generate a fairness report
        
        Returns:
            String report
        """
        if not self.metrics:
            return "No metrics available"
        
        report = []
        report.append("="*60)
        report.append("⚖️ FAIRNESS & BIAS REPORT")
        report.append("="*60)
        
        # Overall metrics
        overall = self.metrics.get('overall', {})
        report.append("\n📊 Overall Performance:")
        for key, value in overall.items():
            report.append(f"  {key.capitalize()}: {value:.4f}")
        
        # Group metrics
        group_metrics = self.metrics.get('group_metrics', {})
        report.append("\n📊 Group-wise Performance:")
        for group, metrics in group_metrics.items():
            report.append(f"\n  {group}:")
            for key, value in metrics.items():
                report.append(f"    {key.capitalize()}: {value:.4f}")
        
        # Fairness gaps
        gaps = self.metrics.get('fairness_gaps', {})
        report.append("\n📊 Fairness Gaps:")
        for key, value in gaps.items():
            report.append(f"  {key.capitalize()} Gap: {value:.4f}")
        
        # Assessment
        max_gap = max(gaps.values()) if gaps else 0
        report.append("\n⚖️ Assessment:")
        if max_gap < 0.05:
            report.append("  ✅ Model is FAIR - Small disparities across groups")
        elif max_gap < 0.1:
            report.append("  ⚠️ Model has MODERATE bias - Some disparities detected")
        else:
            report.append("  ❌ Model has SIGNIFICANT bias - Large disparities across groups")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
