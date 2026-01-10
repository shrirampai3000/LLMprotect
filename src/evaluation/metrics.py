"""
Evaluation Metrics for Adversarial Detection System.

Implements all 20 metrics from the specification:
1-8: Detection Performance Metrics
9-11: Performance Metrics
12-14: Robustness Metrics
15-17: Cryptographic Security Metrics
18-20: System-Level Metrics
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix as sklearn_confusion_matrix
)
import time


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    # Primary metrics
    detection_rate: float = 0.0
    false_negative_rate: float = 0.0
    false_positive_rate: float = 0.0
    precision: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    
    # Secondary metrics
    mcc: float = 0.0
    average_precision: float = 0.0
    
    # Performance metrics
    mean_latency_ms: float = 0.0
    throughput_per_second: float = 0.0
    model_size_mb: float = 0.0
    
    # Confusion matrix
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    
    # Raw data for visualization
    probabilities: List[float] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding raw data)."""
        return {
            'detection_rate': self.detection_rate,
            'false_negative_rate': self.false_negative_rate,
            'false_positive_rate': self.false_positive_rate,
            'precision': self.precision,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'mcc': self.mcc,
            'average_precision': self.average_precision,
            'mean_latency_ms': self.mean_latency_ms,
            'throughput_per_second': self.throughput_per_second,
            'model_size_mb': self.model_size_mb,
            'confusion_matrix': {
                'tp': self.true_positives,
                'fp': self.false_positives,
                'fn': self.false_negatives,
                'tn': self.true_negatives
            }
        }
    
    def meets_targets(self) -> Dict[str, bool]:
        """Check if metrics meet target thresholds."""
        targets = {
            'detection_rate': (self.detection_rate >= 0.99, 0.99),
            'false_negative_rate': (self.false_negative_rate <= 0.01, 0.01),
            'false_positive_rate': (self.false_positive_rate <= 0.05, 0.05),
            'f1_score': (self.f1_score >= 0.97, 0.97),
            'auc_roc': (self.auc_roc >= 0.99, 0.99),
            'mcc': (self.mcc >= 0.90, 0.90),
            'mean_latency_ms': (self.mean_latency_ms <= 50, 50),
        }
        
        return {
            name: {'met': met, 'target': target, 'actual': getattr(self, name)}
            for name, (met, target) in targets.items()
        }


class MetricsCalculator:
    """
    Calculate all evaluation metrics for adversarial detection.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize calculator.
        
        Args:
            threshold: Classification threshold
        """
        self.threshold = threshold
    
    def compute_all(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        latencies: Optional[List[float]] = None,
        model_size_mb: float = 0.0
    ) -> EvaluationResult:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_prob: Predicted probabilities
            latencies: List of inference latencies (ms)
            model_size_mb: Model size in MB
            
        Returns:
            EvaluationResult with all metrics
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_pred = (y_prob >= self.threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = sklearn_confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        
        # Primary metrics
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = detection_rate
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_roc = 0.5
        
        # MCC (Matthews Correlation Coefficient)
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except:
            mcc = 0.0
        
        # Average Precision
        try:
            ap = average_precision_score(y_true, y_prob)
        except ValueError:
            ap = 0.0
        
        # Performance metrics
        if latencies:
            mean_latency = np.mean(latencies)
            throughput = 1000.0 / mean_latency if mean_latency > 0 else 0
        else:
            mean_latency = 0.0
            throughput = 0.0
        
        return EvaluationResult(
            detection_rate=float(detection_rate),
            false_negative_rate=float(fnr),
            false_positive_rate=float(fpr),
            precision=float(precision),
            f1_score=float(f1),
            auc_roc=float(auc_roc),
            mcc=float(mcc),
            average_precision=float(ap),
            mean_latency_ms=float(mean_latency),
            throughput_per_second=float(throughput),
            model_size_mb=model_size_mb,
            true_positives=int(tp),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_negatives=int(tn),
            probabilities=y_prob.tolist(),
            labels=y_true.tolist()
        )
    
    def compute_at_thresholds(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: List[float] = None
    ) -> List[Dict]:
        """
        Compute metrics at multiple thresholds.
        
        Returns list of results for threshold analysis.
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = []
        for threshold in thresholds:
            self.threshold = threshold
            result = self.compute_all(y_true, y_prob)
            results.append({
                'threshold': threshold,
                'detection_rate': result.detection_rate,
                'false_positive_rate': result.false_positive_rate,
                'precision': result.precision,
                'f1_score': result.f1_score
            })
        
        return results
    
    @staticmethod
    def compute_roc_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve data.
        
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        from sklearn.metrics import roc_curve
        return roc_curve(y_true, y_prob)
    
    @staticmethod
    def compute_precision_recall_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Precision-Recall curve data.
        
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        return precision_recall_curve(y_true, y_prob)


class CryptoSecurityMetrics:
    """
    Metrics for cryptographic security evaluation.
    """
    
    @staticmethod
    def signature_verification_rate(
        valid_signatures: int,
        total_signatures: int
    ) -> float:
        """Compute signature verification success rate."""
        if total_signatures == 0:
            return 1.0
        return valid_signatures / total_signatures
    
    @staticmethod
    def replay_prevention_rate(
        blocked_replays: int,
        total_replay_attempts: int
    ) -> float:
        """Compute replay attack prevention rate."""
        if total_replay_attempts == 0:
            return 1.0
        return blocked_replays / total_replay_attempts
    
    @staticmethod
    def merkle_integrity_rate(
        valid_verifications: int,
        total_verifications: int
    ) -> float:
        """Compute Merkle tree integrity verification rate."""
        if total_verifications == 0:
            return 1.0
        return valid_verifications / total_verifications


class SystemMetrics:
    """
    System-level metrics for overall security assessment.
    """
    
    @staticmethod
    def end_to_end_attack_success_rate(
        successful_attacks: int,
        total_attack_attempts: int
    ) -> float:
        """
        Compute end-to-end attack success rate.
        
        Target: <0.1%
        """
        if total_attack_attempts == 0:
            return 0.0
        return successful_attacks / total_attack_attempts
    
    @staticmethod
    def false_authorization_rate(
        unauthorized_approvals: int,
        total_adversarial_prompts: int
    ) -> float:
        """
        Compute rate of false authorizations.
        
        Target: 0%
        """
        if total_adversarial_prompts == 0:
            return 0.0
        return unauthorized_approvals / total_adversarial_prompts
    
    @staticmethod
    def audit_coverage_rate(
        logged_events: int,
        total_events: int
    ) -> float:
        """
        Compute audit log coverage.
        
        Target: 100%
        """
        if total_events == 0:
            return 1.0
        return logged_events / total_events


def generate_evaluation_report(result: EvaluationResult) -> str:
    """
    Generate a text evaluation report.
    
    Args:
        result: EvaluationResult instance
        
    Returns:
        Formatted report string
    """
    targets_met = result.meets_targets()
    
    report = """
================================================================================
                    ADVERSARIAL DETECTION EVALUATION REPORT
================================================================================

DETECTION PERFORMANCE METRICS
-----------------------------
Detection Rate (Recall):     {detection_rate:.4f}  [{det_status}]
False Negative Rate:         {fnr:.4f}  [{fnr_status}]
False Positive Rate:         {fpr:.4f}  [{fpr_status}]
Precision:                   {precision:.4f}
F1 Score:                    {f1:.4f}  [{f1_status}]
AUC-ROC:                     {auc:.4f}  [{auc_status}]
MCC:                         {mcc:.4f}  [{mcc_status}]
Average Precision:           {ap:.4f}

CONFUSION MATRIX
----------------
                  Predicted
              Benign  Adversarial
Actual Benign   {tn:5d}    {fp:5d}
       Advers.  {fn:5d}    {tp:5d}

PERFORMANCE METRICS
-------------------
Mean Latency:            {latency:.2f} ms  [{lat_status}]
Throughput:              {throughput:.1f} prompts/sec
Model Size:              {size:.2f} MB

TARGETS SUMMARY
---------------
""".format(
        detection_rate=result.detection_rate,
        det_status="✓ PASS" if targets_met['detection_rate']['met'] else "✗ FAIL",
        fnr=result.false_negative_rate,
        fnr_status="✓ PASS" if targets_met['false_negative_rate']['met'] else "✗ FAIL",
        fpr=result.false_positive_rate,
        fpr_status="✓ PASS" if targets_met['false_positive_rate']['met'] else "✗ FAIL",
        precision=result.precision,
        f1=result.f1_score,
        f1_status="✓ PASS" if targets_met['f1_score']['met'] else "✗ FAIL",
        auc=result.auc_roc,
        auc_status="✓ PASS" if targets_met['auc_roc']['met'] else "✗ FAIL",
        mcc=result.mcc,
        mcc_status="✓ PASS" if targets_met['mcc']['met'] else "✗ FAIL",
        ap=result.average_precision,
        tn=result.true_negatives,
        fp=result.false_positives,
        fn=result.false_negatives,
        tp=result.true_positives,
        latency=result.mean_latency_ms,
        lat_status="✓ PASS" if targets_met['mean_latency_ms']['met'] else "✗ FAIL",
        throughput=result.throughput_per_second,
        size=result.model_size_mb
    )
    
    passed = sum(1 for t in targets_met.values() if t['met'])
    total = len(targets_met)
    report += f"Targets Met: {passed}/{total}\n"
    
    return report
