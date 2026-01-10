"""Tests for evaluation metrics."""
import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.evaluation.metrics import MetricsCalculator, EvaluationResult, generate_evaluation_report


class TestMetricsCalculator:
    """Tests for metrics calculation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = MetricsCalculator(threshold=0.5)
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.95])
        
        result = self.calculator.compute_all(y_true, y_prob)
        
        assert result.detection_rate == 1.0
        assert result.false_positive_rate == 0.0
        assert result.false_negative_rate == 0.0
        assert result.precision == 1.0
        assert result.f1_score == 1.0
    
    def test_worst_predictions(self):
        """Test metrics with completely wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])  # All wrong
        
        result = self.calculator.compute_all(y_true, y_prob)
        
        assert result.detection_rate == 0.0
        assert result.false_positive_rate == 1.0
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.6, 0.2, 0.9, 0.4, 0.8])
        
        result = self.calculator.compute_all(y_true, y_prob)
        
        # Check confusion matrix components add up
        total = result.true_positives + result.false_positives + \
                result.true_negatives + result.false_negatives
        assert total == len(y_true)
    
    def test_meets_targets(self):
        """Test target checking."""
        result = EvaluationResult(
            detection_rate=0.995,
            false_negative_rate=0.005,
            false_positive_rate=0.03,
            f1_score=0.98,
            auc_roc=0.995,
            mcc=0.92,
            mean_latency_ms=30.0
        )
        
        targets = result.meets_targets()
        
        assert targets['detection_rate']['met'] == True
        assert targets['false_negative_rate']['met'] == True
        assert targets['false_positive_rate']['met'] == True
    
    def test_threshold_analysis(self):
        """Test metrics at multiple thresholds."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.3, 0.4, 0.6, 0.8, 0.5, 0.9])
        
        results = self.calculator.compute_at_thresholds(
            y_true, y_prob,
            thresholds=[0.3, 0.5, 0.7]
        )
        
        assert len(results) == 3
        for r in results:
            assert 'threshold' in r
            assert 'detection_rate' in r
            assert 'false_positive_rate' in r


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = EvaluationResult(
            detection_rate=0.99,
            false_negative_rate=0.01,
            false_positive_rate=0.05,
            precision=0.95,
            f1_score=0.97,
            auc_roc=0.99,
            mcc=0.90,
            average_precision=0.98,
            mean_latency_ms=25.0,
            throughput_per_second=40.0,
            model_size_mb=100.0,
            true_positives=99,
            false_positives=5,
            false_negatives=1,
            true_negatives=95
        )
        
        d = result.to_dict()
        
        assert 'detection_rate' in d
        assert 'confusion_matrix' in d
        assert d['confusion_matrix']['tp'] == 99


class TestEvaluationReport:
    """Tests for evaluation report generation."""
    
    def test_generate_report(self):
        """Test report generation."""
        result = EvaluationResult(
            detection_rate=0.99,
            false_negative_rate=0.01,
            false_positive_rate=0.03,
            precision=0.95,
            f1_score=0.97,
            auc_roc=0.99,
            mcc=0.92,
            average_precision=0.98,
            mean_latency_ms=25.0,
            throughput_per_second=40.0,
            model_size_mb=100.0,
            true_positives=990,
            false_positives=30,
            false_negatives=10,
            true_negatives=970
        )
        
        report = generate_evaluation_report(result)
        
        assert "ADVERSARIAL DETECTION EVALUATION REPORT" in report
        assert "Detection Rate" in report
        assert "CONFUSION MATRIX" in report
        assert "PASS" in report or "FAIL" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
