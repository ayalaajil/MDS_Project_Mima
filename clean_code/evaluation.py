import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    hamming_loss, jaccard_score, average_precision_score, confusion_matrix)


# - False Negatives (FN) are more dangerous than False Positives (FP) → Sensitivity & F2 Score are prioritized.
# - Specificity ensures minimal false alarms → avoids unnecessary tests/treatments

class MultiLabelEvaluator:

    def __init__(self, y_true, y_pred, class_names=None, threshold=0.5, alpha=5):

        """
        Initializes the evaluator with true labels and predicted probabilities.
        
        Parameters:

        - y_true: np.array, shape (N, C), true binary labels (0 or 1)
        - y_pred: np.array, shape (N, C), predicted probabilities
        - class_names: list, names of classes
        - threshold: float, probability threshold to convert to binary labels
        - alpha: float, parameter for alpha-Softmax/Softmin aggregation
    
        """

        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.class_names = class_names if class_names else [f"Class {i}" for i in range(self.y_true.shape[1])]
        self.threshold = threshold
        self.alpha = alpha
        self.y_pred_binary = (self.y_pred >= self.threshold).astype(int)

        # Confusion matrix components for each class
        self.tps, self.fps, self.fns, self.tns = [], [], [], []
        num_labels = self.y_true.shape[1]
        fig, axes = plt.subplots(1, num_labels, figsize=(4 * num_labels, 4))

        for c in range(num_labels):

            cm = confusion_matrix(self.y_true[:, c], self.y_pred_binary[:, c])
            self.tns.append(cm[0, 0])
            self.fps.append(cm[0, 1])
            self.fns.append(cm[1, 0])
            self.tps.append(cm[1, 1])
            ax = axes[c] if num_labels > 1 else axes
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            ax.set_title(f"Label {c}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        self.tps, self.fps, self.fns, self.tns = map(np.array, [self.tps, self.fps, self.fns, self.tns])
        
        plt.tight_layout()
        plt.show()

    def compute_binary_metrics(self):

        """Computes AUC, Accuracy, Precision, Recall for each class."""

        aucs, accs, precs, recs = [], [], [], []
        for c in range(self.y_true.shape[1]):
            aucs.append(roc_auc_score(self.y_true[:, c], self.y_pred[:, c]))
            accs.append(accuracy_score(self.y_true[:, c], self.y_pred_binary[:, c]))
            precs.append(precision_score(self.y_true[:, c], self.y_pred_binary[:, c], zero_division=1))
            recs.append(recall_score(self.y_true[:, c], self.y_pred_binary[:, c], zero_division=1))
        return np.array(aucs), np.array(accs), np.array(precs), np.array(recs)

    def aggregate_metrics(self, metrics):

        """Computes average, alpha-Softmax, and alpha-Softmin for a given metric set."""

        avg = np.mean(metrics)
        softmax = np.sum(metrics * np.exp(self.alpha * metrics)) / np.sum(np.exp(self.alpha * metrics))
        softmin = np.sum(metrics * np.exp(-self.alpha * metrics)) / np.sum(np.exp(-self.alpha * metrics))
        return {"Average": avg, "Alpha-Softmax": softmax, "Alpha-Softmin": softmin}

    def compute_false_distributions(self):

        """Computes false positive & false negative distributions across all samples."""

        false_pos_per_sample = np.sum(self.y_pred_binary > self.y_true, axis=1)
        false_neg_per_sample = np.sum(self.y_pred_binary < self.y_true, axis=1)
        return {
            "Avg False Positives Over all samples": np.mean(false_pos_per_sample),
            "Avg False Negatives Over all samples": np.mean(false_neg_per_sample),
            "FP Distribution": false_pos_per_sample,
            "FN Distribution": false_neg_per_sample
        }

    def compute_work_saved(self):

        """Computes the work saved metric."""

        T1 = np.sum(self.y_true, axis=1)  # All Positives (TP + FN)
        T2 = np.sum(self.y_pred_binary != self.y_true, axis=1)  # FN + FP (corrections needed)
        work_saved = 1 - (T2 / np.maximum(T1, 1))  # Avoid division by zero
        return {"Avg Work Saved": np.mean(work_saved)}
    
    def overall_metrics(self):

        """Compute overall multi-label classification metrics, including MAP."""
        
        Overall_metrics ={
            "Hamming Loss": hamming_loss(self.y_true, self.y_pred_binary),
            "Micro F1-score": f1_score(self.y_true, self.y_pred_binary, average='micro'),
            "Macro F1-score": f1_score(self.y_true, self.y_pred_binary, average='macro'),
            "Weighted F1-score": f1_score(self.y_true, self.y_pred_binary, average='weighted'),
            "Jaccard Index (Macro)": jaccard_score(self.y_true, self.y_pred_binary, average='macro'),
            "Subset Accuracy": accuracy_score(self.y_true, self.y_pred_binary),
            "Mean Average Precision (MAP)": np.mean([
                average_precision_score(self.y_true[:, i], self.y_pred_binary[:, i]) 
                for i in range(self.y_true.shape[1]) if np.unique(self.y_true[:, i]).size > 1
            ])
        }
        return Overall_metrics
    
    def evaluate(self):

        """Computes and prints all evaluation metrics."""

        aucs, accs, precs, recs = self.compute_binary_metrics()
        metrics = {
            "AUC": self.aggregate_metrics(aucs),
            "Accuracy": self.aggregate_metrics(accs),
            "Precision": self.aggregate_metrics(precs),
            "Recall": self.aggregate_metrics(recs),
        }
        false_distributions = self.compute_false_distributions()
        work_saved = self.compute_work_saved()
        overall_metrics = self.overall_metrics()
        
        # self.decision_curve_analysis()

        return {**overall_metrics, **metrics, **false_distributions, **work_saved}

if __name__ == "__main__":

##### example to test the pipeline  ######

    y_true = np.random.randint(0, 2, (10, 3))
    y_pred = np.random.rand(10, 3)

    evaluator = MultiLabelEvaluator(y_true, y_pred)

    results = evaluator.evaluate()
    for key, value in results.items():
        print(f"{key}: {value}")