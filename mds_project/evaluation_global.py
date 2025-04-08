import pandas as pd
from transformers import pipeline
from extracting_prompt import ExtractingPrompt
from LLM import LLM
from tqdm import tqdm
import re
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score, 
    precision_score, recall_score, hamming_loss, f1_score, jaccard_score, 
    average_precision_score
)

import warnings
warnings.filterwarnings('ignore')


def round_dict(d, ndigits=2):
    """
    Recursively rounds numeric values in a dictionary (or list/tuple) to 'ndigits' decimals.
    """
    if isinstance(d, dict):
        return {k: round_dict(v, ndigits) for k, v in d.items()}
    elif isinstance(d, (list, tuple)):
        return type(d)(round_dict(v, ndigits) for v in d)
    elif isinstance(d, (float, np.floating)):
        return round(float(d), ndigits)
    else:
        return d


class MultiLabelEvaluator:

    def __init__(self, y_true, y_pred, class_names=None, threshold=0.5, alpha=5, plot_confusion_matrices=False):

        """
        Evaluate multi-label classification performance using class-based metrics.
        
        Parameters:
            y_true (np.array): Binary array of shape (num_samples, num_classes) for ground truth.
            y_pred (np.array): Real-valued array (or probabilities) of shape (num_samples, num_classes).
            class_names (list): List of class names. If None, uses generic class labels.
            threshold (float): Threshold to convert y_pred to binary.
            alpha (float): Parameter for alpha-softmax/softmin aggregation.
        
        """

        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.class_names = class_names if class_names else [f"Class {i}" for i in range(self.y_true.shape[1])]
        self.threshold = threshold
        self.alpha = alpha
        self.y_pred_binary = (self.y_pred >= self.threshold).astype(int)

        # Compute confusion matrices for each class and display them.
        self.tps, self.fps, self.fns, self.tns = [], [], [], []
        num_labels = self.y_true.shape[1]
        
        if plot_confusion_matrices  :
            fig, axes = plt.subplots(1, num_labels, figsize=(4 * num_labels, 4))
            for c in range(num_labels):
                # Force 2x2 confusion matrix with labels [0, 1]
                cm = confusion_matrix(self.y_true[:, c], self.y_pred_binary[:, c], labels=[0, 1])
                self.tns.append(cm[0, 0])
                self.fps.append(cm[0, 1])
                self.fns.append(cm[1, 0])
                self.tps.append(cm[1, 1])
                ax = axes[c] if num_labels > 1 else axes
                sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
                ax.set_title(f"{self.class_names[c]}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
            self.tps, self.fps, self.fns, self.tns = map(np.array, [self.tps, self.fps, self.fns, self.tns])
            plt.tight_layout()
            plt.show()

    def compute_binary_metrics(self):
        """Computes AUC, Accuracy, Precision, and Recall for each class."""
        aucs, accs, precs, recs = [], [], [], []
        for c in range(self.y_true.shape[1]):
            # aucs.append(roc_auc_score(self.y_true[:, c], self.y_pred[:, c]))
            accs.append(accuracy_score(self.y_true[:, c], self.y_pred_binary[:, c]))
            precs.append(precision_score(self.y_true[:, c], self.y_pred_binary[:, c], zero_division=1))
            recs.append(recall_score(self.y_true[:, c], self.y_pred_binary[:, c], zero_division=1))
        return np.array(aucs), np.array(accs), np.array(precs), np.array(recs)


    def aggregate_metrics(self, metrics):
        """
        Aggregates a list of metrics using average, alpha-softmax, and alpha-softmin.
        
        Returns a dictionary with aggregated values.
        """
        avg = np.mean(metrics)
        softmax = np.sum(metrics * np.exp(self.alpha * metrics)) / np.sum(np.exp(self.alpha * metrics))
        softmin = np.sum(metrics * np.exp(-self.alpha * metrics)) / np.sum(np.exp(-self.alpha * metrics))
        return {"Average": avg, "Alpha-Softmax": softmax, "Alpha-Softmin": softmin}


    def compute_false_distributions(self):
        """Computes false positive and false negative distributions per sample."""
        false_pos_per_sample = np.sum(self.y_pred_binary > self.y_true, axis=1)
        false_neg_per_sample = np.sum(self.y_pred_binary < self.y_true, axis=1)
        return {
            "Avg False Positives Over all samples": np.mean(false_pos_per_sample),
            "Avg False Negatives Over all samples": np.mean(false_neg_per_sample),
            "FP Distribution": false_pos_per_sample,
            "FN Distribution": false_neg_per_sample
        }

    def compute_work_saved(self):
        """Computes a work-saved metric indicating the proportion of corrections saved."""
        T1 = np.sum(self.y_true, axis=1)  # All positives (TP + FN)
        T2 = np.sum(self.y_pred_binary != self.y_true, axis=1)  # Corrections needed (FN + FP)
        work_saved = 1 - (T2 / np.maximum(T1, 1))  # Avoid division by zero
        return {"Avg Work Saved": np.mean(work_saved)}
    

    def overall_metrics(self):
        """Compute overall multi-label classification metrics."""
        overall = {
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
        return overall

    def evaluate(self):
        """Compute and return all class-based evaluation metrics."""
        aucs, accs, precs, recs = self.compute_binary_metrics()
        metrics = {
            # "AUC": self.aggregate_metrics(aucs),
            "Accuracy": self.aggregate_metrics(accs),
            "Precision": self.aggregate_metrics(precs),
            "Recall": self.aggregate_metrics(recs),
        }
        false_dist = self.compute_false_distributions()
        work_saved = self.compute_work_saved()
        overall = self.overall_metrics()
        return {
            "Class-Based Metrics": {**overall, **metrics, **false_dist, **work_saved}
        }


class SymptomMultiLabelEvaluator(MultiLabelEvaluator):
    def __init__(self, true_symptoms, extracted_symptoms, symptom_universe=None, threshold=0.5, alpha=5):
        """
        Adapts the MultiLabelEvaluator for symptom extraction evaluation using both class-based
        and sample-based metrics.
        
        Parameters:
            - true_symptoms: List of sets, each containing the true symptoms for a dialogue.
            - extracted_symptoms: List of sets, each containing the extracted symptoms.
            - symptom_universe: List of all possible symptoms. If None, computed as union of all sets.
            - threshold: Unused here (for compatibility) because we use binary sets.
            - alpha: Parameter for alpha-softmax/softmin aggregation.
        """
        self.true_symptom_sets = true_symptoms
        self.predicted_symptom_sets = extracted_symptoms

        if symptom_universe is None:
            symptom_universe = sorted(set().union(*true_symptoms, *extracted_symptoms))
        self.symptom_universe = symptom_universe

        # Convert the sets into binary matrices.
        y_true = self.sets_to_binary(true_symptoms, symptom_universe)
        y_pred = self.sets_to_binary(extracted_symptoms, symptom_universe)
        super().__init__(y_true, y_pred, class_names=symptom_universe, threshold=threshold, alpha=alpha)

    
    @staticmethod
    def sets_to_binary(sets_list, symptom_universe):
        """
        Converts a list of symptom sets into a binary numpy array.
        """
        num_samples = len(sets_list)
        num_symptoms = len(symptom_universe)
        binary_matrix = np.zeros((num_samples, num_symptoms), dtype=int)
        symptom_to_idx = {sym: idx for idx, sym in enumerate(symptom_universe)}
        for i, symptom_set in enumerate(sets_list):
            for sym in symptom_set:
                if sym in symptom_to_idx:
                    binary_matrix[i, symptom_to_idx[sym]] = 1
        return binary_matrix

    def compute_set_based_metrics(self):
        """
        Computes per-sample set-based metrics: Precision, Recall, F1, and Jaccard index.
        """
        precisions, recalls, f1s, jaccards = [], [], [], []
        for true_set, pred_set in zip(self.true_symptom_sets, self.predicted_symptom_sets):
            # print(true_set , pred_set )
            true_set = set(true_set)
            pred_set = set(pred_set)

            if len(pred_set) == 0:
                
                precision = 1.0 if len(true_set) == 0 else 0.0
            else:
                precision = len(true_set.intersection(pred_set)) / len(pred_set)
            recall = len(true_set.intersection(pred_set)) / (len(true_set) if len(true_set) > 0 else 1)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            jaccard = len(true_set.intersection(pred_set)) / (len(true_set.union(pred_set)) if len(true_set.union(pred_set)) > 0 else 1)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            jaccards.append(jaccard)
        return {
            "Avg Precision (per sample)": np.mean(precisions),
            "Avg Recall (per sample)": np.mean(recalls),
            "Avg F1 (per sample)": np.mean(f1s),
            "Avg Jaccard (per sample)": np.mean(jaccards)
        }

    def evaluate(self):
        """
        Computes overall evaluation metrics, returning a dictionary with:
          - "Class-Based Metrics": Metrics computed from binary vectors.
          - "Sample-Based Metrics": Set-based (per dialogue) metrics.
        """
        base_evaluation = super().evaluate()["Class-Based Metrics"]
        set_based = self.compute_set_based_metrics()
        all_metrics = {
            "Class-Based Metrics": base_evaluation,
            "Sample-Based Metrics": set_based
        }
        # Round all numeric values to 2 decimals.
        return round_dict(all_metrics, ndigits=2)
    
def sets_to_binary(sets_list, symptom_universe):

    """
    Converts a list of symptom sets into a binary numpy array.
    """
    num_samples = len(sets_list)
    num_symptoms = len(symptom_universe)
    binary_matrix = np.zeros((num_samples, num_symptoms), dtype=int)
    symptom_to_idx = {sym: idx for idx, sym in enumerate(symptom_universe)}
    
    for i, symptom_set in enumerate(sets_list):
        for sym in symptom_set:
            if sym in symptom_to_idx:
                binary_matrix[i, symptom_to_idx[sym]] = 1
    
    return binary_matrix


def evaluation_to_dataframe(evaluation_dict):
    """
    Flattens the nested evaluation output into a pandas DataFrame with columns:
    'metric', 'value', 'type'
    
    Parameters:
        evaluation_dict (dict): output of evaluator.evaluate()
    
    Returns:
        pd.DataFrame
    """
    rows = []
    for block_type, metrics in evaluation_dict.items():  # "Class-Based Metrics", "Sample-Based Metrics"
        for metric_name, value in metrics.items():
            if isinstance(value, dict):  # Aggregated metrics (e.g., AUC with Avg / Softmax / Softmin)
                for sub_metric, v in value.items():
                    rows.append({
                        "metric": f"{metric_name} ({sub_metric})",
                        "value": v,
                        "type": block_type.replace(" Metrics", "")
                    })
            else:
                rows.append({
                    "metric": metric_name,
                    "value": value,
                    "type": block_type.replace(" Metrics", "")
                })
    return pd.DataFrame(rows)

def Evaluation_pipeline(df) : 
    
    true_symptoms = [{ast.literal_eval(el)[i] for i in range(len(ast.literal_eval(el)))} for el in list(df['True_Symptom'])]
    extracted_symptoms = [{ast.literal_eval(el)[i] for i in range(len(ast.literal_eval(el)))} for el in list(df['Extracted_Symptom'])]
    evaluator = SymptomMultiLabelEvaluator(true_symptoms, extracted_symptoms)
    evaluation_results = evaluator.evaluate()

    df_metrics = evaluation_to_dataframe(evaluation_results)

    return df_metrics


prompting_methods = [
    # "explicit",
    # "zero_shot",
    # "few_shot",
    # "chain_of_thought",
    # "self_refinement",
    # "multiple_demonstrations",
    "explicit_with_RAG",
    "zero_shot_with_RAG",
    "few_shot_with_RAG",
    "chain_of_thought_with_RAG",
    "self_refinement_with_RAG",
    "multiple_demonstrations_with_RAG"
]

data_names = [
    "dataset_extracting_multi_poisson_correl",
    "dataset_extracting_multi_poisson",
    "dataset_extracting_multi_poisson_correl",
    "dataset_extracting_multi_predef",
    "dataset_extracting_one_symptom"
]


for data_name in data_names :

    list_all = []

    for method in prompting_methods:

        try:

            path = f"{data_name}_{method}.csv"
            
            df_path = pd.read_csv(path)
            df_path['Extracted_Symptom'] = df_path['Extracted_Symptom'].apply(lambda x: [s.strip() for s in str(x).split(',')])
            df_path['Extracted_Symptom'] = df_path['Extracted_Symptom'].apply(lambda lst: str(lst))
            
            result = Evaluation_pipeline(df_path)

            list_all.append(result[['metric', 'value']].set_index('metric').rename(columns={'value': f'value_{method}'}))
        
        except Exception as e : 
            print(e)
        
    try : 
        l = pd.concat(list_all, axis=1).reset_index()
        l.to_csv(f"comparison_prompting_methods_{data_name}.csv")

    except Exception as e : 
        print(e)


