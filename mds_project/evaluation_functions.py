import pandas as pd
import numpy as np
import ast

def calculate_accuracy_degree_1(true_symptoms, extracted_symptoms, df_results):  

    score = 0
    for i in range(len(df_results)) : 

        True_symptom = true_symptoms[i]
        Extracted_Symptom = extracted_symptoms[i]
        
        if True_symptom == Extracted_Symptom :
            score +=1 

    accuracy = score / len(df_results)
    return accuracy

# we extract the symptom with the highest score, when dealing with many symptoms
def calculate_accuracy_degree_1_bis(df_results):

    score = 0
    for i in range(len(df_results)) : 
        True_symptom = df_results.iloc[i]['True_Symptom']
        Extracted_Symptom = ast.literal_eval(df_results.iloc[i]['Extracted_Symptom'])[0]

        if True_symptom == Extracted_Symptom :
            score +=1 

    accuracy = score / len(df_results)
    return accuracy


# * If the model extracts extra symptoms, it lowers precision.
# * If the model misses true symptoms, recall decreases.

def evaluate_penalization_degree_1(true_symptoms, extracted_symptoms):
    
    # Compute precision (only penalizes false positives)
    precision_scores = [
        len(pred & true) / len(pred) if pred else 0 
        for true, pred in zip(true_symptoms, extracted_symptoms)
    ]

    # Compute recall (penalizes missing true symptoms)
    recall_scores = [
        len(pred & true) / len(true) if true else 0 
        for true, pred in zip(true_symptoms, extracted_symptoms)
    ]

    # Compute F1-score (avoiding division by zero)
    f1_scores = [
        (2 * p * r) / (p + r) if (p + r) > 0 else 0
        for p, r in zip(precision_scores, recall_scores)
    ]

    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)
    f1_score = np.mean(f1_scores)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")


# Consider True Positives when at least one true symptom is extracted

def evaluate_penalization_degree_0(true_symptoms, extracted_symptoms):
    
    # Compute True Positives: At least one true symptom is extracted

    true_positives = [
        1 if len(pred & true) > 0 else 0  # If there's at least one correct symptom
        for true, pred in zip(true_symptoms, extracted_symptoms)
    ]

    # Compute Precision: Is at least one extracted symptom correct?

    precision_scores = [
        1 if len(pred & true) > 0 else 0  # If at least one match, it's a precision hit
        for true, pred in zip(true_symptoms, extracted_symptoms)
    ]

    # Compute Recall: Did the model miss all true symptoms?

    recall_scores = [
        1 if len(pred & true) > 0 else 0  # If at least one match, recall is 1
        for true, pred in zip(true_symptoms, extracted_symptoms)
    ]

    # Compute F1-Score (since precision and recall are binary in this case)

    f1_scores = [
        1 if len(pred & true) > 0 else 0  # If at least one match, F1 is 1
        for true, pred in zip(true_symptoms, extracted_symptoms)
    ]

    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)
    f1_score = np.mean(f1_scores)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")


