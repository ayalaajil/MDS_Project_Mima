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
from evaluation_global import Evaluation_pipeline
import warnings
warnings.filterwarnings('ignore')


# Helper to convert cell to float if possible
def safe_to_float(x):
    try:
        # If it's a list, try to extract the first element
        if isinstance(x, list) and len(x) == 1:
            return float(x[0])
        elif isinstance(x, (int, float, np.float64)):
            return x
        else:
            return float(x)
    except:
        return np.nan  # mark as missing if not convertible
    
metadatas = ["Language_Style", "Tone", "Detail_Level", "Enumeration", "Explicit_Symptom", "Spelling_Errors"]




df_onesymptom = pd.read_csv(r'dataset_generated_one_symptom_per_phrase.csv')
df_onesymptom.rename(columns={"symptom": "Symptoms"}, inplace=True)

df_multi_predef = pd.read_csv(r'dataset_generated_multiple_symptoms_per_phrase_predefined_distrib.csv')
df_multi_poisson = pd.read_csv(r'dataset_generated_multiple_symptoms_per_phrase_poisson_distrib.csv')

df_multi_poisson_correl = pd.read_csv(r'dataset_generated_multiple_symptom_per_phrase_poisson_correlations.csv')


df_lists = {"multi_poisson_correl" : df_multi_poisson_correl, "multi_poisson" : df_multi_poisson, "multi_predef": df_multi_predef, "one_symptom": df_onesymptom}


prompting_methods = [
    "explicit",
    "zero_shot",
    "few_shot",
    "chain_of_thought",
    "self_refinement",
    "multiple_demonstrations"
]


for el,df in df_lists.items():

    for method in prompting_methods:

        try :

            path = f'dataset_extracting_{el}_{method}.csv'
            data_generated = df

            print('Running the evaluation sensibility for ', el,' and prompting method : ', method)


            df_extracted = pd.read_csv(r'./'+path)
        
            df_extracted['Extracted_Symptom'] = df_extracted['Extracted_Symptom'].apply(lambda x: [s.strip() for s in str(x).split(',')])
            df_extracted['Extracted_Symptom'] = df_extracted['Extracted_Symptom'].apply(lambda lst: str(lst))

        
            merged_df = pd.merge(df_extracted, data_generated, left_on='Dialogue', right_on='Dialogue_Generated')
                
            final_df = {}

            for metadata in metadatas : 

                print('The meta data we check is : ' , metadata)
                list_all = []
                grouped = merged_df.groupby(metadata)
                for group_name, group_df in grouped:
                    
                    eval_df = Evaluation_pipeline(group_df)
                    
                    list_all.append(eval_df[['metric', 'value']].set_index('metric').rename(columns={'value': f'value_{group_name}'}))

                final_df[metadata] = pd.concat(list_all, axis=1).reset_index()

                df_plot = final_df[metadata]
                
                # Apply the cleaning
                for col in df_plot.columns[1:]:
                    df_plot[col] = df_plot[col].apply(safe_to_float)



                metrics = df_plot['metric'].values
                group_columns = df_plot.columns[1:]
                x = np.arange(len(metrics))
                width = 0.8 / len(group_columns)

                fig, ax = plt.subplots(figsize=(10, 6))

                for i, group in enumerate(group_columns):
                    offset = (i - (len(group_columns) - 1) / 2) * width
                    y_values = df_plot[group].values.astype(float)
                    ax.bar(x + offset, y_values, width, label=group)

                ax.set_xticks(x)
                ax.set_xticklabels(metrics, rotation=45, ha='right')
                ax.set_ylabel('Score')
                ax.set_title(f'Scores of Each Metric per Group for {metadata}')
                ax.legend()
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)

                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(e)