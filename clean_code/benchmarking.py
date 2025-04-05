import pandas as pd
from transformers import pipeline
import ast
from extractor import ExtractingPrompt
from LLM import LLM
from tqdm import tqdm
import re
import numpy as np
from extractor import ExtractingPrompt 

import warnings
warnings.filterwarnings('ignore')


class LLMBenchmark:

    def __init__(self, model_names, extractor, df, extract_scores_fn, output_file_prefix="results"):
        self.model_names = model_names
        self.extractor = extractor
        self.df = df
        self.extract_scores_fn = extract_scores_fn
        self.output_file_prefix = output_file_prefix
        self.results = {}

    def evaluate_model(self, model_name):
        
        print(f"Evaluating {model_name}")
        model = LLM(model_name=model_name, max_length=50)
        results = []

        for i, phrase in tqdm(enumerate(self.df['Dialogue_Generated']), total=len(self.df)):

            prompt = self.extractor.build_extraction_prompt(phrase)
            generated_output = model.generate_text(messages=prompt)

            symptom_scores = self.extract_scores_fn(generated_output)
            extracted_symptoms = [s for s, score in symptom_scores.items() if score > 0.80]

            if not extracted_symptoms:
                formatted = None

            elif len(extracted_symptoms) == 1:
                formatted = extracted_symptoms[0]
                
            else:
                formatted = ", ".join(extracted_symptoms)

            results.append({
                "Dialogue": phrase,
                "True_Symptom": self.df['Symptom'][i],
                "Extracted_Symptom": formatted
            })

            df_results = pd.DataFrame(results)

            df_results.to_csv(f"{self.output_file_prefix}_{model_name.replace('/', '_')}.csv", index=False)

            self.results[model_name] = df_results

        return df_results

    def calculate_accuracy(self, df_results):
        """
        Calculate accuracy of symptom extraction, case-insensitive and with proper handling of extracted symptoms.
        
        Args:
            df_results: DataFrame containing columns 'True_Symptom' and 'Extracted_Symptom'
            
        Returns:
            accuracy: Float between 0 and 1 representing match accuracy
        """
        score = 0
        total = len(df_results)
        
        if total == 0:
            return 0.0  # handle empty dataframe case
        
        for i in range(total):
            true_symptom = str(df_results.iloc[i]['True_Symptom']).strip().lower()
            
            try:
                # Safely handle the extracted symptom (assuming it might be a string representation of a list)
                extracted = df_results.iloc[i]['Extracted_Symptom']
                
                # Handle different possible formats:
                if isinstance(extracted, str):

                    # Try to evaluate if it's a string representation of a list
                    if extracted.startswith('[') and extracted.endswith(']'):
                        extracted_list = [s.strip().lower() for s in extracted[1:-1].split(',')]
                    else:
                        extracted_list = [extracted.strip().lower()]

                elif isinstance(extracted, (list, tuple)):
                    extracted_list = [str(s).strip().lower() for s in extracted]
                    
                else:
                    extracted_list = [str(extracted).strip().lower()]
                    
                # Check if true symptom exists in extracted list
                if true_symptom in extracted_list:
                    score += 1
                    
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                continue
        
        accuracy = score / total
        return accuracy

    def run(self):
        all_metrics = {}

        for model_name in self.model_names:
            df_result = self.evaluate_model(model_name)
            accuracy = self.calculate_accuracy(df_result)
            all_metrics[model_name] = accuracy

        return pd.DataFrame(all_metrics).T  # Summary DataFrame


def extract_symptom_scores(output_str):
    # This pattern matches a key enclosed in single or double quotes followed by a colon and a number (integer or float)
    pattern = r'["\']([^"\']+)["\']\s*:\s*([0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, output_str)
    # Convert the extracted values to float and build the dictionary
    return {key: float(value) for key, value in matches}


df = pd.read_csv('/home/laajila/mima_newcode/clean_code/Final_dataset_generated_one_symptom.csv')
df_small = df[:500]

ctcae = pd.read_excel('PRO-CTCAE_Questionnaire_Terminology.xls', sheet_name = 'PRO')
symptoms_list = ctcae['PRO-CTCAE PT'].unique()[:-25]

extractor = ExtractingPrompt(symptoms_list)
model_names = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "iRASC/BioLlama-Ko-8B"
]


benchmark = LLMBenchmark(
    model_names=model_names,
    extractor=extractor,
    df=df,
    extract_scores_fn=extract_symptom_scores,
    output_file_prefix="symptom_extraction"
)

summary_df = benchmark.run()
print(summary_df)