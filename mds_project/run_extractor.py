import pandas as pd
from transformers import pipeline
import ast
from extracting_prompt import ExtractingPrompt
from LLM import LLM
from tqdm import tqdm
import re
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df_onesymptom = pd.read_csv('dataset_generated_one_symptom_per_phrase.csv')
df_multi_predef = pd.read_csv('dataset_generated_multiple_symptoms_per_phrase_predefined_distrib.csv')
df_multi_poisson = pd.read_csv('dataset_generated_multiple_symptoms_per_phrase_poisson_distrib.csv')
df_multi_poisson_correl = pd.read_csv('dataset_generated_multiple_symptom_per_phrase_poisson_correlations.csv')


ctcae = pd.read_excel('PRO-CTCAE_Questionnaire_Terminology.xls', sheet_name='PRO')
symptoms_list = ctcae['PRO-CTCAE PT'].unique()[:-25].tolist()
symptoms_list.remove('Other Symptoms')


model = LLM(model_name="iRASC/BioLlama-Ko-8B", max_length=50)
extractor = ExtractingPrompt(symptoms_list)


def extract_symptoms_from_dataset(df: pd.DataFrame, dialogue_column: str, label_column: str, output_file: str) -> pd.DataFrame:
    results = []

    for i, phrase in tqdm(enumerate(df[dialogue_column]), total=len(df)):

        prompt = extractor.build_extraction_prompt(phrase)
        try:
            response = model.generate_text(messages=prompt)
            symptom_scores = extract_symptom_scores(response)
        except Exception as e:
            symptom_scores = {}
        
        # Extract symptoms with score > 0.60
        symptoms_extracted = [symptom for symptom, score in symptom_scores.items() if score > 0.60]
        
        # Format extracted symptoms
        if not symptoms_extracted:
            formatted_symptoms = None
        elif len(symptoms_extracted) == 1:
            formatted_symptoms = symptoms_extracted[0]
        else:
            formatted_symptoms = ", ".join(symptoms_extracted)
        
        results.append({
            "Dialogue": phrase,
            "True_Symptom": df[label_column][i],
            "Extracted_Symptom": formatted_symptoms,
            "Raw_Output": response,
            "Parsed_Symptoms": symptom_scores
        })

        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
    return df_results

def extract_symptom_scores(output_str):
    pattern = r'["\']([^"\']+)["\']\s*:\s*([0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, output_str)
    return {key: float(value) for key, value in matches}


datasets = [
    {
        "df": df_onesymptom,
        "dialogue_column": "Dialogue_Generated",
        "label_column": "Symptoms",
        "output_file": "extracted_onesymptom.csv"
    },
    {
        "df": df_multi_predef,
        "dialogue_column": "Dialogue_Generated",
        "label_column": "Symptoms",
        "output_file": "extracted_multi_predef.csv"
    },
    {
        "df": df_multi_poisson,
        "dialogue_column": "Dialogue_Generated",
        "label_column": "Symptoms",
        "output_file": "extracted_multi_poisson.csv"
    },
    {
        "df": df_multi_poisson_correl,
        "dialogue_column": "Dialogue_Generated",
        "label_column": "Symptoms",
        "output_file": "extracted_multi_poisson_correl.csv"
    }
]


for dataset in datasets:

    print(f"Processing {dataset['output_file']}...")
    extract_symptoms_from_dataset(
        df=dataset["df"],
        dialogue_column=dataset["dialogue_column"],
        label_column=dataset["label_column"],
        output_file=dataset["output_file"]
    )