import pandas as pd
from transformers import pipeline
import ast
from extracting_prompt import ExtractingPrompt
from LLM import LLM
from tqdm import tqdm
import re
import numpy as np
import string

df_onesymptom = pd.read_csv(r'dataset_generated_one_symptom_per_phrase.csv')
df_onesymptom.rename(columns={"symptom": "Symptoms"}, inplace=True)

df_multi_predef = pd.read_csv(r'dataset_generated_multiple_symptoms_per_phrase_predefined_distrib.csv')
df_multi_poisson = pd.read_csv(r'dataset_generated_multiple_symptoms_per_phrase_poisson_distrib.csv')

df_multi_poisson_correl = pd.read_csv(r'dataset_generated_multiple_symptom_per_phrase_poisson_correlations.csv')

ctcae = pd.read_excel('PRO-CTCAE_Questionnaire_Terminology.xls', sheet_name = 'PRO')
symptoms_list = ctcae['PRO-CTCAE PT'].unique()[:-25]
symptoms_list = symptoms_list.tolist()  
symptoms_list.remove('Other Symptoms')


#df_lists = {"multi_poisson_correl" : df_multi_poisson_correl, "multi_poisson" : df_multi_poisson, "multi_predef": df_multi_predef, "one_symptom": df_onesymptom}

df_lists = {"one_symptom": df_onesymptom}

def extract_symptom_scores(output_str):
    # This pattern matches a key enclosed in single or double quotes followed by a colon and a number (integer or float)
    pattern = r'["\']([^"\']+)["\']\s*:\s*([0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, output_str)
    # Convert the extracted values to float and build the dictionary
    return {key: float(value) for key, value in matches}
            
def true_vs_extracted_symptoms(df_results):
    
    true_symptoms = [{el} for el in list(df_results['True_Symptom'])]
    extracted_symptoms = [{ast.literal_eval(el)[i] for i in range(len(ast.literal_eval(el)))} for el in list(df_results['Extracted_Symptom'])]
    return true_symptoms, extracted_symptoms


extractor = ExtractingPrompt(symptom_list=symptoms_list)

model = LLM(model_name="iRASC/BioLlama-Ko-8B", max_length=50)

# List of prompting methods to test:
prompting_methods = [
    "explicit",
    "zero_shot",
    "few_shot",
    "chain_of_thought",
    "self_refinement",
    "multiple_demonstrations"
]

for name, df in (df_lists).items():
    
    print(name)

    # Loop over each prompting method
    for method in prompting_methods:

        results = []
        print(f"Processing method: {method}")

        for i, phrase in tqdm(enumerate(df['Dialogue_Generated']), total=len(df)):
            # Build prompt for the current dialogue using the current method
            prompt = extractor.build_extraction_prompt(phrase, method=method)
            
            # Generate text using the LLM
            symptoms_extracted_llm = model.generate_text(messages=prompt)
            
            # Extract symptoms with scores > threshold (here 0.6)
            symptom_scores = extract_symptom_scores(symptoms_extracted_llm)
            symptoms_extracted = [symptom for symptom, score in symptom_scores.items() if score > 0.6]
            
            # Format the output: if multiple symptoms, join them with a comma.
            if len(symptoms_extracted) == 0:
                formatted_symptoms = None  
            elif len(symptoms_extracted) == 1:
                formatted_symptoms = symptoms_extracted[0]
            else:
                formatted_symptoms = ", ".join(symptoms_extracted)
 
            true_symptom = df['Symptoms'][i]
            results.append({
                "Dialogue": phrase,
                "True_Symptom": true_symptom,
                "Extracted_Symptom": formatted_symptoms
            })
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(f"dataset_extracting_{name}_{method}.csv", index=False)

        print(f"Saved dataset for method '{method}' to dataset_extracting_{name}_{method}.csv")
