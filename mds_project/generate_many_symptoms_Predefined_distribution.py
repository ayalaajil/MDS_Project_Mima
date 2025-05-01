import pandas as pd
import numpy as np
import random
from autocorrect import Speller
import itertools
import re
import json
from LLM import LLM
from PromptBuilder import PromptBuilder
from metadata import language_registers, discussion_tones

def get_combinations(symptom, symptom_data):
    # Extract the attribute lists for the symptom
    attributes = symptom_data[symptom]
    # Get the attribute names (keys) and corresponding lists (values)
    attribute_names = list(attributes.keys())
    attribute_values = list(attributes.values())
    
    # Generate all combinations using itertools.product
    combinations = list(itertools.product(*attribute_values))
    
    # Optionally, pair each combination with its corresponding attribute names:
    combinations_named = [
        dict(zip(attribute_names, combination)) for combination in combinations
    ]
    
    return combinations_named

a = pd.read_excel(r'PRO-CTCAE_Questionnaire_Terminology.xls', sheet_name = 'PRO')
index = (a [a['PRO-CTCAE PT'] == 'Pain and swelling at injection site']).index.values[0] +1 
symptoms = a.iloc[:index]['PRO-CTCAE PT'].values

Dict = {}

for symptom in symptoms : 

    try:
        Dict[symptom] = {}
        descriptions_code = a[a['PRO-CTCAE PT'] == symptom]['Has PRO-CTCAE Attribute Code'].values[0].split(" || ")
        descriptions = a [a['PRO-CTCAE PT'] == symptom]['Has PRO-CTCAE Attribute PT'].values[0].split(" || ")
        for description in descriptions :
            Dict[symptom][description]  = a[a['PRO-CTCAE PT'] == description  ]['PRO-CTCAE Value PT'].values[0].split(" || ")
    
    except Exception as e :
        print(e)

for el in Dict.keys():
    for el2 in Dict[el].keys():
        if "Not sexually active" in Dict[el][el2]:
            Dict[el][el2].remove("Not sexually active")

del Dict['Other Symptoms']

print("The dictionary has been cleaned and is ready to be used")

model = LLM()
prompt_builder = PromptBuilder()

# Total number of phrases to generate
num_phrases = 1000
data = []

symptom_list = list(Dict.keys())

for i in range(num_phrases):
    # --- Step 1: Decide how many symptoms to include
    num_symptoms = np.random.choice([1,2,3,4], p=[0.3, 0.4, 0.2, 0.1])
    
    # --- Step 2: Select symptoms randomly (without replacement)
    selected_symptoms = random.sample(symptom_list, num_symptoms)
    
    description_list = []
    meta_list = []
    
    # --- Step 3: For each selected symptom, pick one random attribute combination.
    for symptom in selected_symptoms:
        combinations = get_combinations(symptom, Dict)
        chosen_combination = random.choice(combinations)
        
        # Build a description string: list the symptom and its attribute types.
        attribute_names = list(chosen_combination.keys())
        description_list.append(f"{symptom} (attributes: {', '.join(attribute_names)})")
        
        # Build a meta string: list each attribute with its chosen value.
        attr_meta = ", ".join([f"{k}: {v}" for k, v in chosen_combination.items()])
        meta_list.append(f"{symptom} -> {attr_meta}")
    
    # Combine descriptions for the prompt.
    description_str = " , ".join(description_list)
    meta_str = " ,  ".join(meta_list)
    
    # --- Step 4: Choose additional parameters for prompt generation.
    detail_level = np.random.choice([1, 2, 3, 4, 5])
    enumeration = np.random.choice([True, False], p=[0.2, 0.8])
    explicit_symptom = np.random.choice([True, False], p=[0.2, 0.8])
    language_style = random.choice(language_registers)['name']
    tone = random.choice(discussion_tones)['name']
    spelling_errors = np.random.choice([True, False], p=[0.3, 0.7])
    
    # --- Step 5: Build the prompt.
    prompt = PromptBuilder().build_prompt(
        symptoms=selected_symptoms,
        meta_str=meta_str,
        detail_level=detail_level,
        enumeration=enumeration,
        explicit_symptom=explicit_symptom,
        language_style=language_style,
        spelling_errors=spelling_errors,
        tone=tone
    )


    
    # --- Step 6: Generate text using your LLM.
    phrase_generated = model.generate_text(messages=prompt)
    
    # --- Step 7: Append the data.
    data.append([
        phrase_generated,
        selected_symptoms,
        description_str,
        meta_str,
        language_style,
        tone,
        detail_level,
        enumeration,
        explicit_symptom,
        spelling_errors
    ])

# Create a DataFrame from the collected data.
df = pd.DataFrame(data, columns=[
    "Dialogue_Generated", "Symptoms", "Description", "Meta",
    "Language_Style", "Tone", "Detail_Level", "Enumeration",
    "Explicit_Symptom", "Spelling_Errors"
])

# Optionally, save the DataFrame to a CSV file.
df.to_csv("dataset_generated_multiple_symptoms_per_phrase_predefined_distrib.csv", index=False)

print("Dataset of 1000 phrases generated and saved to 'dataset_generated_multiple_symptoms_per_phrase_predefined_distrib.csv'.")
