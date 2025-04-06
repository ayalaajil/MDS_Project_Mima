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
import json
from generate_many_symptoms_1 import truncated_poisson

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

prompt_builder = PromptBuilder()
model = LLM()


json_file_path = "symptom_correlations.json"

with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Print the loaded JSON to verify its contents
print(json.dumps(data, indent=2))

correlation_matrix = json.dumps(data, indent=2)
correlation_matrix = json.loads(correlation_matrix)

def smart_select_symptoms(num_symptoms, symptom_list, correlation_matrix):
    """
    Select a set of symptoms by using the upper triangular correlation matrix to weight the selections.
    
    Args:
        num_symptoms (int): The number of symptoms to select.
        symptom_list (list): The ordered list of all symptom names.
        correlation_matrix (dict): A nested dictionary representing the upper triangular matrix.
                                   For any two symptoms s1 and s2 (with s1 before s2 in symptom_list),
                                   correlation_matrix[s1][s2] gives their correlation.
    
    Returns:
        list: A list of selected symptom names.
    """
    # Step 1: Select the first symptom uniformly at random.
    selected = []
    first_symptom = random.choice(symptom_list)
    selected.append(first_symptom)
    
    # Step 2: Iteratively select the remaining symptoms.
    while len(selected) < num_symptoms:
        candidates = [s for s in symptom_list if s not in selected]
        candidate_weights = []
        
        for candidate in candidates:
            weight = 1.0
            for sel in selected:
                # Determine the order of 'sel' and 'candidate' in the original symptom_list.
                if symptom_list.index(sel) < symptom_list.index(candidate):
                    # sel comes before candidate, so look up directly.
                    corr = correlation_matrix.get(sel, {}).get(candidate, 0.5)
                else:
                    # candidate comes before sel, so look up using candidate as the key.
                    corr = correlation_matrix.get(candidate, {}).get(sel, 0.5)
                weight *= corr
            candidate_weights.append(weight)
        
        # Normalize the weights to form a probability distribution.
        total_weight = sum(candidate_weights)
        probabilities = [w / total_weight for w in candidate_weights]
        
        # Sample one candidate based on these probabilities.
        next_symptom = random.choices(candidates, weights=probabilities, k=1)[0]
        selected.append(next_symptom)
    
    return selected


#  Create a list of all symptom names.
symptom_list = list(Dict.keys())

# Loop to generate 1000 phrases.
data = []

num_phrases = 1000

for i in range(num_phrases):

    # Determine how many symptoms to include
    num_symptoms = truncated_poisson(lam=1.5, min_val=1, max_val=5)
    
    # Use the smart selection based on the correlation matrix.
    selected_symptoms = smart_select_symptoms(num_symptoms, symptom_list, correlation_matrix)
    
    description_list = []
    meta_list = []
    
    # For each selected symptom, get a random attribute combination.
    for symptom in selected_symptoms:
        combinations = get_combinations(symptom, Dict)
        chosen_combination = random.choice(combinations)
        attribute_names = list(chosen_combination.keys())
        description_list.append(f"{symptom} (attributes: {', '.join(attribute_names)})")
        attr_meta = ", ".join([f"{k}: {v}" for k, v in chosen_combination.items()])
        meta_list.append(f"{symptom} -> {attr_meta}")
    
    # Combine descriptions and meta information.
    description_str = "; ".join(description_list)
    meta_str = "; ".join(meta_list)
    
    # Random stylistic parameters.
    detail_level = np.random.choice([1, 2, 3, 4, 5])
    enumeration = np.random.choice([True, False], p=[0.2, 0.8])
    explicit_symptom = np.random.choice([True, False], p=[0.2, 0.8])
    language_style = random.choice(language_registers)['name']
    tone = random.choice(discussion_tones)['name']
    spelling_errors = np.random.choice([True, False], p=[0.3, 0.7])
    
    # Build the prompt.
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
    
    # Generate the phrase using the model.
    phrase_generated = model.generate_text(messages=prompt)
    
    # Append all data to our list.
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

df = pd.DataFrame(data, columns=[
    "Dialogue_Generated", "Symptoms", "Description", "Meta", 
    "Language_Style", "Tone", "Detail_Level", "Enumeration", 
    "Explicit_Symptom", "Spelling_Errors"
])

df.to_csv("dataset_generated_multiple_symptom_per_phrase_poisson_correlations.csv", index=False)
print("Dataset of 1000 phrases generated and saved.")

