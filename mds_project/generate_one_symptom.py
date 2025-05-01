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
    
    # Pair each combination with its corresponding attribute names:
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

data = []

counter = 0  # Initialize the counter

total = sum(len(get_combinations(symptom, Dict)) for symptom in Dict.keys())  # Total combinations


for symptom in Dict.keys():

    combinaisons = get_combinations(symptom, Dict)

    for combi in combinaisons :

        descriptions =  list(combi.keys())

        meta_descriptions = list(combi.items())

        meta_str = " , ".join([f"{k} = {v}" for k, v in meta_descriptions])
        
        detail_level = np.random.choice([1,2,3,4,5])

        enumeration = np.random.choice([True, False], p=[0.2, 0.8])

        explicit_symptom = np.random.choice([True, False], p=[0.2, 0.8])

        language_style = random.choice(language_registers)['name']

        tone = random.choice(discussion_tones)['name']

        spelling_errors = np.random.choice([True, False], p=[0.3, 0.7])

        prompt = prompt_builder.build_prompt([symptom], meta_str, detail_level, enumeration, explicit_symptom, language_style, spelling_errors,
         tone) 
   
        phrase_generated = model.generate_text(messages = prompt)

        data.append([phrase_generated , symptom, descriptions, meta_str, language_style, tone, detail_level, enumeration, explicit_symptom, spelling_errors])

        
        df = pd.DataFrame(data, columns=["Dialogue_Generated", "symptom", "description", "meta", "language_style", "Tone", "Detail_level", "Enumeration", "Explicit_symptom", "Spelling_errors"])
        
        df.to_csv("dataset_generated_one_symptom_per_phrase.csv")

        counter += 1
        print(f"{counter}/{total} phrases generated", end='\r')  # Real-time update