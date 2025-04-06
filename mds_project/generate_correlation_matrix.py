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

def build_correlation_matrix_with_llm(symptom_list):

    prompt = [
        {
            "role": "system",
            "content": "You are an expert medical consultant with deep knowledge of symptom co-occurrence in patients."
        },
        {
            "role": "user",
            "content": (
                f"Here is a list of symptoms: {symptom_list}.\n"
                "For each pair of symptoms (only for pairs where the first symptom appears earlier in the list than the second), "
                "please provide a correlation score between 0 and 1, where 0 means 'completely unrelated' and 1 means 'highly likely to co-occur.' "
                "Return your answer as a valid JSON object. The JSON should have each symptom as a key, and the value should be another JSON object "
                "mapping only the symptoms that come after it in the list to their correlation score. "
                "Do not include any additional text or explanation. Make sure the JSON is complete and has no trailing commas."
            )
        }
    ]
    
    response = model2.generate_text(messages=prompt) 
    
    if response.startswith("```json"):
        response = response[len("```json"):].strip()
    if response.endswith("```"):
        response = response[:-3].strip()

    # Find the end marker and extract JSON.
    if "###END###" in response:
        response = response.split("###END###")[0].strip()
    
    try:
        correlation_matrix = json.loads(response)
    except Exception as e:
        print("Failed to parse JSON, using default correlations.", e)
        correlation_matrix = {}
    
    return correlation_matrix

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

model2 = LLM(max_length= 10000)
correlation_matrix = build_correlation_matrix_with_llm(list(Dict.keys()))

print("LLM-generated correlation matrix:", json.dumps(correlation_matrix, indent=2))