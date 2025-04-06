import pandas as pd
import numpy as np
import random
from autocorrect import Speller
import itertools
import re
from LLM import LLM
from PromptBuilder import PromptBuilder
from metadata import language_registers, discussion_tones



model = LLM()
prompt_builder = PromptBuilder()

a = pd.read_excel('/home/laajila/mima_newcode/mds_project/PRO-CTCAE_Questionnaire_Terminology.xls', sheet_name = 'PRO')
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

data = []
mean = 3
std = 1
c = 0

symptom_list = list(Dict.keys())

# Define how many symptoms per sentence (e.g., 2 to 4 symptoms)
min_symptoms = 2
max_symptoms = 5

for _ in range(1000): 

    # Randomly sample symptoms
    num_symptoms = int(np.clip(np.random.normal(loc=mean, scale=std), min_symptoms, max_symptoms))

    selected_symptoms = random.choices(symptom_list,k=num_symptoms)

    descriptions = []
    metas = []

    # Get description + meta info for each selected symptom
    for symptom in selected_symptoms:

        descriptions_dict = Dict[symptom]

        description = random.choice(list(descriptions_dict.keys()))

        meta_options = descriptions_dict[description]

        meta = random.choice(meta_options)
        
        descriptions.append((symptom, description))
        metas.append(meta)

    if c % 100 == 0:
        print(f"Generated {c+100} sentences...")

    # Parameters shared across the sentence

    detail_level = np.random.choice([1, 2, 3, 4, 5])
    enumeration = np.random.choice([True, False], p=[0.2, 0.8])
    explicit_symptom = np.random.choice([True, False], p=[0.2, 0.8])
    language_style = random.choice(language_registers)['name']
    tone = random.choice(discussion_tones)['name']
    spelling_errors = random.choice([True, False])

    # Build the prompt using all symptoms
    symptoms_only = [s[0] for s in descriptions]
    descriptions_only = [s[1] for s in descriptions]
    
    prompt = prompt_builder.build_prompt(
        symptoms=symptoms_only,
        # description=descriptions_only,  # assumes you support a list of descriptions
        meta_str=metas,
        detail_level=detail_level,
        enumeration=enumeration,
        explicit_symptom=explicit_symptom,
        language_style=language_style,
        spelling_errors=spelling_errors,
        tone=tone
    )

    phrase_generated = model.generate_text(messages=prompt)

    data.append([
        phrase_generated, symptoms_only, descriptions_only, metas, 
        language_style, tone, detail_level, enumeration, 
        explicit_symptom, spelling_errors
    ])
    c += 1

    df = pd.DataFrame(data, columns=[
        "Dialogue_Generated", "Symptoms", "Descriptions", "Metas", 
        "Language_Style", "Tone", "Detail_Level", "Enumeration", 
        "Explicit_Symptom", "Spelling_Errors"
    ])
    df.to_csv("Multi_symptoms_1.csv")