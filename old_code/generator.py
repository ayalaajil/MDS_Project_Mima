import pandas as pd
import numpy as np
import random
from autocorrect import Speller
import re
from LLM import LLM
from PromptBuilder import PromptBuilder
from metadata import language_registers, discussion_tones

prompt_builder = PromptBuilder()

model = LLM(model_name="iRASC/BioLlama-Ko-8B")

a = pd.read_excel('PRO-CTCAE_Questionnaire_Terminology.xls', sheet_name = 'PRO')

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

data = []

for symptom in Dict.keys():

    for description in Dict[symptom].keys() : 

        for meta in Dict[symptom][description] : 
    
            detail_level = np.random.choice([1,2,3,4,5])

            enumeration = np.random.choice([True, False], p=[0.2, 0.8])

            explicit_symptom = np.random.choice([True, False], p=[0.2, 0.8])

            language_style = random.choice(language_registers)['name']

            tone = random.choice(discussion_tones)['name']

            spelling_errors = random.choice([True, False])

            prompt = prompt_builder.build_prompt([symptom], description, meta, detail_level, enumeration, explicit_symptom, language_style, spelling_errors, tone) 
        
            phrase_generated = model.generate_text(messages = prompt)

            data.append([phrase_generated , symptom, description, meta, language_style, tone, detail_level, enumeration, explicit_symptom, spelling_errors])

df = pd.DataFrame(data, columns=["Dialogue_Generated", "symptom", "description", "meta", "language_style", "Tone", "Detail_level", "Enumeration", "Explicit_symptom", "Spelling_errors"])