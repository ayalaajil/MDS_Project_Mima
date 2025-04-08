import pandas as pd
import numpy as np
import ast
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ExtractingPrompt_rag import ExtractingPrompt_rag  
from LLM import LLM


# Load the symptom list from your PRO-CTCAE file
ctcae = pd.read_excel('PRO-CTCAE_Questionnaire_Terminology.xls', sheet_name='PRO')
symptoms_list = ctcae['PRO-CTCAE PT'].unique()[:-25].tolist()
if 'Other Symptoms' in symptoms_list:
    symptoms_list.remove('Other Symptoms')


# Initialize the extraction prompt generator and LLM
extractor = ExtractingPrompt_rag(symptom_list=symptoms_list)
model = LLM(model_name="iRASC/BioLlama-Ko-8B", max_length=50)

def extract_symptom_scores(output_str: str) -> dict:
    """
    Extracts key:score pairs from the LLM output using a regex pattern.
    """
    pattern = r'["\']([^"\']+)["\']\s*:\s*([0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, output_str)
    return {key: float(value) for key, value in matches}



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



for name, df in (df_lists).items():
    
    print(name)

    # Split the dataset into training (knowledge base) and test subsets (e.g., 80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Build a TF-IDF vectorizer on the training dialogues for retrieval
    vectorizer = TfidfVectorizer(stop_words='english')
    train_dialogues = train_df['Dialogue_Generated'].tolist()
    train_vectors = vectorizer.fit_transform(train_dialogues)

    # Loop over each prompting method
    for method in prompting_methods:

        def retrieve_context(test_dialogue: str, top_k: int = 3) -> str:

            """
            Retrieve the top_k most similar training dialogues and their true symptoms as context.
            Returns a string containing the retrieved examples.
            """
            test_vector = vectorizer.transform([test_dialogue])
            sims = cosine_similarity(test_vector, train_vectors).flatten()
            top_indices = sims.argsort()[-top_k:][::-1]
            context_examples = []
            for idx in top_indices:
                retrieved_dialogue = train_dialogues[idx]
                # Get the true symptoms for this training entry
                true_symptoms = train_df.iloc[idx]['Symptoms']
                context_examples.append(f"Dialogue: \"{retrieved_dialogue}\"\nTrue Symptoms: {true_symptoms}")
            return "\n\n".join(context_examples)

        results = []

        # Process each test dialogue using the RAG-style pipeline
        for i, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
            dialogue = row['Dialogue_Generated']
            # Retrieve context from the training set (similar dialogues and their true symptoms)
            context = retrieve_context(dialogue, top_k=3)
            # Build a prompt that includes the retrieved context plus the test dialogue.
         
            prompt = extractor.build_extraction_prompt(dialogue, method= method, context=context)
            # Generate extraction output from the LLM
            symptoms_extracted_llm = model.generate_text(messages=prompt)

            # Extract symptom scores from the generated output using regex
            symptom_scores = extract_symptom_scores(symptoms_extracted_llm)
            # Keep only symptoms with a confidence score above 0.6
            extracted_symptoms = [sym for sym, score in symptom_scores.items() if score > 0.6]
            # Format the output: if multiple symptoms, join them by commas.
            if len(extracted_symptoms) == 0:
                formatted_symptoms = None  
            elif len(extracted_symptoms) == 1:
                formatted_symptoms = extracted_symptoms[0]
            else:
                formatted_symptoms = ", ".join(extracted_symptoms)
            
            true_symptoms = row['Symptoms']
            results.append({
                "Dialogue": dialogue,
                "True_Symptom": true_symptoms,
                "Extracted_Symptom": formatted_symptoms
            })

        df_results = pd.DataFrame(results)
        df_results.to_csv(f"dataset_extracting_{name}_{method}_with_RAG.csv", index=False)

        print(f"Saved dataset for method '{method}' to dataset_extracting_{name}_{method}.csv")
