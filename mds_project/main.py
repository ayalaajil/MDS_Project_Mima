import os
import itertools
import random
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from LLM import LLM
from PromptBuilder import PromptBuilder
from metadata import language_registers, discussion_tones

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config")

def main(cfg: DictConfig) -> None:

    # Initialize prompt builder and LLM with values from config.

    prompt_builder = PromptBuilder()
    model = LLM(model_name=cfg.model.model_name, token=cfg.token.token_key)

    input_file = os.path.join(get_original_cwd(), cfg.data.input_file)
    
    a = pd.read_excel(input_file, sheet_name="PRO")

    index = (a[a['PRO-CTCAE PT'] == 'Pain and swelling at injection site']).index.values[0] + 1
    symptoms = a.iloc[:index]['PRO-CTCAE PT'].values

    # Build a nested dictionary with the symptoms, descriptions, and meta values.
    symptom_dict = {}
    for symptom in symptoms:
        try:
            symptom_dict[symptom] = {}
            descriptions = a[a['PRO-CTCAE PT'] == symptom]['Has PRO-CTCAE Attribute PT'].values[0].split(" || ")
            for description in descriptions:
                # Retrieve meta values for the description.
                meta_vals = a[a['PRO-CTCAE PT'] == description]['PRO-CTCAE Value PT'].values[0].split(" || ")
                symptom_dict[symptom][description] = meta_vals
        except Exception as e:
            print(f"Error processing {symptom}: {e}")


    # Calculate total iterations for progress tracking
    total_iterations = sum(
        len(list(itertools.product(symptom.keys(), *symptom.values())))
        for symptom in symptom_dict.values()
    )
    pbar = tqdm(total=total_iterations, desc="Generating dialogues")

    # Generate the dialogue data.
    data = []

    for symptom, descriptions in symptom_dict.items():

        # Generate all (description, meta) combinations
        description_meta_combinations = list(itertools.product(descriptions.keys(), *descriptions.values()))

        for description, *meta_combinations in description_meta_combinations:

            for meta_set in meta_combinations:
                
                detail_level = np.random.choice(cfg.generation.detail_levels)
                enumeration = np.random.choice([True, False], p=cfg.generation.enumeration_probability)
                explicit_symptom = np.random.choice([True, False], p=cfg.generation.explicit_symptom_probability)
                language_style = random.choice(language_registers)['name']
                tone = random.choice(discussion_tones)['name']
                spelling_errors = random.choice([True, False])


                prompt = prompt_builder.build_prompt(
                    symptoms=[symptom],
                    description=description,
                    meta=meta_set,
                    detail_level=detail_level,
                    enumeration=enumeration,
                    explicit_symptom=explicit_symptom,
                    language_style=language_style,
                    spelling_errors=spelling_errors,
                    tone=tone
                )

            
                phrase_generated = model.generate_text(messages=prompt)

                data.append([
                            phrase_generated, symptom, description, meta_set,
                            language_style, tone, detail_level, enumeration, explicit_symptom, spelling_errors
                            ])
                        
                # Update the progress bar for each generated sentence.
                pbar.update(1)

    pbar.close()

    # Create a DataFrame from the generated data.

    df = pd.DataFrame(
        data,
        columns=[
            "Dialogue_Generated", "symptom", "description", "meta",
            "language_style", "Tone", "Detail_level", "Enumeration",
            "Explicit_symptom", "Spelling_errors"
        ]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_filename, ext = os.path.splitext(cfg.data.output_file)
    output_filename = f"{base_filename}_{timestamp}{ext}"

    output_path = os.path.abspath(output_filename)

    df.to_csv(output_path, index=False)

    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    main()
