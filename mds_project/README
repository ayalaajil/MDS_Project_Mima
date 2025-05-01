
## Dataset Generation Scripts

These scripts are for generating synthetic datasets of symptom phrases. Each script uses different strategies for simulating the presence of symptoms and their attributes.

### `generate_one_symptom.py`
Generates a synthetic dataset where each phrase contains **exactly one symptom**.  
- Symptom attributes such as *frequency*, *severity*, and *interference* are extracted from the PRO-CTCAE dataset.  
- All valid attribute combinations for each symptom are constructed.

### `generate_many_symptoms_Poisson_distribution.py`
Generates phrases containing **multiple symptoms**, with the number of symptoms per phrase sampled from a **Poisson distribution** with λ = 1.5.  
- Each sample contains a variable number of symptoms drawn from this distribution.

### `generate_many_symptoms_Predefined_distribution.py`
Generates phrases with **multiple symptoms**, where the number of symptoms per phrase is drawn from a **custom predefined distribution**:  
- P(k = 1) = 0.3  
- P(k = 2) = 0.4  
- P(k = 3) = 0.2  
- P(k = 4) = 0.1

### `generate_many_symptoms_Correlation_Matrix.py`
Generates phrases using a **Poisson distribution (λ = 1.5)** for symptom count, while incorporating a **learned symptom co-occurrence matrix**.  
- This ensures that symptom combinations reflect realistic clinical correlations.

---

### Customization

All scripts are modular and support parameter modification to fit different generation setups. Users can:
- Adjust the number of samples, distribution parameters (e.g., λ), or symptom pool.
- Modify or extend the co-occurrence matrix or attribute combinations to simulate various clinical scenarios.

---

### Example Usage

To run the generation pipeline, you can run the generate_... files in terminal, after running the synthetic datasets will be generated.

## Symptom Extraction Pipeline

This section describes the pipeline used to extract symptoms from synthetic dialogue datasets using the Biollama model.

### Overview

run_extractor.py : The script loads multiple synthetic datasets and applies an LLM-based prompt extraction method to identify symptoms mentioned in each phrase. It compares the extracted symptoms with ground truth labels and outputs structured results.

### Components

- `LLM` from the LLM.py file : A wrapper for the language model used to generate predictions. Example model: `iRASC/BioLlama-Ko-8B`.
- `ExtractingPrompt Class` from the extracting_prompt.py file : A class for constructing prompts tailored to symptom extraction from clinical dialogues.
- `extract_symptoms_from_dataset(...)`: Main function that applies the LLM to extract symptom names and confidence scores from each phrase.

### Datasets Processed

The following generated datasets are processed for symptom extraction:
- `dataset_generated_one_symptom_per_phrase.csv`
- `dataset_generated_multiple_symptoms_per_phrase_predefined_distrib.csv`
- `dataset_generated_multiple_symptoms_per_phrase_poisson_distrib.csv`
- `dataset_generated_multiple_symptom_per_phrase_poisson_correlations.csv`

Each output is saved to a corresponding CSV file:
- `extracted_onesymptom.csv`
- `extracted_multi_predef.csv`
- `extracted_multi_poisson.csv`
- `extracted_multi_poisson_correl.csv`

### Symptom Filtering

Extracted symptoms are included in the final output if their predicted confidence score exceeds **0.60**. The raw LLM output and parsed confidence scores are retained for reference.

### Customization

You can modify:
- The LLM used (`LLM(model_name=...)`)
- The symptom list (loaded from the `PRO-CTCAE` terminology)
- The filtering threshold
- Prompt formatting logic in `ExtractingPrompt`

---

### Example Usage

To run the extraction pipeline, you can run the run.extractor.py file in terminal. Then, the extracted_.... files will be generated and saved in csv files. 


## Evaluation Pipeline

- `MultiLabelEvaluator`: Evaluates predictions using class-based metrics.
- `SymptomMultiLabelEvaluator`: Extends the evaluator to work with sets of symptoms using both class-based and sample-based evaluations.

### Metrics Computed :

#### Class-Based Metrics (per symptom):
- **Accuracy**
- **Precision**
- **Recall**
- **Hamming Loss**
- **Micro / Macro / Weighted F1-score**
- **Jaccard Index (Macro)**
- **Subset Accuracy**
- **Mean Average Precision (MAP)**

#### Sample-Based Metrics (per dialogue):
- **Precision**
- **Recall**
- **F1-score**
- **Jaccard index**

#### Additional Diagnostics:
- **False Positives / Negatives per sample**
- **Work Saved Score**: Indicates how many corrections are avoided by automatic extraction.

### How it works

1. **Prediction and Ground Truth Formatting**:
   - The true and predicted symptom sets are first binarized based on a fixed symptom universe.

2. **Metric Calculation**:
   - For each class and sample, relevant metrics are computed.
   - Softmax/softmin aggregations are used to capture performance extremes (controlled by the `alpha` parameter).

### Batch Evaluation on Prompting Methods and Datasets

We provide an automated loop to **evaluate and compare** multiple prompting strategies across different dataset configurations using our evaluation pipeline.

### Prompting Strategies Evaluated

The following prompting methods are tested:

- `explicit`
- `zero_shot`
- `few_shot`
- `chain_of_thought`
- `self_refinement`
- `multiple_demonstrations`

*RAG variants are included in the code but commented out.*

### Datasets Handled

The script evaluation_global.py is compatible with the following dataset formats:
- `dataset_extracting_multi_poisson_correl`
- `dataset_extracting_multi_poisson`
- `dataset_extracting_multi_predef`
- `dataset_extracting_one_symptom`

For each dataset, the script produces a .csv file like : 

comparison_prompting_methods_dataset_extracting_multi_poisson.csv

Each file compares all prompting methods side-by-side across multiple evaluation metrics.


### Remark :

RAG variants can be easily tested by uncommenting them in the prompting_methods list.

We encountered some difficulties with the RAG method—further algorithmic refinement or modifications may be required. We leave this aspect as a direction for future work. However, most components of our implementation are modular and functioning well.

Thank you for reading !

