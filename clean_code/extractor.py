class ExtractingPrompt:
    """
    A class to generate prompts for extracting structured symptoms from patient dialogues.
    """
    def __init__(self, symptom_list: list[str]) -> None:
        """
        Initialize the prompt generator with a predefined list of symptoms.
        """
        self.symptom_list = symptom_list

    def build_extraction_prompt(self, dialogue: str) -> list[dict[str, str]]:

        """
        Builds a prompt instructing the LLM to extract symptoms in a structured JSON format.
        """
        # Join your list of 80 symptoms into a comma-separated string.
        symptoms_str = ", ".join(self.symptom_list)
        
        # messages = [
        #     {
        #         "role": "system",                                                                             
        #         "content": (

        #             "You are an AI assistant specialized in extracting medical symptoms. "
        #             "Given a patient dialogue, identify which symptoms from the provided list are mentioned. "
        #             "For each detected symptom, assign a confidence score between 0 and 1 indicating how likely it is present. "
        #             "Return your response as a JSON object where keys are symptom names (only those detected) and values are the corresponding scores. "
        #             "Only include a symptom if its score is above 0. "

        #             # "For example, for the following sentence: "
        #             # "I'm so scared, I've got these cracks at the corners of my mouth that won't go away, it's so painful and itchy, I'm so worried I'll get an infection. "
        #             # "You should output {'Cracking at the corners of the mouth (cheilosis/cheilitis)': 1}"
        #             # "An other example, for the following sentence: "
        #             # "I've been having a dry mouth for a while now."
        #             # "You should output {'Dry Mouth': 1}"
        #         ),
        #     },
        #     {
        #         "role": "user",
        #         "content": (
        #             f"Patient dialogue: \"{dialogue.strip()}\"\n\n"
        #             f"Symptom list: [{symptoms_str}]\n\n"
        #             "Extract the symptoms and output them as instructed in valid JSON format."
        #         ),
        #     },
        # ]



        messages = [
            {
                "role": "system",                                                                             
                "content": (
                "You are a clinical NLP expert specialized in symptom extraction from patient narratives. "
                "Analyze the dialogue to identify symptoms from the provided list, considering:\n"
                "1. Direct mentions and indirect references\n"
                "2. Symptom duration and severity qualifiers\n"
                "3. Patient's explicit concerns and implied symptoms\n"
                "4. Negations and speculative statements\n"
                "5. Related symptoms that may indicate underlying conditions\n\n"
                
                "For each identified symptom:\n"
                "- Assign confidence scores (0-1) using clinical judgment:\n"
                "  1.0 = Explicit, unambiguous mention\n"
                "  0.8-0.9 = Strong implication without direct naming\n"
                "  0.5-0.7 = Possible indication needing interpretation\n"
                "- Map colloquial terms to standardized medical terminology\n"
                "- Handle compound symptoms and comorbidities\n"
                "- Consider temporal aspects (current vs historical symptoms)\n\n"
                
                "Response format: Strict JSON {symptom: score} with ONLY detected symptoms.\n"

                    # "For example, for the following sentence: "
                    # "I'm so scared, I've got these cracks at the corners of my mouth that won't go away, it's so painful and itchy, I'm so worried I'll get an infection. "
                    # "You should output {'Cracking at the corners of the mouth (cheilosis/cheilitis)': 1}"
                    # "An other example, for the following sentence: "
                    # "I've been having a dry mouth for a while now."
                    # "You should output {'Dry Mouth': 1}"
                ),
            },
            
            {
                "role": "user",
                "content": (
                    f"Patient dialogue: \"{dialogue.strip()}\"\n\n"
                    f"Symptom list: [{symptoms_str}]\n\n"
                    "Extract the symptoms and output them as instructed in valid JSON format."
                ),
            },
        ]
        

        messages = [
    {
        "role": "system",
        "content": (
            "You are a clinical NLP expert specialized in symptom extraction from patient narratives. "
            "Analyze the dialogue to identify symptoms from the provided list, considering:\n"
            "1. Direct mentions and indirect references\n"
            "2. Symptom duration and severity qualifiers\n"
            "3. Patient's explicit concerns and implied symptoms\n"
            "4. Negations and speculative statements\n"
            "5. Related symptoms that may indicate underlying conditions\n\n"
            
            "For each identified symptom:\n"
            "- Assign confidence scores (0-1) using clinical judgment:\n"
            "  1.0 = Explicit, unambiguous mention\n"
            "  0.8-0.9 = Strong implication without direct naming\n"
            "  0.5-0.7 = Possible indication needing interpretation\n"
            "- Map colloquial terms to standardized medical terminology\n"
            "- Handle compound symptoms and comorbidities\n"
            "- Consider temporal aspects (current vs historical symptoms)\n\n"
            
            "Response format: Strict JSON {symptom: score} with ONLY detected symptoms.\n"
            "Examples:\n"
            "Dialogue: 'This cough won't quit - been hacking for weeks, sometimes see blood in the phlegm. So tired all the time.'\n"
            "Symptoms: ['hemoptysis', 'chronic cough', 'fatigue']\n"
            "Output: {'hemoptysis (coughing up blood)': 0.9, 'chronic cough': 0.85, 'fatigue': 0.7}\n\n"
            
            "Dialogue: 'My joints ache when it's cold, especially knees and fingers. No swelling though.'\n"
            "Symptoms: ['arthralgia', 'joint swelling']\n"
            "Output: {'arthralgia (joint pain)': 0.9}\n\n"
            
            "Dialogue: 'I'm constantly thirsty and peeing every hour - blood sugar was normal last checkup.'\n"
            "Symptoms: ['polyuria', 'polydipsia', 'hyperglycemia']\n"
            "Output: {'polyuria (excessive urination)': 0.95, 'polydipsia (excessive thirst)': 0.95}"
        )
    },
    {
        "role": "user",
        "content": (
            f"Patient narrative: \"{dialogue.strip()}\"\n\n"
            f"Standardized symptom list: [{symptoms_str}]\n\n"
            "Perform comprehensive symptom analysis:\n"
            "1. Identify explicit and implicit symptom references\n"
            "2. Differentiate between current complaints and historical reports\n"
            "3. Process negations and uncertainty markers\n"
            "4. Apply clinical terminology mapping\n"
            "5. Generate confidence scores with precision\n\n"
            "Output: Strict JSON (no markdown) following all guidelines."
        )
    }
    ]
        

        return messages