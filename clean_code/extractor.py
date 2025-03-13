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
        
        messages = [
            {
                "role": "system",                                                                             
                "content": (

                    "You are an AI assistant specialized in extracting medical symptoms. "
                    "Given a patient dialogue, identify which symptoms from the provided list are mentioned. "
                    "For each detected symptom, assign a confidence score between 0 and 1 indicating how likely it is present. "
                    "Return your response as a JSON object where keys are symptom names (only those detected) and values are the corresponding scores. "
                    "Only include a symptom if its score is above 0. "

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
        
        return messages