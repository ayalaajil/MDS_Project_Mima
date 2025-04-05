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
        
        symptoms_str = ", ".join(self.symptom_list)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical symptom extraction expert. Your task is to analyze patient dialogues "

                    "You are an AI assistant specialized in extracting medical symptoms. "
                    "Given a patient dialogue, identify which symptoms from the provided list are mentioned with high precision using these rules:\n"
                    " STRICT MATCHING: Only output symptoms that are mentioned in the symptoms list {symptoms_str}.\n"
                    " CONTEXT AWARENESS: Consider negation (e.g., 'no fever') and temporal aspects (e.g., 'had headache yesterday')\n"
                    " CONFIDENCE SCORING: Assign scores using this scale:\n"
                    "   - 0.9-1.0: Explicitly mentioned (e.g., 'I have fever')\n"
                    "   - 0.7-0.8: Strongly implied (e.g., 'my head is pounding' → headache)\n"
                    "   - 0.5-0.6: Weakly implied (only include if no better matches exist)\n"
                    "4. OUTPUT FORMAT: Strict JSON with {symptom: score} pairs, no additional text\n\n"
                    "Example Output:\n"
                    "{\"fever\": 0.95, \"headache\": 0.8}\n\n"
                    "Available symptoms: " + symptoms_str
                )
            },
            {
                "role": "user",
                "content": (
                    "DIALOGUE:\n\"\"\"\n" + dialogue.strip() + "\n\"\"\"\n\n"
                    "Analyze this dialogue and extract symptoms according to the rules above. "
                    "Output ONLY valid JSON with no additional commentary."
                )
            }
        ]
        return messages





# class ExtractingPrompt:
#     """
#     A class to generate prompts for extracting structured symptoms from patient dialogues.
#     """
#     def __init__(self, symptom_list: list[str]) -> None:
#         """
#         Initialize the prompt generator with a predefined list of symptoms.
#         """
#         self.symptom_list = symptom_list

#     def build_extraction_prompt(self, dialogue: str) -> list[dict[str, str]]:
        
#         symptoms_str = ", ".join(self.symptom_list)
        
#         messages = [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a medical symptom extraction expert. Your task is to analyze patient dialogues "
#                     "and identify symptoms with high precision using these rules:\n"
#                     "1. STRICT MATCHING: Only output symptoms that are explicitly mentioned or very clearly implied\n"
#                     "2. CONTEXT AWARENESS: Consider negation (e.g., 'no fever') and temporal aspects (e.g., 'had headache yesterday')\n"
#                     "3. CONFIDENCE SCORING: Assign scores using this scale:\n"
#                     "   - 0.9-1.0: Explicitly mentioned (e.g., 'I have fever')\n"
#                     "   - 0.7-0.8: Strongly implied (e.g., 'my head is pounding' → headache)\n"
#                     "   - 0.5-0.6: Weakly implied (only include if no better matches exist)\n"
#                     "4. OUTPUT FORMAT: Strict JSON with {symptom: score} pairs, no additional text\n\n"
#                     "Example Output:\n"
#                     "{\"fever\": 0.95, \"headache\": 0.8}\n\n"
#                     "Available symptoms: " + symptoms_str
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     "DIALOGUE:\n\"\"\"\n" + dialogue.strip() + "\n\"\"\"\n\n"
#                     "Analyze this dialogue and extract symptoms according to the rules above. "
#                     "Be conservative - it's better to miss a symptom than include an incorrect one. "
#                     "Output ONLY valid JSON with no additional commentary."
#                 )
#             }
#         ]
#         return messages

#     # def build_extraction_prompt(self, dialogue: str) -> list[dict[str, str]]:

#     #     """
#     #     Builds a prompt instructing the LLM to extract symptoms in a structured JSON format.
#     #     """
#     #     # Join your list of 80 symptoms into a comma-separated string.
#     #     symptoms_str = ", ".join(self.symptom_list)
        
#     #     messages = [
#     #         {
#     #             "role": "system",                                                                             
#     #             "content": (

#     #                 "You are an AI assistant specialized in extracting medical symptoms. "
#     #                 "Given a patient dialogue, identify which symptoms from the provided list are mentioned. "
#     #                 "For each detected symptom, assign a confidence score between 0 and 1 indicating how likely it is present. "
#     #                 "Return your response as a JSON object where keys are symptom names (only those detected) and values are the corresponding scores. "
#     #                 "Only include a symptom if its score is above 0. "
#     #             ),
#     #         },
#     #         {
#     #             "role": "user",
#     #             "content": (
#     #                 f"Patient dialogue: \"{dialogue.strip()}\"\n\n"
#     #                 f"Symptom list: [{symptoms_str}]\n\n"
#     #                 "Extract the symptoms and output them as instructed in valid JSON format."
#     #             ),
#     #         },
#     #     ]

#     #     return messages