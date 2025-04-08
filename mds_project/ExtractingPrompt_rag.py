class ExtractingPrompt_rag:
    """
    A class to generate prompts for extracting structured symptoms from patient dialogues.
    It supports various prompting strategies via the 'method' parameter and can incorporate
    additional context (e.g., retrieved similar dialogues with their true symptoms) into the prompt.
    """
    def __init__(self, symptom_list: list[str]) -> None:
        """
        Initialize the prompt generator with a predefined list of symptoms.
        
        Parameters:
            symptom_list (list[str]): The list of symptoms to extract from dialogues.
        """
        self.symptom_list = symptom_list

    def build_extraction_prompt(self, dialogue: str, method: str = "explicit", context: str = None) -> list[dict[str, str]]:
        """
        Build a prompt for symptom extraction according to the chosen method and optional context.
        
        Parameters:
            dialogue (str): The patient dialogue to analyze.
            method (str): The prompting strategy to use. Options include:
                - "explicit"
                - "zero_shot"
                - "few_shot"
                - "chain_of_thought"
                - "self_refinement"
                - "multiple_demonstrations"
            context (str): Optional additional context (e.g., similar dialogues with true symptoms).
        
        Returns:
            list[dict[str, str]]: The prompt messages (e.g., a system message and a user message).
        """
        # Prepare the symptom list string for use in the prompt.
        symptoms_str = ", ".join(self.symptom_list)
        
        # Define instructions based on the chosen method.
        if method == "explicit":
            instruction = (
                "You are a medical symptom extraction expert. Your task is to extract symptoms from patient dialogues "
                "strictly using the provided symptom list. Use EXACT matching, and assign confidence scores as follows: "
                "0.9-1.0 for explicit mentions, 0.7-0.8 for strong implications, and 0.5-0.6 for weak implications. "
                "Output ONLY a valid JSON object mapping each extracted symptom to its confidence score, with no additional text."
            )
        elif method == "zero_shot":
            instruction = (
                "You are a medical symptom extraction expert. Extract the symptoms mentioned in the dialogue based on the provided list, "
                "and output a valid JSON object mapping each symptom to a confidence score."
            )
        elif method == "few_shot":
            instruction = (
                "You are a medical symptom extraction expert. Here is an example:\n"
                "Dialogue: \"I have a pounding headache and blurry vision.\"\n"
                "Output: {\"Headache\": 0.95, \"Blurred vision\": 0.90}\n\n"
                "Now, extract the symptoms from the dialogue below using the provided symptom list. "
                "Output ONLY a valid JSON object."
            )
        elif method == "chain_of_thought":
            instruction = (
                "You are a medical symptom extraction expert. First, internally think through the dialogue and list potential symptom mentions along with confidence scores. "
                "Then, output ONLY the final valid JSON object mapping symptoms to their confidence scores. Do not show your internal reasoning."
            )
        elif method == "self_refinement":
            instruction = (
                "You are a medical symptom extraction expert. Extract symptoms from the dialogue using the provided symptom list. "
                "After extraction, verify that the output is a valid JSON object containing only symptoms from the list. "
                "Then, output ONLY the final valid JSON object mapping each symptom to its confidence score."
            )
        elif method == "multiple_demonstrations":
            instruction = (
                "You are a medical symptom extraction expert. Here are two examples:\n"
                "Example 1:\nDialogue: \"I feel a severe headache and constant fatigue.\"\nOutput: {\"Headache\": 0.95, \"Fatigue\": 0.90}\n\n"
                "Example 2:\nDialogue: \"My vision is blurry and I experience nausea occasionally.\"\nOutput: {\"Blurred vision\": 0.90, \"Nausea\": 0.80}\n\n"
                "Now, extract the symptoms from the dialogue below using the provided symptom list. "
                "Output ONLY a valid JSON object."
            )
        else:
            instruction = (
                "You are a medical symptom extraction expert. Extract symptoms from the dialogue using the provided list and output a valid JSON object mapping each symptom to its confidence score."
            )
        
        # If context is provided, prepend it to the dialogue section.
        if context:
            full_dialogue = "Context Examples:\n" + context + "\n\nTest Dialogue:\n" + dialogue.strip()
        else:
            full_dialogue = dialogue.strip()
        
        # Build the final prompt messages.
        messages = [
            {
                "role": "system",
                "content": instruction + f" The provided symptom list is: {symptoms_str}."
            },
            {
                "role": "user",
                "content": (
                    "DIALOGUE:\n\"\"\"\n" + full_dialogue + "\n\"\"\"\n\n"
                    "Extract symptoms as per the instructions above."
                )
            }
        ]
        return messages




