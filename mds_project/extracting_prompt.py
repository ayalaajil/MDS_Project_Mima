class ExtractingPrompt:
    """
    A class to generate prompts for extracting structured symptoms from patient dialogues,
    supporting different prompting strategies.
    """
    def __init__(self, symptom_list: list[str]) -> None:
        """
        Initialize the prompt generator with a predefined list of symptoms and a chosen method.

        Args:
            symptom_list (list[str]): The list of symptoms.
            method (str): The prompting strategy to use. Options include:
                - "explicit": Use clear, explicit instructions and structured output format.
                - "zero_shot": Use a zero-shot style (instruction-only, without examples).
                - "few_shot": Provide one or two few-shot examples.
                - "chain_of_thought": Ask the model to internally reason before outputting the final JSON.
                - "self_refinement": Instruct the model to verify its own output format before finalizing.
                - "multiple_demonstrations": Provide several demonstration examples to ensemble reasoning.
        """
        self.symptom_list = symptom_list
        

    def build_extraction_prompt(self, dialogue: str , method: str = "explicit") -> list[dict[str, str]]:
        """
        Build the extraction prompt based on the chosen method.

        Args:
            dialogue (str): The patient dialogue from which to extract symptoms.
        
        Returns:
            list[dict[str, str]]: The prompt messages in conversational format.
        """
        
        self.method = method
        
        symptoms_str = ", ".join(self.symptom_list)
        
        # Common instruction: tell the model what to extract.
        base_instruction = (
            "You are an AI assistant specialized in extracting medical symptoms. "
            "Given a patient dialogue, identify which symptoms from the provided list are mentioned. "
            "Use STRICT MATCHING: Only output symptoms that are mentioned in the symptoms list {symptoms_str}.\n"
            "Consider negations (e.g., 'no headache') so that if a symptom is negated, do not extract it. "
        )
        
        # Define different strategies:
        if self.method == "explicit":
            # Explicit instruction and structured output only.
            task_instruction = (
                base_instruction +
                "Assign a confidence score between 0 and 1 for each symptom: "
                "eg 0.95-1.0 for explicit mentions, 0.7-0.9 for strong implications, 0.3-0.6 for weak implications. "
                "Output a valid JSON object with symptom: score pairs, and no extra text."
            )
        
        elif self.method == "zero_shot":
            # Zero-shot: minimal examples, just clear instructions.
            task_instruction = (
                base_instruction +
                "Output ONLY a JSON object mapping symptoms to their confidence scores. "
                "For example, if the dialogue mentions 'Nausea' explicitly, you might output {\"Nausea\": x} with x very close to 1 ."
                "Other Example Output:\n"
                    "{\"Fecal incontinence\": 0.95, \"Cracking at the corners of the mouth (cheilosis/cheilitis)\": 0.1}\n\n"
            )
        
        elif self.method == "few_shot":
            # Few-shot: include one example demonstration.
            task_instruction = (
                    base_instruction +
                    "Below are three examples:\n\n"
                    "Dialogue: \"I have a terrible headache and my vision is blurred.\"\n"
                    "Output: {\"Headache\": 0.95, \"Blurred vision\": 0.95}\n\n"
                    "Now, analyze the following dialogue and output ONLY a valid JSON object."
                )


        
        elif self.method == "chain_of_thought":
            # Chain-of-thought: instruct the model to internally reason then output the final JSON.
            task_instruction = (
                base_instruction +
                "Before providing your final answer, internally list the symptoms mentioned and assign a preliminary score to each. "
                "Then, output ONLY the final JSON object with the symptom: score pairs."
            )
        
        elif self.method == "self_refinement":
            # Self-refinement: instruct the model to verify its output format.
            task_instruction = (
                base_instruction +
                "After processing the dialogue, verify that all extracted symptoms are from the provided list and that the JSON format is correct. "
                "Then output ONLY the valid JSON object with symptom: score pairs."
            )
        
        elif self.method == "multiple_demonstrations":
            # Multiple demonstrations: provide several examples.
            task_instruction = (
                    base_instruction +
                    "Below are three examples:\n\n"
                    "Dialogue: \"I have a terrible headache and my vision is blurred.\"\n"
                    "Output: {\"Headache\": 0.95, \"Blurred vision\": 0.95}\n\n"
                    "Dialogue: \"Alas, my pitiful bladder doth betray me, releasin' its waters with a frequency most unbecomin', "
                    "a rare occurence indeed, yet causin' me such distress, it doth interfere with my daily pursuits, "
                    "leavin' me in a state of utter discomfort, a true calamity, if I do say so myself.\"\n"
                    "Output: {\"Urinary incontinence\": 0.95}\n\n"
                    "Dialogue: \"Malaise, Disorientation, Confusion, Headaches, Fatigue, Difficulty Concentrating, Inability to Focus, "
                    "Irritability, Anxiety, Restlessness, Sleep Disturbances, Memory Loss, Forgetfulness, Decreased Productivity, "
                    "Discomfort, Unsettled Feeling, Mild Nausea, Abdominal Discomfort, Mild Diarrhea, Loss of Interest in Activities, "
                    "Difficulty Engaging in Daily Routine, Mild Depression, Mild Anxiety, Mild Panic Attacks, Mild Insomnia, Mild Hypersomnia.\"\n"
                    "Output: {"
                        "\"Rash\": 0.95, \"Decreased appetite\": 0.99, \"Breast swelling and tenderness\": 0.93}\n\n"
                    "Now, analyze the following dialogue and output ONLY a valid JSON object."
                )
            
        else:
            # Fallback to explicit instructions if unknown method.
            task_instruction = (
                base_instruction +
                "Output a valid JSON object with symptom: score pairs, and no additional text."
            )
        
        # Build the full prompt messages.
        messages = [
            {
                "role": "system",
                "content": task_instruction.replace("{symptoms_str}", symptoms_str)
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