from typing import List

class PromptBuilder:
    def init(self) -> None:
        """
        Initialize the prompt generator.
        """
        pass


    def build_prompt(
            self,
            symptoms : list , 
            meta_str : str , 
            detail_level: float, 
            enumeration: bool = False, 
            explicit_symptom: bool = False, 
            language_style: str = "everyday language", 
            spelling_errors: bool = False, 
            tone: str = "neutral"

            ) -> list[dict]:
        
        """
        Construct a prompt based on the given parameters.
        """
        if not isinstance(symptoms, list) or not all(isinstance(s, str) for s in symptoms):
            raise ValueError("symptoms must be a list of strings.")
        

        messages = [
            {
                "role": "system",
                "content": f"You are a chatbot that writes in the voice of a patient using a {language_style} register with a {tone} tone."
            },
            {
                "role": "user",
                "content": (
                    f"A sick patient filled out a medical form . From the information he filled out , generate a phrase putting your shoes into this patient describing its symptoms. The symptoms the patient describes are: {', '.join(symptoms)}. "
                    f"The patient speaks in {language_style} Register and uses a {tone} tone. "
                )
            }
        ]
        
        
        messages[1]['content'] += (
                f"The informations he filled for these symptoms are thefollowing :  {meta_str} , the phrase should be generated based on these informations"
            )
        

        # Specify the level of detail
        detail_instructions = {
            1: "The description should be very brief with minimal details.",
            2: "The description should be brief with some basic details.",
            3: "The description should provide a moderate level of detail.",
            4: "The description should be detailed and thorough.",
            5: "The description should be very detailed and comprehensive."
        }
        messages[1]['content'] += detail_instructions.get(detail_level, "Provide a moderate level of detail.") + " "

        if enumeration:
            messages[1]['content'] += (
                "The patient should just enumerate the descriptions. For example, for symptom Anemia, a sentence could be: "
                "'Fatigue and Weakness, Pale Skin, Shortness of Breath, Dizziness, Irregular Heartbeat, Chest Pain, Cold Hands and Feet, Headaches'. "
            )

        if explicit_symptom:
            messages[1]['content'] += f"The patient should explicitly mention the symptoms they have (here {', '.join(symptoms)} should be all mentioned) "

        if spelling_errors:
            messages[1]['content'] += " Include spelling mistakes in the generated phrase."

        messages[1]['content'] += " Respond with only the sentence in the patient's voice, enclosed in double quotes, without any additional text, comments, or notes."

        return messages