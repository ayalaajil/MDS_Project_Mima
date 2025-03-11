class ExtractingPrompt:
    """
    A class to generate prompts for extracting structured symptoms from patient dialogues
    based on the PRO-CTCAE dataset.
    """

    def __init__(self) -> None:
        """
        Initialize the prompt generator. This can be extended with additional attributes
        if needed in the future.
        """
        pass

    def build_extraction_prompt(self, dialogue: str) -> list[dict[str, str]]:
        """
        Builds a structured prompt instructing the LLM to extract symptoms from the provided dialogue.

        :param dialogue: The dialogue text from which to extract symptoms.
        :return: A structured list of messages in a format suitable for LLM-based chat models.
        """
        dialogue = dialogue.strip()  # Ensures clean input

        # ajouter les symptomes PRO CTCAE.......

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant trained to extract structured symptoms from patient inputs "
                    "based on the PRO-CTCAE dataset. You must identify and return only the symptoms mentioned, "
                    "ensuring accuracy and completeness."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Identify and extract the symptoms described in the following patient statement:\n\n"
                    f"\"{dialogue}\"\n\n"
                    "Return only the symptoms, separated by commas if multiple, "
                    "without any additional text, comments, or explanations."
                ),
            },
        ]

        return messages



#  class Extracting_prompt:
#     def __init__(self) -> None:
#         """
#         Initialize the prompt generator.
#         """
#         pass

#     def build_extraction_prompt(self , dialogue: str) -> str:

#         """
#         Build a prompt that instructs the LLM to extract specific information
#         from the provided dialogue.

#         :param dialogue: (str) The dialogue text from which to extract details.
#         :return: (str) A prompt message with extraction instructions.
#         """
#         messages = [
#             {
#                 "role": "system",
#                 "content": f"You are a chatbot and you act as an assistant that extracts structured symptoms from patient inputs, based on the PRO-CTCAE dataset."
#             },
#             {
#                 "role": "user", 
#             }
#         ]
        
#         messages[1]['content'] = (

#             "Find the symptoms described in the phrase : "

#             f"{dialogue}\n"

#             "Respond with only the extracted symptoms, enclosed in double quotes, without any additional text, comments, or notes."
#         )

#         return messages











# "2. Find severity among this list:\n"
# "   [Very severe, Not sexually active, Prefer not to answer, None, Not applicable, Moderate, Mild, Severe]\n\n"

# "3. Find frequency among this list:\n"
# "   [Almost constantly, Not sexually active, Prefer not to answer, Occasionally, Frequently, Never, Rarely]\n\n"

# "4. Determine language_style from this list:\n"
# "   ['Neutral/Standard Register', 'Informal Register', 'Formal Register', 'Poetic/Literary Register', 'Vulgar Register']\n\n"

# "5. Determine Tone from this list:\n"
# "   ['Confused', 'Neutral', 'Angry', 'Fearful', 'Friendly', 'Insulting']\n\n"

# "6. Determine Detail_level on a scale from 1 to 5, where:\n"
# "   1: a description very brief with minimal details.\n"
# "   2: a description brief with some basic details.\n"
# "   3: a description with a moderate level of detail.\n"
# "   4: a description that is detailed and thorough.\n"
# "   5: a description that is very detailed and comprehensive.\n\n"


# "Please analyze the following dialogue and extract the following details:\n\n"