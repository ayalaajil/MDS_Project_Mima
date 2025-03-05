import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

class LLM:
    def __init__(self, 
                token,
                max_length: int = 300,
                model_name: str = "iRASC/BioLlama-Ko-8B"):
        
        """
        Initialize the LLM model and its tokenizer.

        :param max_length: (int) Maximum length of the generated text (default: 300).
        :param model_name: (str) Name of the model on Hugging Face Hub.
        """


        self.token = token

        if not self.token:
            raise ValueError("Authentication token is missing. Set it in the environment or config.json.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=self.token)

        self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            quantization_config=bnb_config, 
            use_auth_token=self.token
        )

        self.max_length = max_length

        self.generate_kwargs = {
                "do_sample": True,
                "temperature": 0.7,
                "max_new_tokens": 200,
                "eos_token_id": self.tokenizer.eos_token_id,
                "top_p": 0.9, 
                "repetition_penalty": 1.2  # Penalize repetitive phrases
            }

        self.text_generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_length,
        )

        print(f"Model loaded on device(s): {self.model.hf_device_map}")


    def generate_text(self, messages: list) -> str:
        """
        Generate text based on a list of messages (conversational format).

        :param messages: (list) A list of dictionaries containing conversation history.
                         Example: [{"role": "system", "content": "You are..."},
                                   {"role": "user", "content": "Hello!"}]
        :return: (str) Generated text response from the model.
        """

        outputs = self.text_generator(messages)

        # Extract the generated text
        generated_text = outputs[0]["generated_text"]
        return generated_text[-1]['content']