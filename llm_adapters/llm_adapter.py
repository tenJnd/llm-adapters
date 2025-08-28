import json
import os
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import openai
import tiktoken
from huggingface_hub import snapshot_download
from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer

import llm_adapters.model_config as model_config


class ParsingError(Exception):
    """Exception raised for errors in the parsing process."""


class BaseLLMClient(ABC):
    def __init__(self, config: model_config.ModelConfig):
        self.config = config
        self.tokenizer = tiktoken.get_encoding(self.config.TOKENIZER)

    @abstractmethod
    def call_agent(self, user_prompt: str, system_prompt: str) -> Dict:
        pass

    @abstractmethod
    def call_with_functions(self, user_prompt: str, system_prompt: str, functions: List[Dict],
                            function_call: str = "auto") -> Dict:
        pass

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate the number of tokens used by the given messages.

        Parameters:
        - messages: List of message dicts for the conversation.

        Returns:
        - The total number of tokens used by all the messages.
        """
        total_tokens = 0

        for message in messages:
            # Calculate the number of tokens in the message content
            total_tokens += len(self.tokenizer.encode(message['content']))

            # Each message has some overhead for role tokens (e.g., "system", "user")
            total_tokens += 2  # Estimate for the role tokens and separator tokens

        return total_tokens

    def calculate_max_tokens(self, messages):
        # Use the count_tokens method to estimate the token usage
        total_tokens = self.count_tokens(messages)

        # Calculate how many tokens we have left for the completion
        max_available_tokens = self.config.CONTEXT_WINDOW - total_tokens

        # Cap the max_tokens to ensure the total does not exceed the context window
        max_tokens = min(self.config.MAX_TOKENS, max_available_tokens)

        if max_tokens <= 0:
            return {"error": "The messages are too long to fit within the context window."}
        return max_tokens

    @staticmethod
    def parse_json_output(response):
        try:
            if isinstance(response, dict):
                return json.loads(response['choices'][0]['message']['content'])
            return json.loads(response.dict()['choices'][0]['message']['content'])
        except (KeyError, TypeError) as exc:
            raise ParsingError(f"Cannot parse the agent response: {traceback.format_exc()}\n"
                               f"response: {response}") from exc
        except json.JSONDecodeError as exc:
            raise ParsingError(f"Cannot decode the agent response: {traceback.format_exc()}\n"
                               f"response: {response}") from exc


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: model_config.ModelConfig):
        super().__init__(config)
        self.client = openai

    # def count_tokens(self, messages: List[Dict[str, str]]) -> int:
    #     """
    #     Estimate the number of tokens used by the given messages using OpenAI's token counting.
    #
    #     Parameters:
    #     - messages: List of message dicts for the conversation.
    #
    #     Returns:
    #     - The estimated number of tokens used.
    #     """
    #     try:
    #         # Call the OpenAI API with max_tokens=1 to estimate the token usage
    #         total_tokens = self.client.chat.completions.create(
    #             model=self.config.MODEL,
    #             messages=messages,
    #             max_tokens=1  # We don't need an actual completion, just the token usage
    #         )['usage']['total_tokens']
    #         return total_tokens
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to count tokens: {str(e)}")

    def call_agent(self, user_prompt: str, system_prompt: str) -> Dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        max_tokens = self.calculate_max_tokens(messages)

        # Call the OpenAI API with the adjusted max_tokens
        return self._generic_openai_call(messages, max_tokens=max_tokens)

    def _generic_openai_call(self, messages: List[Dict[str, str]], functions: Optional[List[Dict]] = None,
                             function_call: Optional[str] = "auto", **kwargs) -> Dict:
        """
        A generic method to interact with OpenAI's API, supporting function calls.

        Parameters:
        - messages: List of message dicts for the conversation.
        - functions: A list of functions that the model can call.
        - function_call: How the model should call functions, either 'auto', or a specific function name.
        - kwargs: Additional parameters to customize the OpenAI API call.

        Returns:
        - The API response as a dictionary.
        """
        try:
            # Build the payload for the API request
            if str.startswith(self.config.MODEL, 'gpt-5'):
                payload = {
                    "model": self.config.MODEL,
                    "messages": messages,
                }
            else:
                payload = {
                    "model": self.config.MODEL,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.config.TEMPERATURE),
                    "max_tokens": kwargs.get("max_tokens", self.config.MAX_TOKENS),
                    "top_p": kwargs.get("top_p", self.config.TOP_P),
                    "frequency_penalty": kwargs.get("frequency_penalty", self.config.FREQUENCY_PENALTY),
                    "presence_penalty": kwargs.get("presence_penalty", self.config.PRESENCE_PENALTY),
                    "stop": kwargs.get("stop", ["```"])
                }

            # Only include functions and function_call if functions are provided
            if functions:
                payload["functions"] = functions
                payload["function_call"] = function_call

            response = self.client.chat.completions.create(**payload)
            return response

        except Exception as e:
            return {"error": str(e)}

    def call_with_functions(self, user_prompt: str, system_prompt: str, functions: List[Dict],
                            function_call: str = "auto") -> Dict:
        """
        Method to call OpenAI's API with functions.

        Parameters:
        - user_prompt: The user prompt to send to the model.
        - system_prompt: The system prompt to provide context to the model.
        - functions: A list of function definitions that the model can call.
        - function_call: Specifies whether to auto-call functions or a specific function.

        Returns:
        - The API response as a dictionary.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self._generic_openai_call(messages, functions=functions, function_call=function_call)


class BaseLocalClient(BaseLLMClient):
    def __init__(self, config: model_config.ModelConfig, download_model: bool = False):
        super().__init__(config)
        try:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ModuleNotFoundError:
            self.device = None

        if not os.path.exists(config.MODEL_PATH) and download_model:
            self.download_model(config.REPO_ID, config.FILENAME, config.MODEL_PATH)

        if config.MODEL_PATH.endswith(".gguf"):
            self.client = Llama(model_path=config.MODEL_PATH,
                                n_ctx=config.CONTEXT_WINDOW,
                                n_batch=config.CONTEXT_WINDOW,
                                device=self.device)  # Ensure this is supported by Llama
        else:
            self.client, self.tokenizer = self.load_transformers_model(config.MODEL_PATH)

    @staticmethod
    def download_model(repo_id: str, filename: str, model_path: str):
        if repo_id and filename:
            print(f"Downloading model {repo_id} to {model_path}...")
            snapshot_download(repo_id=repo_id, allow_patterns=filename, local_dir=model_path)
            print(f"Model downloaded to {model_path}.")
        else:
            raise ValueError("Model path or repository ID not provided for downloading the model.")

    @classmethod
    def load_transformers_model(cls, model_path: str):
        print(f"Loading transformers model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    def call_agent(self, user_prompt: str, system_prompt: str) -> Dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        max_tokens = self.calculate_max_tokens(messages)
        if self.config.MODEL_PATH.endswith(".gguf"):
            try:
                response = self.client.create_chat_completion(
                    messages=messages,
                    temperature=self.config.TEMPERATURE,
                    max_tokens=max_tokens,
                    top_p=self.config.TOP_P,
                    frequency_penalty=self.config.FREQUENCY_PENALTY,
                    presence_penalty=self.config.PRESENCE_PENALTY,
                    stop=["```"]
                )
            except Exception as e:
                response = {"error": str(e)}
        else:
            input_ids = self.tokenizer.encode(system_prompt + user_prompt, return_tensors='pt')
            output = self.client.generate(
                input_ids,
                max_length=max_tokens,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P
            )
            response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = {"choices": [{"message": {"content": response_text}}]}
        return response

    def call_with_functions(self, user_prompt: str, system_prompt: str, functions: List[Dict],
                            function_call: str = "auto") -> Dict:
        raise NotImplementedError("Function calling is not supported for local models.")


class LLMClientFactory:
    @staticmethod
    def create_llm_client(model_conf: model_config.ModelConfig) -> BaseLLMClient:
        if model_conf.MODEL in ['llama', 'gpt4all-falcon']:
            return BaseLocalClient(model_conf)
        else:
            return OpenAIClient(model_conf)


if __name__ == '__main__':
    user_prompt_test = "Please tell me a joke."
    system_prompt_test = "You are a helpful assistant."

    # Example usage for Llama model
    llm_client = LLMClientFactory.create_llm_client(model_config.LlamaConfig)
    final_response = llm_client.call_agent(user_prompt_test, system_prompt_test)
    print(final_response)

    # Example usage with function calls for OpenAI API
    llm_client = LLMClientFactory.create_llm_client(model_config.Gpt35Config)
    funcs = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    ]
    final_response = llm_client.call_with_functions(user_prompt_test, system_prompt_test, funcs)
    print(final_response)
