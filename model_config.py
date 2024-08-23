from abc import ABC


class ModelConfig(ABC):
    MODEL: str
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 4096
    TOP_P: float = 1.0
    FREQUENCY_PENALTY: float = 0.0
    PRESENCE_PENALTY: float = 0.0
    RESPONSE_TOKENS: int = 2000
    CONTEXT_WINDOW: int
    TOKENIZER: str = "cl100k_base"
    MODEL_PATH: str = ""
    REPO_ID: str = ""
    FILENAME: str = ""


class Gpt35Config(ModelConfig):
    MODEL = 'gpt-3.5-turbo'
    CONTEXT_WINDOW = 4096


class Gpt4Config(Gpt35Config):
    MODEL = 'gpt-4'
    MAX_TOKENS = 5000
    CONTEXT_WINDOW = 8000


class LlamaConfig(ModelConfig):
    MODEL = 'llama'
    MODEL_PATH = '/home/tomas/.models/Meta-Llama-3-8B-Instruct.Q4_0.gguf'
    CONTEXT_WINDOW = 4096
    RESPONSE_TOKENS = 2500


class Gpt4AllConfig(ModelConfig):
    MODEL = 'gpt4all-falcon'
    MODEL_PATH = '/home/tomas/.models/gpt4all-falcon-newbpe-q4_0.gguf'
    CONTEXT_WINDOW = 512