import os

# Base paths
# Using absolute path for output file to ensure consistent deletion/writing regardless of CWD
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(PROJECT_DIR, "../memes_data/hateful_memes/")
JSONL_FILE = "dev_seen.jsonl"
IMG_DIR = "img/"
OFFLINE_FILE = os.path.join(PROJECT_DIR, "historical_tags.csv")

# Model Configs
POLICY_THRESHOLDS = {
    "Harmful_Content": 0.8,
    "Political_Content": 0.7,
    "Spam": 0.9,
    "Copyright_Infringement": 0.85
}

# Network Local VLM (Ollama)
OLLAMA_URL = "http://192.168.2.16:11434/api/generate"
OLLAMA_MODEL = "llama3.2-vision:11b"
OLLAMA_TEXT_MODEL = "llama3:8b"
PROMPT_FILE = "prompt.txt"
LLM_PROMPT_FILE = "llm_prompt.txt"
test = "dev_unseen.jsonl"