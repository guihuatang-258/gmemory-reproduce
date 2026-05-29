import os
from dotenv import load_dotenv
load_dotenv()

for key in (
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "HF_ENDPOINT",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_BASE",
    "EMBEDDING_API_KEY",
):
    value = os.getenv(key)
    if value is not None:
        os.environ[key] = value
