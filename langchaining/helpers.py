import os
from pathlib import Path

from dotenv import load_dotenv

from langchain import OpenAI

BASE_DIR = Path(__file__).parent.resolve()

load_dotenv()
gpt_key = os.environ["GPT_KEY"]

llm_gpt_turbo_0_temp = OpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=gpt_key
)


llm_gpt_turbo_high_temp = OpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=gpt_key
)

llm_gpt_davinci_0_temp = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    openai_api_key=gpt_key
)
