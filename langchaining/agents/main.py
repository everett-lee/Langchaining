import pandas as pd
from langchain import WikipediaAPIWrapper
from langchain.agents import Tool, create_pandas_dataframe_agent

from langchaining.agents.custom_agent import create_pandas_dataframe_agent_with_search
from langchaining.agents.tools import is_prime
from langchaining.helpers import llm_gpt_turbo_0_temp, BASE_DIR, llm_gpt_davinci_0_temp

df = pd.read_csv(BASE_DIR / "agents/action.csv")
wikipedia = WikipediaAPIWrapper()

additional_tools = [
    Tool(
        name="Wikipedia search",
        func=wikipedia.run,
        description="useful for when you want to find out additional information about a person"
    ),
    Tool(
        name="Prime checker",
        func=is_prime,
        description="useful for when you want to check if a number is prime"
    )
]

# pandas_executor = create_pandas_dataframe_agent(llm=llm_gpt_turbo_0_temp, df=df, verbose=True)
executor = create_pandas_dataframe_agent_with_search(
    llm=llm_gpt_davinci_0_temp,
    df=df,
    tools=additional_tools,
    verbose=True
)

query = "the lowest rated film with a gross of more than $100,000,000"
command = f"""
The question begins below, use this to guide your actions:
First, list all the column names and inspect a value for each in a single row so that you are familiar with the content.
Second, inspect the dataframe df to find the name of the director that directed {query} present in the CSV file.
Third, use Wikipedia to find the year this director was born. Refer to this year as birth_year.
Fourth, find out if birth_year is a prime number.
Finally output the director's name, year of birth, and whether birth_year is prime.
"""
executor.run(command)
