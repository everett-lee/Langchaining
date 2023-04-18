import pandas as pd
from langchain.agents import Tool, create_pandas_dataframe_agent

# from langchaining.agents.custom_agent import create_pandas_dataframe_agent_with_search, print_agent_prompt
from langchaining.agents.custom_agent import create_pandas_dataframe_agent_with_extra_tools, print_agent_prompt
from langchaining.agents.tools import is_prime, wikipedia
from langchaining.helpers import llm_gpt_turbo_0_temp, BASE_DIR, llm_gpt_davinci_0_temp

df = pd.read_csv(BASE_DIR / "agents/action.csv")
df = df.dropna()
# Give agent type information in df.dtypes section of input
df = df.astype(
    {
        "movie_id": str,
        "movie_name": str,
        "year": int,
        "certificate": str,
        "runtime": str,
        "genre": str,
        "rating": float,
        "description": str,
        "director": str,
        "star": str,
        "votes": int,
        "gross(in $)": int,
    }
)

additional_tools = [
    Tool(
        name="Wikipedia search",
        func=wikipedia.run,
        description="useful for when you want to find out additional information about a person",
    ),
    Tool(
        name="Prime checker",
        func=is_prime,
        description="useful for when you want to check if a number is prime",
    ),
]

# pandas_executor = create_pandas_dataframe_agent(llm=llm_gpt_davinci_0_temp, df=df, verbose=True)
agent_executor_chain = create_pandas_dataframe_agent_with_extra_tools(
    llm=llm_gpt_davinci_0_temp, df=df, tools=additional_tools, verbose=True
)

# print_agent_prompt(agent_executor_chain)

# query = "the lowest rated film with a gross of more than $100,000,000"
# query = "the highest grossing film with a below average rating"
query = "the highest grossing film that is not suitable for children"
command = f"""
The question begins below, use this to guide your actions:
First, inspect the dataframe df to find the name of the director that directed {query} present in the CSV file.
Second, use Wikipedia to find the year this director was born. Refer to this year as birth_year.
Third, find out if birth_year is a prime number.
Finally, output the director's name, year of birth, and whether birth_year is prime.
"""
agent_executor_chain.run(command)
