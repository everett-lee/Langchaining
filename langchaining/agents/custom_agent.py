from typing import Sequence
import pandas as pd

from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.tools import BaseTool
from langchain.tools.python.tool import PythonAstREPLTool

PREFIX = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
Answer the following questions as best you can. You have access to the following tools:"""

SUFFIX = """
This is the result of `print(df.head())`:
{df_head}
This is the result of `print(df.dtypes)`:
{df_types}

The question you must answer is provided below. Begin!
Question: {input}
{agent_scratchpad}"""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""


def create_pandas_dataframe_agent_with_extra_tools(
        llm: BaseLLM,
        tools: Sequence[BaseTool],
        df: pd.DataFrame,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        verbose: bool = True
) -> AgentExecutor:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas object, got {type(df)}")

    # Adding custom tools to list of tools
    tools = list(tools) + [PythonAstREPLTool(locals={"df": df})]

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        format_instructions=format_instructions,
        input_variables=["input", "df_head", "df_types", "agent_scratchpad"]
    )

    # Partially complete prompt with additional info
    partial_prompt = prompt.partial(df_head=str(df.head()), df_types=str(df.dtypes))

    llm_chain = LLMChain(
        llm=llm,
        prompt=partial_prompt,
    )

    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        tools=tools,
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)


def print_agent_prompt(agent_executor: AgentExecutor) -> None:
    print("*" * 100)
    print("\n")
    print(agent_executor.agent.llm_chain.prompt.template)
    print("\n\n")
    print("*" * 100)
