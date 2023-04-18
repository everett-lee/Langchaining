from langchain import LLMChain
from langchain.chains import SimpleSequentialChain

from langchaining.helpers import llm_gpt_turbo_high_temp
from langchaining.sf.vectorstore import get_db
from langchaining.sf.templates import prompt_scene_one, prompt_scene_two

LINE_BREAK = "=" * 25

db = get_db()
query = "George quits his job"
# similarity search
retriever = db.as_retriever(search_kwargs={"k": 1})
docs = retriever.get_relevant_documents(query)
closest = docs[0]

print()
print(f"Episode: {closest.metadata['episode_title']}")
print(f"{LINE_BREAK}  STARTING SCENE  {LINE_BREAK}")
print(closest.page_content)
print(LINE_BREAK * 3)

chain_one = LLMChain(llm=llm_gpt_turbo_high_temp, prompt=prompt_scene_one)
chain_two = LLMChain(llm=llm_gpt_turbo_high_temp, prompt=prompt_scene_two)

overall_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
overall_chain.run(closest)
