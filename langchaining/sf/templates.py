from langchain import PromptTemplate

template_scene_one = """
Continue the following scene with one character deciding to adopt a Cockney accent.
Output the text in script format, including the previous scene.
Previous scene: {scene}
"""

template_scene_two = """
Continue the previous scene by introducing Brian Blessed as himself, in typical bombastic style.
Output the text in script format, including the previous scene.
Previous scene: {previous_scene}
"""

prompt_scene_one = PromptTemplate(
    input_variables=["scene"], template=template_scene_one
)

prompt_scene_two = PromptTemplate(
    input_variables=["previous_scene"],
    template=template_scene_two,
)
