from src.context.model import Context
from src.generator.lcel_generator import LCELGenerator
from src.llm.google_studio import GoogleStudioGenerator



LLM = GoogleStudioGenerator()
llm = LLM.get_llm()
generator = LCELGenerator(llm)

context = Context(
    text="""
Python is a SNAKE created by Parag.
""",
    sources=[],
)

response = generator.generate_response(
    query="Who created Python?",
    context=context,
)
print(response)

response = generator.generate_response(
    query="what is Python?",
    context=context,
)

print(response)