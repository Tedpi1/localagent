from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectory import retriever

model=OllamaLLM(
    model="llama3.2",
)

template="""
Your are an Expert in Answereing Computer related questions.
Here are some relevant reviews{reviews}
here is the question to answer{question}

"""

prompt=ChatPromptTemplate.from_template(template)

chain=prompt | model

result=chain.invoke({"reviews": [], "question": "What is the best desktop computer for building AI agents?"})
print(result)