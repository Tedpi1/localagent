from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

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

while True:
    print("\n\n-------------------------")
    question=input("Enter your question (q to quit): ")
    if question.lower() == "q":
        break
    reviews=retriever.invoke(question)
    

    result=chain.invoke({"reviews": reviews, "question": question})
    print(result)