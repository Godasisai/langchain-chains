from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

model2 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text.\n\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text.\n\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="""
Merge the provided notes and quiz into a single document.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser,
    }
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, choosing the correct Kernel and regularization is crucial.

SVMs do not directly provide probability estimates.

The support vector machines in scikit-learn support both dense and sparse data.
"""

result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()