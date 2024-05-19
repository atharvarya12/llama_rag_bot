
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama


data_path = input("give the text book location: ")



embedding_func = OpenAIEmbeddings(openai_api_key = "YOUR_API_KEY")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=30,
    length_function=len,)

documents = PyPDFLoader(data_path).load_and_split(text_splitter=text_splitter)



vectordb = Chroma.from_documents(documents, embedding=embedding_func)



template = '''[INST]given the context - {context} [INST] [INST] Answer the following question {question}[/INST] '''

pt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )
     

rag = RetrievalQA.from_chain_type(
            llm=Ollama(model="phi"),
            retriever=vectordb.as_retriever(),
            memory=ConversationSummaryMemory(llm = Ollama(model="phi")),
            chain_type_kwargs={"prompt": pt, "verbose": True},
        )

rag.invoke(input("ask me anything: "))

