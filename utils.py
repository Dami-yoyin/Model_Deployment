from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.document_loaders import TextLoader, DirectoryLoader
import os
from langchain import VectorDBQA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader

os.environ["OPENAI_API_KEY"] = "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"


#Loading and splitting documents
def load_and_split(file_path):
    loader = CSVLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts=text_splitter.split_documents(documents)
    return texts

#creating embedding for vector storage
def db():
    persist_directory = 'C:/Users/damil/Documents/db'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents= load_and_split('C:/Users/damil/Documents/data_.csv'),
                                embedding=embedding,
                                persist_directory=persist_directory)
    retriever = vectordb.as_retriever(search_kwargs={"k":2})
    return retriever

#Making a chain
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                      chain_type = "stuff",
                                      retriever = db(),
                                      return_source_documents= True)
def output_question(query):
    try:
        llm_response = qa_chain(query)
        user_question = query
        ground_truth = llm_response['source_documents']
        ground_truth_answers = ground_truth[0].page_content.split("question: ")[1].split("\n")[0]
        answers =ground_truth[0].page_content.split("additional_info: ")[1].split("\n")[0]
#         print(answers)
    except IndexError:
        ground_truth_answers = " "
    
    vectorizer = TfidfVectorizer()
    answer_vectors = vectorizer.fit_transform([user_question, ground_truth_answers])
    similarity = cosine_similarity(answer_vectors[0], answer_vectors[1])[0][0]
#     print(similarity)
    if similarity >= 0.5:
        print(f"Result: {answers}")
    else:
        persist_directory = 'gpt_3'
        embedding = OpenAIEmbeddings()
        vectordb2 = Chroma(embedding_function=embedding,
                           persist_directory=persist_directory)
        retriever = vectordb2.as_retriever(search_kwargs={"k":2})
        gpt = ChatOpenAI(temperature=0,
                model_name = "gpt-3.5-turbo")
        qa_chain2 = RetrievalQA.from_chain_type(llm=gpt,
                                      chain_type = "stuff",
                                      retriever = retriever,
                                      return_source_documents= True)
        llm_response2 = qa_chain2(query)
        print(f"Result: {llm_response2['result']}")      