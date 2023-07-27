import streamlit as st
# from streamlit_chat import message
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chat_models import ChatOpenAI
import tempfile

openai_api_key = "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"

def generate_response(uploaded_file, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','})
        documents = loader.load()
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts=text_splitter.split_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        database = FAISS.from_documents(texts, embeddings)
        # Create retriever interface
        retrievers = database.as_retriever(search_kwargs={"k":2})
        # Create QA chain
        qa_chain = ConversationalRetrievalQA.from_llm(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retrievers, return_source_documents= True)
        return qa_chain.run(query_text)
        
# def output():
#     try:
#         llm_response = generate_response(uploaded_file, query_text)
#         user_question = query_text
#         ground_truth = llm_response['source_documents']
#         ground_truth_answers = ground_truth[0].page_content.split("question: ")[1].split("\n")[0]
#         answers =ground_truth[0].page_content.split("additional_info: ")[1].split("\n")[0]
        
#     except IndexError:
#         ground_truth_answers = " "
    
#     vectorizer = TfidfVectorizer()
#     answer_vectors = vectorizer.fit_transform([user_question, ground_truth_answers])
#     similarity = cosine_similarity(answer_vectors[0], answer_vectors[1])[0][0]
# #     print(similarity)
#     if similarity >= 0.5:
#         return answers
#     else:
#         persist_directory = 'gpt_3'
#         embedding = OpenAIEmbeddings()
#         vectordb2 = Chroma(embedding_function=embedding,
#                            persist_directory=persist_directory)
#         retriever = vectordb2.as_retriever(search_kwargs={"k":2})
#         gpt = ChatOpenAI(temperature=0,
#                 model_name = "gpt-3.5-turbo")
#         qa_chain2 = RetrievalQA.from_chain_type(llm=gpt,
#                                       chain_type = "stuff",
#                                       retriever = retriever,
#                                       return_source_documents= True)
#         return qa_chain2.run(query_text)['result']
#         # llm_response2 = qa_chain2(query)
#         # return llm_response2['result']

    
# Page title
st.set_page_config(page_title='🦜🔗BG- Ask the Doc App')
st.title('🦜🔗 BG - Ask the Doc App')

# File upload
uploaded_file = st.sidebar.file_uploader('Upload document', type='csv')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.spinner('In Progress...'):
    # response = output()
    response = generate_response(uploaded_file, query_text)
    result.append(response)
    
# with st.form('myform', clear_on_submit=True):
#     openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
#     submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
#     if submitted and openai_api_key.startswith('sk-'):
#         with st.spinner('Calculating...'):
#             response = generate_response(uploaded_file, openai_api_key, query_text)
#             result.append(response)
#             del openai_api_key

if len(result):
    st.info(response)
