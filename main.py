from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import streamlit as st
from utils import *
from streamlit_chat import message

def main():
    st.title(" BG Chatbot")

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="")

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


    # conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

    # container for chat history
    response_container = st.container()

    # container for text box
    textcontainer = st.container()


    with textcontainer:
        query = st.text_input("Question: ", key="input")
        if query:
            with st.spinner("typing..."):
                # conversation_string = get_conversation_string()
                # # st.code(conversation_string)
                # refined_query = query_refiner(conversation_string, query)
                # st.subheader("Refined Query:")
                # st.write(refined_query)
                # context = find_match(refined_query)
                # # print(context)  
                # response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                response = output_question(query)
            st.session_state.requests.append(query)
            st.session_state.responses.append(response) 

#     with response_container:
#         if st.session_state['responses']:

#             for i in range(len(st.session_state['responses'])):
#                 message(st.session_state['responses'][i],key=str(i))
#                 if i < len(st.session_state['requests']):
#                     message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

if __name__ == "__main__":
    main()          