import streamlit as st

from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
import logging

logging.basicConfig(level=logging.INFO)

# Initialize chat history in session state.
if 'messages' not in st.session_state:
    st.session_state.messages = []

def generate_response(model, messages):
    """Stream chat response based on selected model."""

    try:
        # Initialize the language model
        llm = Ollama(model=model, request_timeout=120.0) 
        # Stream chat responses from the model
        resp = llm.stream_chat(messages)
        # Initialize the output
        response = ''
        response_placeholder = st.empty()
        # Append the response to the output
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        # Log the response
        logging.info(f'Model: {model}, Messages: {messages}, Response: {response}')
        return response

    except Exception as e:
        logging.error(f'Error during streaming: {str(e)}')
        raise e

st.title('LLM Chatbot')
logging.info('App started')

# Show the available models in the sidebar and let user pick.
model = st.sidebar.selectbox('Choose a model', ['llama3','mistral'])
logging.info(f'Model selected: {model}')

# Ask for user input.
prompt = st.chat_input('Ask the chatbot')
if prompt:
    # Append the message of user to the messages.
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    logging.info(f'User input: {prompt}')

    # Display all messages written so far.
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])

    # Generate a new response if the last message is from the user.
    if st.session_state.messages[-1]['role'] != 'assistant':
        with st.chat_message('assistant'):
            logging.info('Generating response')

            with st.spinner('Writing...'):
                try:
                    # Get all message history with their corresponding owner.
                    messages = [ChatMessage(role=message['role'], content=message['content']) for message in st.session_state.messages]
                    # Send the messages and get a response.
                    response_message = generate_response(model, messages)
                    # Append the response to messages.
                    st.session_state.messages.append({'role': 'assistant', 'content': response_message})
                    logging.info(f'Response: {response_message}')

                except Exception as e:
                    st.session_state.messages.append({'role': 'assistant', 'content': str(e)})
                    st.error('An error occurred while generating the response.')
                    logging.error(f'Error: {str(e)}')