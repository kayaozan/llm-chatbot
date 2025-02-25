# LLM Chatbot
This project demonstrate how to build a chatbot powered by LLMs and deploy it locally.

## Dependencies
The script is written in Python. The required software and libraries are as following:
- [Ollama](https://ollama.com/): lets the user to run LLMs. It is installed locally and the LLMs used in this project are downloaded.
- [LlamaIndex](https://github.com/run-llama/llama_index): a data framework for LLM applications. It is, in simple terms, the bridge between Python and Ollama(or any LLM like ChatGPT).
- [Streamlit](https://streamlit.io/): the library to build the chatbot app.

## Breakdown
- The user is asked to provide a prompt.
- The prompt is sent to the LLM, and the response is generated by the LLM.
- All messaging history is shown in real time.
- A couple of LLMs are also listed to be selected by the user.

## Final Look
Here is how the page looks once the script is run by streamlit in a web browser:

![Screenshot 2025-02-02 173819](https://github.com/user-attachments/assets/86b1b251-1b3b-44de-856a-ce661e49c6c6)
