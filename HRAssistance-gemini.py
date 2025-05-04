# Crafting an AI-Powered HR Assistant: A Use Case for Nestle's HR Policy Documents

#
# Install python libraries (run these in your terminal, not in the script)
#
# pip install langchain langchain_community python-dotenv  gradio langchain_experimental sentence-transformers langchain_chroma langchainhub unstructured langchain_core faiss-cpu
# pip install langchain-google-genai langchain-chroma python-dotenv langchain_google_genai

import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from google import genai

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr
from dotenv import load_dotenv    
    
def load_main_program():
    #
    # Load pdf file: the_nestle_hr_policy_pdf_2012
    #
    # The Nestlé Human Resources Policy
    #
    # Path is /voc/work
    loader = PyPDFLoader("the_nestle_hr_policy_pdf_2012.pdf")
    data = loader.load()  # entire PDF is loaded as a single Document 
    
    load_dotenv()
    api_key=os.environ['GOOGLE_API_KEY']

    #
    # Split the document
    #
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    #
    # Store the document in vector database
    # Add retriever based on similarity
    #
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
       google_api_key=api_key
    )
    
    llm = ChatGoogleGenerativeAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500)
    
    
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vectorstore = FAISS.from_texts(texts, embedding=embedding, metadatas=metadatas)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    #                                                                     
    
    # Generative AI Chat Model
    #
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=500)
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    question_answering_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answering_chain)
    return rag_chain

def tellme(rag_chain, prompt):
    response = rag_chain.invoke({"input": prompt})
    return response["answer"]

def main():
    rag_chain = load_main_program()
    gr.close_all()

    def generate_answer(message, history):
        return tellme(rag_chain, message)

    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# Nestle's HR Policy")
        gr.Markdown("Ask any questions below:")
        chatbot = gr.Chatbot(height=300)
        message_input = gr.Textbox(placeholder="Ask any questions on HR policy", label="Your question")
        with gr.Row():
            ask_button = gr.Button("Ask")
            clear_button = gr.Button("Clear History")
            quit_button = gr.Button("Quit")
        history = []
        def respond(user_message, chat_history):
            response = generate_answer(user_message, chat_history)
            chat_history = chat_history + [(user_message, response)]
            return chat_history, ""
        def clear_chat():
            return [], ""
        def quit_app():
            sys.exit()
        ask_button.click(fn=respond, inputs=[message_input, chatbot], outputs=[chatbot, message_input])
        clear_button.click(fn=clear_chat, outputs=[chatbot, message_input])
        quit_button.click(fn=quit_app)
    demo.launch(share=True)

if __name__ == "__main__":
    main()
