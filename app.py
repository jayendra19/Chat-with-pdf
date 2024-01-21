import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
import time
from gtts import gTTS#google text to speech 
import os



load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#this function is used to go all the pages and extract the texts from each pages
def get_pdf_text(pdf):
    text=""
    for i  in pdf:
        Pdfreade=PdfReader(i)
        for page in Pdfreade.pages:
            text+=page.extract_text()

    return text


#Now i have text now i'll divide this text into chunks 
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversation():
    prompt_temp="""Answer the question as detailed as possible from the provided context,make sure to provide all the details ,if answer is not in the 
    provided context just say ,"answer is not available in the provided context",don't provide the wrong answer\n\n\
    Context:\n{context}?\n
    question:\n{question}\n

    Answer:
    """

    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompts=PromptTemplate(template=prompt_temp,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompts)

    return chain



def text_to_speech(text):
    speech = gTTS(text = text, lang = 'en', slow = False)
    speech.save("text.mp3")
    os.system("start text.mp3")


def user_input(user_question):
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db=FAISS.load_local("faiss_index",embedding)
    docs=new_db.similarity_search(user_question)

    chain=get_conversation()

    response=chain({"input_documents":docs,"question":user_question},return_only_outputs=True)

    print(response)
    st.write("Reply:",response["output_text"])
    # Convert response to speech
    text_to_speech(response["output_text"])
    st.markdown("Download response as audio")


def listen_and_transcribe():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.record(source, duration=6)  # Listen for 5 seconds
        try:
            text = r.recognize_google(audio)
            return text  # Return the transcribed text
        except sr.UnknownValueError:
            st.text("Sorry, I did not understand that.")
        except sr.RequestError as e:
            st.text(f"Could not request results from Google Speech Recognition service; {e}")
        time.sleep(0.5)  # Optional short pause to limit CPU usage




def main():
    st.set_page_config("Chat_Pdf")
    st.header("Chat with Pdf")

    # Add a button for speech input
    if st.button("Speak"):
         st.text("Listening...")
         user_question = listen_and_transcribe()
         st.text(f"You said: {user_question}")

        

    else:
        user_question = st.text_input("Or type your question here")



    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()











