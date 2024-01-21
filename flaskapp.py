
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
from flask_wtf import FlaskForm
from wtforms import FileField
from gtts import gTTS
import os
from flask import Flask,request,render_template,jsonify,flash
import logging
from io import BytesIO

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#this function is used to go all the pages and extract the texts from each pages
def get_pdf_text(pdf):
    # Read the PDF file from the BytesIO object
    pdf_reader =  PdfReader(pdf)
    text=""
    for page in range(len(pdf_reader.pages)):
        text+=pdf_reader.pages[page]. extract_text()

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
    gTTS(text = text, lang = 'en', slow = False)




def listen_and_transcribe(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
        try:
            text = r.recognize_google(audio)
            return text  # Return the transcribed text
        except sr.UnknownValueError:
            return "Sorry, I did not understand that."
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"
        


def user_input(user_question):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Add logging to check user_question
    logging.debug(f"User Question: {user_question}")

    new_db = FAISS.load_local("faiss_index", embedding)

    docs = new_db.similarity_search(user_question)

    chain = get_conversation()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    response_text = response["output_text"]

     # Convert response to speech if input is speech
    
    text_to_speech(response_text)

    return response_text



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

@app.route("/",methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pdf_file = request.files.get('pdf_docs')
        if pdf_file:
            # Process the PDF file
            pdf_file_io = BytesIO(pdf_file.read())
            raw_text = get_pdf_text(pdf_file_io)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)

            # Display a success message
            flash('File uploaded successfully!', 'success')

    return render_template('home.html')  # Redirect to a success page or update the current page

    


@app.route('/user', methods=['POST'])
def handle_user_input():
    user_question = request.form.get('user_question')


    #Text input
    response_text=user_input(user_question)

    return jsonify({'response_text': response_text})







if __name__ == '__main__':
    app.run(debug=True)






