# cd Desktop\DATA\WCS_Projet3\Job_Pirates
# streamlit run streamlit_job_pirates.py

import os
import pandas as pd
import streamlit as st
from groq import Groq
from PIL import Image
import base64
import numpy as np
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from PyPDF2 import PdfReader
from docx import Document

background_url = "https://jsginc.com/wp-content/uploads/2018/10/bigstock-170353778-1024x576.jpg.webp"
page_bg_img = f'''
<style>
.stApp {{
  background-image: url("{background_url}");
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# image de notre logo sur fond rouge
st.image(r"C:\Users\user\Downloads\JobPirates.png")

url_image = "https://www.childfun.com/wp-content/uploads/2009/01/pirate.png"
header_html = f"""
<div style="display: flex; align-items: center; justify-content: space-between;">
    <h1>Shoot your questions, sailor!</h1>
    <img src="{url_image}" width="100" style="margin-left: 20px;">
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Config API Groq
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ["GROQ_API_KEY"])
def chat_groq(t=0, choix="llama3-70b-8192", api=client): 
    return ChatGroq(temperature=t, model_name=choix, groq_api_key=api.api_key)
model_chat = chat_groq()

# données jobs dans le monde
loader = CSVLoader(file_path=r"C:\Users\user\Desktop\DATA\WCS_Projet3\Job_Pirates\World_jobs_data.csv", 
                   csv_args={'delimiter': ',', 'fieldnames': ['description_cie', 'description_job']}, encoding='utf8')
documents = loader.load()

# Définir l'embedding
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
text_chunks = text_splitter.split_documents(documents)

# Créer le VectorStore FAISS et ajouter les documents
db = FAISS.from_documents(text_chunks, embeddings_model)

# fonction pour les données de job
class jobs(BaseModel):
    programming_tools: List[str] = Field(description="List of programming software for the job posting")
    hard_skills: List[str] = Field(description="List of technical skills keywords for the job posting")
    soft_skills: List[str] = Field(description="List of soft skills for the job posting")
    salary: List[str] = Field(description="List of salary with units")
    experience_level: List[str] = Field(description="Required level of experience")
    study_level: List[str] = Field(description="Required level of study")
    remote: List[str] = Field(description="Level of remote")
    city: List[str] = Field(description="City")
    dpt: List[str] = Field(description="Department")
    contract: List[str] = Field(description="Type of contract")

parser = JsonOutputParser(pydantic_object=jobs)
template = """
You are an expert assistant in extracting keywords related to technical and soft skills, salaries, experience level, and education from a job posting. 
Use the following retrieved context elements to answer the question. If you don't know the answer, simply say you don't know. 
Return only the result in JSON format without any other text. Be precise in one or two words. The returned result must be in English.
<context>
{context}
</context>
Question: {input}
{format_instruction}
"""
prompt = ChatPromptTemplate.from_template(template, partial_variables={"format_instruction": parser.get_format_instructions()})
document_chain = create_stuff_documents_chain(model_chat, prompt, output_parser=parser)
retriever = db.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

def load_cv(file_path):
    if file_path.endswith('.pdf'):
        return load_pdf(file_path)
    elif file_path.endswith('.docx'):
        return load_docx(file_path)
    else:
        raise ValueError("Format de fichier non supporté")

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ''.join([page.extract_text() + '\n' for page in reader.pages])
    return text

def load_docx(file_path):
    doc = Document(file_path)
    text = ''.join([paragraph.text + '\n' for paragraph in doc.paragraphs])
    return text

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []
if 'cv_text' not in st.session_state:
    st.session_state['cv_text'] = None
if 'cv_vectors' not in st.session_state:
    st.session_state['cv_vectors'] = []

uploaded_file = st.file_uploader("Upload your CV", type=["pdf", "docx"])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    st.write(file_details)
    try:
        if uploaded_file.type == "application/pdf":
            st.session_state['cv_text'] = load_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            st.session_state['cv_text'] = load_docx(uploaded_file)
        else:
            st.error("Format de fichier non supporté.")
        st.write("Contenu du CV :")
        st.write(st.session_state['cv_text'][:500])
        texts_split = text_splitter.split_text(st.session_state['cv_text'])
        cv_docs = text_splitter.create_documents(texts_split)
        st.session_state['cv_vectors'] = embeddings_model.embed_documents([chunk.page_content for chunk in cv_docs])
    except Exception as e:
        st.error(f"Erreur lors du chargement du CV : {e}")

with st.form(key='form'):
    user_input = st.text_input("You: ", key="input")
    submit_button_pressed = st.form_submit_button("Submit")
    search_jobs_pressed = st.form_submit_button("Search matching jobs with uploaded CV")

def generate_chat_response(conversation):
    try:
        chat_completion = client.chat.completions.create(
            messages=conversation,
            model="llama3-70b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error in generating chat response: {e}")
        return "Sorry, there was an error processing your request."

def search_matching_jobs():
    if st.session_state['cv_vectors']:
        k = 3
        results = []
        for cv_vector in st.session_state['cv_vectors']:
            similar_docs = db.similarity_search_by_vector(cv_vector, k=k)
            results.extend(similar_docs)
        unique_results = list({doc.metadata['source']: doc for doc in results}.values())
        top_2_results = sorted(unique_results, key=lambda x: x.metadata.get('score', 0), reverse=True)[:2]
        return top_2_results
    else:
        st.warning("Please upload your CV first.")
        return []

if submit_button_pressed:
    conversation = [
        {"role": "system", "content": "You are an assistant that helps with various queries."}
    ]
    for past_input, past_output in zip(st.session_state.past, st.session_state.generated):
        conversation.append({"role": "user", "content": past_input})
        conversation.append({"role": "assistant", "content": past_output})
    conversation.append({"role": "user", "content": user_input})
    if st.session_state['cv_text']:
        conversation.append({"role": "user", "content": f"Here is my CV content: {st.session_state['cv_text'][:2000]}"})

    response = generate_chat_response(conversation)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)

if search_jobs_pressed:
    top_2_results = search_matching_jobs()
    if top_2_results:
        for i, doc in enumerate(top_2_results, 1):
            st.write(f"Annonce {i}:")
            st.write(f"Source: {doc.metadata['source']}")
            st.write(f"Contenu: {doc.page_content[:200]}...")

if st.button("Réinitialiser"):
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['uploaded_files'] = []
    st.session_state['cv_text'] = None
    st.session_state['cv_vectors'] = []

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.text_area(f"Assistant (Message {i+1})", st.session_state['generated'][i], key=f"generated_{i}")
        st.text_area(f"You (Message {i+1})", st.session_state['past'][i], key=f"past_{i}", disabled=True)