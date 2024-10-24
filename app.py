import streamlit as st
import pandas as pd
import re
import pdfplumber
import io
import os
import spacy
from transformers import BertTokenizer, BertModel
import torch
import base64

# Cache the loading of spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load('en_core_web_md')

# Cache the loading of BERT tokenizer and model
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

# Function to check if two words are similar using spaCy
def are_synonyms(word1, word2, nlp, threshold=0.8):
    doc1 = nlp(word1)
    doc2 = nlp(word2)
    if doc1.vector_norm and doc2.vector_norm:  # Check for valid vectors
        return doc1.similarity(doc2) >= threshold
    return False

# Function to list all PDF files in the resumes folder
def list_resume_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_content):
    pdf_text = ""
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            pdf_text += page.extract_text() or ""
    return pdf_text

# Function to extract key information from the resume
def extract_information(text):
    text = text.lower()
    skills_match = re.search(r'skills\s*([\s\S]*?)(?:experience|project|certification|$)', text)
    skills = skills_match.group(1).strip() if skills_match else ""
    experience_match = re.search(r'experience\s*([\s\S]*?)(?:project|certifications|$)', text)
    experience = experience_match.group(1).strip() if experience_match else ""
    projects_match = re.search(r'project\s*([\s\S]*?)(?:certification|experience|$)', text)
    projects = projects_match.group(1).strip() if projects_match else ""
    certifications_match = re.search(r'certification\s*([\s\S]*?)(?:experience|project|skills|$)', text)
    certifications = certifications_match.group(1).strip() if certifications_match else ""

    return {
        "Skills": skills,
        "Experience": experience,
        "Projects": projects,
        "Certifications": certifications
    }

# Function to preprocess and tokenize skills
def preprocess_skills(skills_text):
    skills = [skill.strip() for skill in skills_text.split(',') if skill.strip()]
    return skills

# Function to encode text using BERT
def encode_text(text, tokenizer, model):
    if not text:  # Handle empty text
        return None
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to analyze project complexity and count projects
def analyze_project_complexity(projects):
    complexity_scores = []
    project_count = count_projects(projects)  # Count the number of projects

    project_list = re.split(r'\s*[\n.]+\s*', projects.strip())
    project_descriptions = [proj for proj in project_list if proj]  # Filter out empty strings

    for description in project_descriptions:
        word_count = len(description.split())
        complexity_scores.append(word_count)

    return complexity_scores, project_count

nlp = load_spacy_model()

def count_projects(projects):
    doc = nlp(projects)
    project_count = 1  # Start with 1 as base count
    project_keywords = ["project", 'projects', 'source code', 'github']

    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in project_keywords):
            project_count += 1

    return project_count

# Function to create a download link for a PDF file
def create_download_link(pdf_content, file_name):
    b64 = base64.b64encode(pdf_content).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return href

# Streamlit UI
st.title("Resume Analyzer")

# Load models and cache them
tokenizer, model = load_bert_model()

# Set the folder path for resumes
folder_path = "resumes"

# Use session_state to manage data between interactions
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
    st.session_state.resume_files = {}

# Fetch resumes from the folder
if st.button("Fetch Resumes"):
    files = list_resume_files(folder_path)
    
    if files:
        st.write("Available Resumes:")
        extracted_data = []
        resume_files = {}

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "rb") as f:
                pdf_content = f.read()
                text = extract_text_from_pdf(pdf_content)
                extracted_info = extract_information(text)
                extracted_info['Resume Name'] = file_name
                extracted_data.append(extracted_info)
                resume_files[file_name] = pdf_content

        st.session_state.resume_data = pd.DataFrame(extracted_data)
        st.session_state.resume_files = resume_files
        st.write(st.session_state.resume_data)
    else:
        st.write("No resumes found in the specified folder.")

# Input fields for job description and skills
job_description = st.text_area("Job Description").lower()
must_have_skills = st.text_input("Must-Have Skills (comma separated)")
extra_skills = st.text_input("Extra Skills (comma separated)")
num_top_resumes = st.number_input("Number of Top Ranked Resumes to Display", min_value=1, value=5)

# Analyze resumes
if st.button("Analyze Resumes") and st.session_state.resume_data is not None:
    job_description_embedding = encode_text(job_description, tokenizer, model)

    must_have_skills_list = preprocess_skills(must_have_skills)
    extra_skills_list = preprocess_skills(extra_skills)

    st.session_state.resume_data['processed_text'] = (
        st.session_state.resume_data['Skills'] + ' ' + 
        st.session_state.resume_data['Experience'] + ' ' + 
        st.session_state.resume_data['Projects'] + ' ' + 
        st.session_state.resume_data['Certifications']
    )
    st.session_state.resume_data['embedding'] = st.session_state.resume_data['processed_text'].apply(lambda x: encode_text(x, tokenizer, model) if x else None)

    # Filter out rows where embedding is None (i.e., missing text)
    st.session_state.resume_data = st.session_state.resume_data[st.session_state.resume_data['embedding'].apply(lambda x: x is not None)]

    # Initialize the score column
    st.session_state.resume_data['score'] = 0.0

    st.session_state.resume_data['project_complexity'], st.session_state.resume_data['project_count'] = zip(*st.session_state.resume_data['Projects'].apply(analyze_project_complexity))

    for index, row in st.session_state.resume_data.iterrows():
        skills = row['Skills'].lower()
        projects = row['Projects'].lower()
        certifications = row['Certifications'].lower()

        total_must_have = len(must_have_skills_list)
        matched_skills = sum(any(are_synonyms(skill, must_have_skill, nlp) for skill in skills.split()) for must_have_skill in must_have_skills_list)

        if total_must_have > 0:
            match_percentage = (matched_skills / total_must_have) * 100

            if match_percentage >= 50:
                st.session_state.resume_data.at[index, 'score'] += 5
                if any(any(are_synonyms(skill, extra_skill, nlp) for skill in skills.split()) for extra_skill in extra_skills_list):
                    st.session_state.resume_data.at[index, 'score'] += 2

                project_count = row['project_count']
                project_complexity = sum(row['project_complexity'])
                
                st.session_state.resume_data.at[index, 'score'] += project_count * 1  # One point per project
                st.session_state.resume_data.at[index, 'score'] += project_complexity / 100  # Normalize complexity score

                # Score for certifications
                if 'certification' in certifications:
                    st.session_state.resume_data.at[index, 'score'] += 1  # Small score for certifications

    # Ensure that resumes with zero scores are excluded
    st.session_state.resume_data = st.session_state.resume_data[st.session_state.resume_data['score'] > 0]
    # Update the Experience and Certifications columns to 'Yes' or 'No'
    st.session_state.resume_data['Experience'] = st.session_state.resume_data['Experience'].apply(lambda x: 'Yes' if x.strip() else 'No')
    st.session_state.resume_data['Certifications'] = st.session_state.resume_data['Certifications'].apply(lambda x: 'Yes' if x.strip() else 'No')


    # Sort resumes by score and display top N
    sorted_resumes = st.session_state.resume_data.sort_values(by='score', ascending=False)
    
    if 'score' in sorted_resumes.columns:
        top_resumes = sorted_resumes.head(num_top_resumes)
        
        # Define the columns to display
        display_columns = ['Resume Name', 'score', 'project_count', 'Experience', 'Certifications']
        st.table(top_resumes[display_columns])

        # Provide download links for top resumes
        for resume_name in top_resumes['Resume Name']:
            st.markdown(create_download_link(st.session_state.resume_files[resume_name], resume_name), unsafe_allow_html=True)
    else:
        st.write("No resumes found with valid scores. Ensure that the analysis process completed successfully.")
else:
    st.write("Click the 'Fetch Resumes' button first to load resumes for analysis.")
