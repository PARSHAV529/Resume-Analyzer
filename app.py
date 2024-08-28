import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity



# Load resumes dataset
resumes_df = pd.read_csv('gpt_dataset.csv')

# Define the job description
job_description = """
Looking for a backend developer with experience in building responsive and user-friendly web interfaces.
Must be proficient in JavaScript Node.js Express.js, MongoDB, SQL, NoSQL.
Experience with API testing is a plus.
"""

# Function to extract important terms (keywords) from text
def extract_keywords(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    frequency_dist = nltk.FreqDist(tokens)
    return set(frequency_dist.keys())

# Function to clean and preprocess text
def preprocess_text(text, important_terms):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and (word not in stop_words or word in important_terms)]
    return ' '.join(tokens)

# Extract important terms from the job description
important_terms = extract_keywords(job_description)

# Preprocess the job description
processed_job_description = preprocess_text(job_description, important_terms)

# Apply preprocessing to resumes
resumes_df['processed_text'] = resumes_df['Resume'].apply(lambda x: preprocess_text(x, important_terms))

# Define a function to encode text using BERT
def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode the job description
job_description_embedding = encode_text(processed_job_description, tokenizer, model)

# Encode resumes
resumes_df['embedding'] = resumes_df['processed_text'].apply(lambda x: encode_text(x, tokenizer, model))

# Function to check if all must-have skills, extra skills, or projects are present
def check_criteria(resume, must_have_skills, extra_skills, projects):
    resume_words = set(resume.split())
    missing_skills = [skill for skill in must_have_skills if skill not in resume_words]
    extra_skill_matches = [skill for skill in extra_skills if skill in resume_words]
    project_matches = [project for project in projects if project in resume_words]
    
    priority = 0
    if project_matches:
        priority = 1
    elif not missing_skills and extra_skill_matches:
        priority = 2
    elif not missing_skills:
        priority = 3
    else:
        priority = 4
    
    return priority, missing_skills, extra_skill_matches, project_matches

# Define criteria
must_have_skills = ['javascript', 'node.js', 'express.js', 'sql', 'nosql']
extra_skills = ['mongodb']
projects = ['frontend development', 'responsive web design', 'web interface development']

# Apply criteria check and calculate similarities
resumes_df['criteria_check'] = resumes_df['processed_text'].apply(
    lambda x: check_criteria(x, must_have_skills, extra_skills, projects)
)

# Add columns for priority and missing elements
resumes_df['priority'] = resumes_df['criteria_check'].apply(lambda x: x[0])
resumes_df['missing_skills'] = resumes_df['criteria_check'].apply(lambda x: x[1])
resumes_df['extra_skill_matches'] = resumes_df['criteria_check'].apply(lambda x: x[2])
resumes_df['project_matches'] = resumes_df['criteria_check'].apply(lambda x: x[3])

# Convert embeddings to numpy array for similarity calculation
resume_embeddings = torch.stack([torch.tensor(embedding) for embedding in resumes_df['embedding']])
similarities = cosine_similarity(job_description_embedding.reshape(1, -1), resume_embeddings)

# Add similarity scores to DataFrame
resumes_df['similarity_score'] = similarities[0]

# Sort resumes by priority and similarity score
resumes_df = resumes_df.sort_values(by=['priority', 'similarity_score'], ascending=[True, False])

# Save the ranked resumes
resumes_df.to_csv('ranked_resumes.csv', index=False)

# Print top 20 resumes
print("Top 20 Resumes:")
print(resumes_df[['Resume', 'priority', 'similarity_score', 'missing_skills', 'extra_skill_matches', 'project_matches']].head(20))
