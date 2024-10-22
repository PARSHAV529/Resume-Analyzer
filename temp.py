import nltk

# # Print current NLTK data paths
# print("Current NLTK data paths:", nltk.data.path)

# # Try downloading the required resources
# try:
#     nltk.download('punkt')
#     nltk.download('punkt_tab')
#     nltk.download('stopwords')
    
#     print("NLTK resources downloaded successfully.")
# except Exception as e:
#     print("Error downloading NLTK resources:", e)

# Create a sample array
# import numpy as np
# import torch

# # Create a sample tensor
# tensor = torch.tensor([1.0, 2.0, 3.0])
# np_array = tensor.numpy()
# print(np_array)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
# import pandas as pd

# # Load the dataset
# df = pd.read_csv('gpt_dataset.csv')


# # Sample 50 random rows
# sampled_df = df.sample(n=50, random_state=1)  # random_state ensures reproducibility

# # Save the sampled data to a new CSV file
# sampled_df.to_csv('sampled_data.csv', index=False)

# print("Sampled data saved to 'sampled_data.csv'")

