import re

# Function to analyze project complexity and count
def analyze_project_complexity(projects):
    complexity_scores = []
    
    # Pattern to match project titles followed by descriptions
    project_pattern = r'([a-zA-Z0-9\- ]+?)\s*\|([^|]*?)(?=\s*[a-zA-Z0-9\- ]+\s*\||$)'

    project_matches = re.findall(project_pattern, projects, re.DOTALL)

    # Each match gives a project title and its details
    for title, details in project_matches:
        # Complexity could be defined as the length of the project description
        word_count = len(details.split())
        complexity_scores.append(word_count)

    return complexity_scores, len(complexity_scores)  # Return both scores and count of projects

# Example usage
projects_string = """
education
framefusion|github b.tech in information technology
mernstack,tailwindcss dharmsinhdesaiuniversity,nadiad
(cid:17) 2021–2025 cpi:8.04
• developedanartmarketplace(framefusion)usingmernstackfor
seamlessartist-to-buyertransactions. higher secondary education
• enablingartiststocreateprofiles,upload,andsellartwork,and
mauniinternationalschool,surat
buyerstobrowseandpurchaseartseamlessly.
(cid:17) 2020–2021 per:86.30%
• utilizedawss3forscalableandreliablestorageofartworkimages,
ensuringfastloadingtimes. secondary secondary education
• integraterazorpayforsecurepaymentprocessing,allowingartiststo
mauniinternationalschool,surat
selltheircreationsseamlessly.
(cid:17) 2018–2019 per:83.50%
• integratedsearchandfilteroptionstohelpbuyersfindspecificart
pieceseasily.
skills
quizilla|github • languages:
◦ c++,java,javascript
springboot,react,tailwindcss
• development:
• builtquizilla’suser-friendlyinterface(react)forsubjectselection,
◦ react,node,express,springboot,
quizattempts,andresultviewing,creatinganengaginglearning
springmvc
"""

complexity_scores, project_count = analyze_project_complexity(projects_string)
print(f"Complexity Scores: {complexity_scores}")
print(f"Number of Projects: {project_count}")
