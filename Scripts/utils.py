#!/Users/qb21134/miniconda3/envs/automation/bin/python

# Load the right libraries and set up openai API key
import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load API Key
load_dotenv()
#api_KEY = os.getenv('OPENAI_API_KEY') - Personal access key

api_KEY = os.getenv('openaiuob_api_key') # API key via UoB organisation

# Set up OpenAI client
client = OpenAI(api_key=api_KEY)

def test_utils(a):
    return (a) # Test running the script

def clean_abstract(abstract):
    """
    Assess existence of an abstract within the csv file.
    If the abstract is missing, then replace it with a NA.

    Parameters:
    abstract - article abstracts from the csv file.

    Returns: N/A for missing abstract
    Final output is: abstract
    """
    if pd.isna(abstract) or abstract.strip() == "":
        return "N/A"
    return abstract



def construct_prompt(title, abstract):
    """Create a structured prompt for OpenAI API based on title and abstract.

    Parameters:
    user_instructions - user supplied set of instructions.

    title - article title.
    abstract - article abstract.

    Returns a formated prompt to be passed to GPT query func.
    """
    return f"""
    You are a researcher rigorously screening titles and abstracts of scientific papers for inclusion and exclusion in a review paper.
Extract key terms from the following title and abstract and provide analysis in the specified JSON format.

Terms to extract:
1. Adulthood body mass index OR BMI OR adiposity
2. Mendelian randomisation (including alternate spelling "Mendelian randomization")
3. Blood pressure terms: hypertension, high blood pressure, systolic blood pressure, diastolic blood pressure (note: multiple blood pressure terms count as just one match for category #3)
4. European ancestry terms such as European OR white OR caucasian population/ancestry OR mentions European country
5. Review terms such as umbrella review OR scoping review OR systematic review


For each term category, extract the EXACT phrasing as it appears in the document.

IMPORTANT RULES:
- For category #4, terms like "nonwhite," "non-white," "non-European," or similar negations should NOT be counted as matches for European/white/caucasian ancestry/mentions European country.
- Only count exact matches and do not include negated terms (terms with "non-" prefix or similar negations).
- For country names, first extract ALL country names mentioned, then identify which ones are European countries
- If no relevant terms are found, state "No European ancestry or country terms found"
- For each match, provide the exact quote with minimal surrounding context
- CHECK ancestry is European
- Extract only information related to adult body mass index (BMI), obesity in adults, or adiposity measurements in adult populations. DO NOT include any findings related to childhood obesity or BMI in subjects under 18 years of age.
- For BMI/adiposity terms: Check if BMI is being studied as an exposure/predictor of blood pressure, NOT just as a covariate or mediator or outcome.
- For Blood pressure terms: Check if Blood pressure term is studied as an outcome, NOT an exposure or covariate. 
- MUST EXCLUDE any pregnancy and maternal related Blood pressure terms.
- CHECK blood pressure term is Main outcome
- Mendelian randomisation MUST be the method used to study the causal relationship
- Check Main method in use is Mendelian randomisation


Title: "{title}"
Abstract: "{abstract}"

Output MUST strictly follow this JSON structure:
{{
  "document_info": {{
    "title": "{title}",
    "abstract_preview": "{abstract[:50]}..."
  }},
  "term_analysis": {{
    "bmi_adiposity": {{
      "present": true/false,
      "locations": ["title", "abstract"],
      "variations_found": ["exact phrases as they appear in the text"],
      "is_main_exposure": true/false
    }},
    "mendelian_randomisation": {{
      "present": true/false,
      "locations": ["title", "abstract"],
      "variations_found": ["exact phrases as they appear in the text"],
      "is_main_method": true/false
    }},
    "blood_pressure": {{
      "present": true/false,
      "locations": ["title", "abstract"],
      "variations_found": ["exact phrases as they appear in the text"],
      "is_main_outcome": true/false
      
    }},
    "european_ancestry": {{
      "present": true/false,
      "locations": ["title", "abstract"],
      "variations_found": ["exact phrases as they appear in the text"],
      "is_ancestry_European": true/false
    }},
    "reviews": {{
      "present": true/false,
      "locations": ["title", "abstract"],
      "variations_found": ["exact phrases as they appear in the text"]
    }}
  }},
  "Reason": {{
    "justify": "Brief string of all the EXACT terms present in the title and abstract AND record those that are missing."
  }}
}}

Respond **ONLY** with the JSON output and nothing else.

"""


def query_openai(prompt):
    """
    Query OpenAI API and return a structured JSON response.
    
    Parameter:
    Prompt - a structured, precise and clear question with the json output format to pass to LLM.
    Return:
    A json output to be written out by another function.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in biomedical text analysis. Return valid JSON ONLY."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        response_text = response.choices[0].message.content.strip()
        
        # Debugging: print response for inspection
        print(f"DEBUG: OpenAI Response:\n{response_text}")

        # Ensure the response is valid JSON
        return json.loads(response_text)
    
    except json.JSONDecodeError:
        print("ERROR: OpenAI returned invalid JSON. Skipping this article.")
        return None
    except Exception as e:
        print(f"ERROR: Failed to query OpenAI - {e}")
        return None


def process_articles(uploaded_file):
    """
    Process the csv file with title and abstract, query using GPT and return a structured json format for each article.
    Parameter:
    uploaded_file - csv file containing article title and abstract.
    Return:
    Json file - json file for each article.
    
    """
    
    df = pd.read_csv(uploaded_file) # read in the article
    for idx, row in df.iterrows():
        title = row["title"]
        abstract = clean_abstract(row.get("abstract", ""))  # Check validity of the article

        prompt = construct_prompt(title, abstract)
        response_data = query_openai(prompt)

        if response_data is None:
            print(f"Skipping article {idx} due to invalid OpenAI response.")
            continue  # Skip this iteration if response is invalid

        output_filename = f"article_{idx}.json"
        with open(output_filename, "w") as json_file:
            json.dump(response_data, json_file, indent=4)

        print(f"Processed {title} -> {output_filename}")