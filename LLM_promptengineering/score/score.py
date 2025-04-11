import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load API Key
load_dotenv()
api_KEY = os.getenv('OPENAI_API_KEY')

# Set up OpenAI client
client = OpenAI(api_key=api_KEY)

def get_csv_file():
    """Prompt the user for the CSV file path."""
    file_path = input("Enter the path to the CSV file: ").strip()
    while not os.path.exists(file_path):
        print("File not found. Please enter a valid file path.")
        file_path = input("Enter the path to the CSV file: ").strip()
    return file_path

def clean_abstract(abstract):
    """Replace missing or empty abstracts with 'N/A'."""
    if pd.isna(abstract) or abstract.strip() == "":
        return "N/A"
    return abstract

def construct_prompt(title, abstract):
    """Create a structured prompt for OpenAI API based on title and abstract."""
    return f"""
    You are a researcher rigorously screening titles and abstracts of scientific papers for inclusion or exclusion in a review paper.
Extract key terms from the following title and abstract and provide analysis in the specified JSON format.

Terms to extract:
1. Adulthood body mass index OR BMI OR adiposity
2. Mendelian randomisation (including alternate spelling "Mendelian randomization")
3. Blood pressure terms: hypertension, high blood pressure, systolic blood pressure, diastolic blood pressure (note: multiple blood pressure terms count as just one match for category #3)
4. European OR white OR Caucasian population/ancestry OR mentions European country

For each term category, extract the EXACT phrasing as it appears in the document.

IMPORTANT RULES:
- For category #4, terms like "nonwhite," "non-white," "non-European," or similar negations should NOT be counted as matches for European/white/caucasian ancestry/mentions European country.
- Only count exact matches and do not include negated terms (terms with "non-" prefix or similar negations).
- For BMI/adiposity terms: Check if BMI is being studied as an exposure/predictor of blood pressure, not just as a covariate or mediator or outcome.
- For Blood pressure terms: Check if Blood pressure term is studied as an outcome, not an exposure or mediator or covariate.

Calculate relevance score using these point system:
- BMI/adiposity as main exposure: +1 point
- BMI/adiposity present but not as main exposure: -1 points
- BMI/adiposity absent: -2 points
- Blood pressure terms present: +1 point
- Blood pressure terms present but not as main outcome: -1 point
- Blood pressure terms absent: -2 point
- Mendelian randomisation present: +1 point
- Mendelian randomisation absent: -1 point
- European/white/caucasian ancestry present: +1 point
- European/white/caucasian ancestry absent: -2 point

The final score is the sum of these individual scores.

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
      "variations_found": ["exact phrases as they appear in the text"]
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
      "variations_found": ["exact phrases as they appear in the text"]
    }}
  }},
  "score": {{
    "value": [calculated total score],
    "breakdown": {{
      "bmi_adiposity": [+1 or -1 or -2],
      "mendelian_randomisation": [+1 or -1],
      "blood pressure terms": [+1 or -1 or -2],
      "european_ancestry": [+1 or -2]
    }},
    "justification": "Brief explanation of score based on exact terms present/absent and whether BMI is a main exposure variable."
  }}
}}

Respond **ONLY** with the JSON output and nothing else.

"""

def query_openai(prompt):
    """Query OpenAI API and return a structured JSON response."""
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

def process_articles(csv_file):
    """Process each article from the CSV and save the extracted information as JSON files."""
    df = pd.read_csv(csv_file)

    for idx, row in df.iterrows():
        title = row["title"]
        abstract = clean_abstract(row.get("abstract", ""))  # Ensure abstract is valid

        prompt = construct_prompt(title, abstract)
        response_data = query_openai(prompt)

        if response_data is None:
            print(f"Skipping article {idx} due to invalid OpenAI response.")
            continue  # Skip this iteration if response is invalid

        output_filename = f"article_{idx}.json"
        with open(output_filename, "w") as json_file:
            json.dump(response_data, json_file, indent=4)

        print(f"Processed {title} -> {output_filename}")

if __name__ == "__main__":
    csv_file = get_csv_file()
    process_articles(csv_file)