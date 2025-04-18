
# Build a streamlit webapp

import streamlit as st
import time
import os
import pandas as pd
import json
import plotly.express as px
from utils import test_utils,clean_abstract,construct_prompt,query_openai,process_articles, process_json_files


st.set_page_config(page_title="ArticleSieve",
    page_icon="🌀",
    layout="wide")
st.write(test_utils('Hello world'))
# st.subheader("Portable function")
# def clean_abstract(abstract):
#     """Replace missing or empty abstracts with 'N/A'."""
#     if pd.isna(abstract) or abstract.strip() == "":
#         return "N/A"
#     return abstract
st.title("🔍🌀🧠 ArticleSieve")
st.caption('Your smart article filter ')
# Sidebar with expanders
with st.sidebar:
    st.title("📚 ArticleSieve")
    
    with st.expander("🧭 How to Use the App", expanded=True):
        st.markdown("""
        **1. Upload Articles**  
        Upload your `.csv` file with article abstracts (format: title, abstract).
                    
        **2. Data preprocessing**   
        Read in the dataset using pandas and preprocess the data.                        
        **2. Set Screening Criteria**  
        Write a custom prompt or choose from presets.

        **3. Run Screening**  
        Click "Start Screening" to let ChatGPT analyze the articles.

        **4. Review & Download**  
        Review included/excluded articles and download results.
        """)

    with st.expander("📊 Visualizations", expanded=False):
        st.markdown("Visual summaries of scores and selections will appear here.")

# Main section
st.title("🧠 Article Screening App")

uploaded_file = st.file_uploader("Upload your file", type=["csv"])
data =None
if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == 'csv':
        with st.spinner('Wait as I query each article using GPT models....', show_time=True):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.05)
                #process_articles(uploaded_file)
                progress_bar.progress(percent_complete + 1)
        st.success('Hooray! Completed querying the articles')
        st.button("Rerun")
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file) 
        st.success("File uploaded successfully!")
        st.write(data.head())
        
    else:
        st.error("Unsupported format")
else:
    st.info("👈 Upload a file to get started.")


# with st.spinner('Wait as I query GPT....', show_time=True):
#      st.write(process_articles(uploaded_file)
# st.success('Completed querying the articles')
# st.button("Rerun")

#st.write(data.head())
# st.write(os.getcwd())

# st.subheader("Ask the user to input a prompt of interest")
# user_instructions = st.text_input("Please enter a set of instructions for GPT.")
# st.write(user_instructions)
# st.title('Filtering articles by title and abstract')

# st.header("This app takes in a dataset of titles and abstracts")

# st.subheader("This app uses GPT to extract key terms")

# st.text('It is time to launch the app')

# st.markdown("## This is a markdown text")

# st.success('Hooray')

# st.info("Information box")

# st.write(range(10))

# st.write("Normalise the json files")
# # Reading in the data and setting up tabs

# tab1,tab2 =st.tabs(["Data", "visualisation"])
# with tab1:
#     with open ("../LLM_promptengineering/json1.json") as inputfile:
#         data = json.load(inputfile)
#         df= pd.json_normalize(data)
#     st.write(df.columns)

# with tab2:
#     st.write(df.head())

# st.write("Calculate the score for each article")

st.subheader("Data preprocessing")
#st.write("Prompt the user for output file name.")
output_file = st.text_input("Enter the name for the full output CSV (e.g., all_data.csv): ")
st.write(output_file)

# May consider future modifications to be asking user the path to where the json files were written to
process_json_files(directory="./",full_output_file=output_file,table_keyword='article')


st.text('Calculate the total scores from the merged dataframe')

st.subheader("Data visualisation")