
# Build a streamlit webapp

import streamlit as st
import time
import os
import pandas as pd
import json
import plotly.express as px
from utils import test_utils, clean_abstract, construct_prompt, query_openai, process_articles,process_json_files,scorecard_modified


st.set_page_config(page_title="ArticleSieve",
    page_icon="🌀",
    layout="wide")
# st.write(test_utils('Hello world'))
st.title("🔍🌀🧠 ArticleSieve")
st.caption('Your smart article screening helper')
# Sidebar with expanders
with st.sidebar:
    st.title("📚 ArticleSieve")
    
    with st.expander("🧭 How to Use the App", expanded=True):
        st.markdown("""
        **1. Upload Articles**      
        Upload your `.csv` file with article abstracts (format: title, abstract).
                    
        **2. View the `.csv` file**      
        Pandas to view the header of the `.csv` file.     
                    
        **3. Run Screening**     
        Click "▶️ Run GPT Query" to let OpenAI analyze the articles and returns `.json` format file for each article.
        
        **4. Data Preprocessing tab**      
        Enter a desired `.csv` file name to be used to write out all `.json` files.  
        Calculate the average scores across the terms and store the dataframe.   
                    
        **5.Data 📊 visualisation tab**     
        Use plotly to plot a scatter plot of the label and scores vs a short title of articles under investigation.
        """)

# Main section
# st.title("🧠 Article Screening App")

st.markdown("<div style='margin-bottom: -30px;color:blue; font-weight:bold;'>🧾Upload your file</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"])
if uploaded_file:
    uploaded_file.seek(0)
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.write(data.head()) # Check the content of the file

        # "Run" button to process articles
        if st.button("▶️ Run GPT Query"):
            with st.spinner("Querying GPT model... please wait",show_time=True):
                uploaded_file.seek(0)  # reset before reading again
                time.sleep(20)
                #process_articles(uploaded_file)  # This function takes the abstract and queries the GPT using the prompt
            st.success("🎉 Hooray! Articles have been processed.")

    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file is empty or unreadable.")
else:
    st.info("📥 Please upload a CSV file to get started.")


tab1,tab2 =st.tabs(["Data preprocessing", "visualisation"])
with tab1:
    st.subheader("Data preprocessing")
    #st.write("Prompt the user for output file name.")
    output_file = st.text_input("Enter the name for the full output CSV (e.g., all_data.csv): ")
    st.write(output_file)

    # May consider future modifications to be asking user the path to where the json files were written to
    json_to_df = process_json_files(directory="./",full_output_file=output_file,table_keyword='article')
    st.write(json_to_df.head())

    st.subheader('Calculate the total scores and display the dataframe')

 
    json_to_df['SCORES'] = json_to_df.apply(scorecard_modified,axis=1)
    # Rename the first column
    json_to_df.rename(columns={"document_info.title": "title"}, inplace=True)
    st.write(json_to_df.head())
    columns = ("term_analysis.bmi_adiposity.present", "term_analysis.bmi_adiposity.is_main_exposure",
            "term_analysis.blood_pressure.present", "term_analysis.blood_pressure.is_main_outcome",
            "term_analysis.mendelian_randomisation.present", "term_analysis.mendelian_randomisation.is_main_method",
            "term_analysis.european_ancestry.present",'term_analysis.european_ancestry.is_ancestry_European')

    data= json_to_df[["title","term_analysis.bmi_adiposity.present", "term_analysis.bmi_adiposity.is_main_exposure",
            "term_analysis.blood_pressure.present", "term_analysis.blood_pressure.is_main_outcome",
            "term_analysis.mendelian_randomisation.present", "term_analysis.mendelian_randomisation.is_main_method",
            "term_analysis.european_ancestry.present",'term_analysis.european_ancestry.is_ancestry_European', 'SCORES']]

    data['short_title'] = data['title'].str[:50]
    data = data.sort_values(by="SCORES", ascending=True)       

# Actual visualisation
with tab2:
    st.subheader("Data visualisation")
    #st.write('Actual visualisation using scatter plot')
    # Create tick/cross label for each row
    tick_cross_map = {True: "✓", False: "✗"}
    data["label"] = data.apply(lambda row: ''.join([tick_cross_map[row[col]] for col in columns]), axis=1)
    # Update the label
    data['label_SCORES'] = data['label']+"_"+data['SCORES'].astype(str)
    print(data["label_SCORES"])
    # Assign a unique position on x-axis
    data["x"] = data["label_SCORES"].astype("category").cat.codes
    #st.write(data.head())
    #st.write(len(data))
    # Use Plotly to create interactive scatter plot
    fig = px.scatter(
        data_frame=data,
        x="x",
        y="short_title",
        color="label_SCORES",
        hover_data=["SCORES"],
        labels={"x": "Variable Combination", "SCORE": "SCORES"},
        title="All articles with (✓/✗) with Scores"
    )

    # Replace x-tick labels with tick/cross combinations
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=data["x"],
            ticktext=data["label_SCORES"],
            tickangle=90
        ),
        height=1000,
        width=500,
        showlegend=False
    )

    #fig.show()
    st.plotly_chart(fig,use_container_width=True)