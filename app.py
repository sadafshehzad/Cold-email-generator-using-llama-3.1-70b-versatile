import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import pandas as pd
import re

import uuid
import chromadb

load_dotenv()

llm = ChatGroq(temperature=0, api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")


st.title(" Cold email Generator")
url_input=st.text_input("Enter a URL:", value="")

submit_button=st.button("Submit")



if submit_button:
    
    loader = WebBaseLoader([url_input])
    text=loader.load().pop().page_content

    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    # Remove extra whitespace
    text = ' '.join(text.split())

    prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the 
            following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):    
            """
    )

    chain_extract = prompt_extract | llm 
    res = chain_extract.invoke(input={'page_data':text})



    json_parser = JsonOutputParser()
    json_res = json_parser.parse(res.content)


    data = pd.read_csv("company_portfolio.csv")

    client = chromadb.PersistentClient('vectorstore')
    collection = client.get_or_create_collection(name="portfolio")

    if not collection.count():
        for _, row in data.iterrows():
            collection.add(documents=row["Techstack"],
                        metadatas={"links": row["Links"]},
                        ids=[str(uuid.uuid4())])
            

    jobs=json_res
    for job in jobs:
                skills = job.get('skills', [])
    
    links = collection.query(query_texts=job['skills'], n_results=2).get('metadatas', [])
    

    prompt_email = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}

    ### INSTRUCTION:
    You are Sadaf, a business development executive at ABC company . ABC company is an AI & Software Consulting company dedicated to facilitating
    the seamless integration of business processes through automated tools and offer skilled people for their respective job description. 
    Over our experience, the company has empowered numerous enterprises with tailored solutions, fostering scalability, 
    process optimization, cost reduction, and heightened overall efficiency. 
    Your job is to write a COLD EMAIL to the client regarding the job mentioned above ,describing in detail about the capability of ABC 
    in fulfilling their needs and the successful completion of prior projects. Also mention in tabular form  professionally about the 'roles' and 'skills' required in the job description.
    Also add the most relevant ones from the following links to showcase ABC's portfolio: {link_list}
    Remember you are Sadaf, BDE at ABC. 
    Do not provide a preamble.
    ### EMAIL (NO PREAMBLE):
    
    """
    )

    chain_email = prompt_email | llm
    res = chain_email.invoke({"job_description": json_res, "link_list": links})

    email=res.content
    
    st.code(email,language="markdown")