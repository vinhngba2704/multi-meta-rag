import ast
import json
import re
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Your constant file
QUERY_FILTERS_FILE = "query_metadata_filters_copy.json"

# Extraction Prompt Template
EXTRACT_FILTER_TEMPLATE = """Some questions will be provided below. 
Given the question, extract the metadata to filter the database about article sources. Avoid stopwords.
-----------------------------------------------------------------------------
The sources can only be from the list: 
['Yardbarker', 'The Guardian', 'Revyuh Media', 'The Independent - Sports', 'Wired', 'Sport Grill', 
'Hacker News', 'Iot Business News', 'Insidesport', 'Sporting News', 'Seeking Alpha', 
'The Age', 'CBSSports.com', 'The Sydney Morning Herald', 'FOX News - Health', 
'Science News For Students', 'Polygon', 'The Independent - Life and Style', 'FOX News - Entertainment', 
'The Verge', 'Business Line', 'The New York Times', 'The Roar | Sports Writers Blog', 'Sportskeeda', 
'BBC News - Entertainment & Arts', 'Business World', 'BBC News - Technology', 'Essentially Sports', 
'Mashable', 'Advanced Science News', 'TechCrunch', 'Financial Times', 'Music Business Worldwide', 
'The Independent - Travel', 'FOX News - Lifestyle', 'TalkSport', 'Yahoo News', 
'Scitechdaily | Science Space And Technology News 2017', 'Globes English | Israel Business Arena', 
'Wide World Of Sports', 'Rivals', 'Fortune', 'Zee Business', 'Business Today | Latest Stock Market And Economy News India', 
'Sky Sports', 'Cnbc | World Business News Leader', 'Eos: Earth And Space Science News', 
'Live Science: The Most Interesting Articles', 'Engadget']
-----------------------------------------------------------------------------
Examples to follow:

Question: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?
Answer: {{'source': {{'$in': ['The Verge', 'TechCrunch']}}}}

Question: After the TechCrunch report on October 7, 2023, concerning Dave Clark's comments on Flexport, and the subsequent TechCrunch article on October 30, 2023, regarding Ryan Petersen's actions at Flexport, was there a change in the nature of the events reported?
Answer: {{'source': {{'$in': ['TechCrunch']}}, 'published_at': {{'$in': ['October 7, 2023', 'October 30, 2023']}}}}

Question: Which company, known for its dominance in the e-reader space and for offering exclusive invite-only deals during sales events, faced a stock decline due to an antitrust lawsuit reported by 'The Sydney Morning Herald' and discussed by sellers in a 'Cnbc | World Business News Leader' article?
Answer: {{'source': {{'$in': ['The Sydney Morning Herald', 'Cnbc | World Business News Leader']}}}}
-----------------------------------------------------------------------------
If you detect multiple queries, return the answer for the first. Now it is your turn:

Question: {query}
Answer:
"""

# Clean filter function
def clean_filter(filter_dict: dict) -> dict:
    for filter_key in list(filter_dict.keys()):
        if filter_key not in ["source", "published_at"]:
            del filter_dict[filter_key]

    if "published_at" in filter_dict:
        if isinstance(filter_dict["published_at"], dict):
            for key in list(filter_dict["published_at"].keys()):
                dates = filter_dict["published_at"][key]
                if isinstance(dates, list):
                    valid_dates = []
                    for date in dates:
                        try:
                            datetime.strptime(date, "%B %d, %Y")
                            valid_dates.append(date)
                        except (ValueError, TypeError):
                            pass
                    if valid_dates:
                        filter_dict["published_at"][key] = valid_dates
                    else:
                        del filter_dict["published_at"]
                else:
                    del filter_dict["published_at"]
        else:
            del filter_dict["published_at"]
    return filter_dict

# Initialize Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Extract and parse output more safely
def parse_llm_output(llm_output: str):
    # Extract content inside first outermost {...}
    match = re.search(r'(\{.*\})', llm_output, re.DOTALL)
    if not match:
        raise ValueError("Failed to extract dictionary from LLM output")

    dict_str = match.group(1)
    return ast.literal_eval(dict_str)

# Call Gemini with simple retry logic
def call_gemini_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.1}
            )
            if response.text:
                return response.text
            else:
                raise ValueError("Empty response from Gemini")
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    raise RuntimeError("Failed to get valid response after retries")

# The main function
def process_and_append_query(new_query: str):
    filename = Path(QUERY_FILTERS_FILE)
    if filename.exists():
        with open(filename, "r") as f:
            query_filters = json.load(f)
    else:
        query_filters = []

    if any(qf['query'] == new_query for qf in query_filters):
        print("Query already exists in file. Skipping.")
        return

    full_prompt = EXTRACT_FILTER_TEMPLATE.format(query=new_query)

    try:
        llm_output = call_gemini_with_retry(full_prompt)
        filter_dict = parse_llm_output(llm_output)
        filter_dict = clean_filter(filter_dict)
        print("Extracted metadata:", filter_dict)
    except Exception as e:
        print("Failed to parse LLM output:", e)
        print("Raw output:", llm_output)
        return

    query_filters.append({"query": new_query, "filter": filter_dict})
    with open(filename, "w") as f:
        json.dump(query_filters, f, indent=4, sort_keys=True)
    print("Successfully appended to file.")

# === Simple interface ===
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    process_and_append_query(user_query)
