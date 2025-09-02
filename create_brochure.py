import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI

load_dotenv(override=True)

MODEL = "gpt-4o-mini"
# MODEL = "llama3.2"
if MODEL.startswith("gpt-"):
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith("sk-proj-") and len(api_key) > 20:
        print("API key looks good so far")
    else:
        print("There might be an issue with your API key")
    openai = OpenAI()
else:
    openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "Untitled"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]
    
    def get_content(self):
        return f"Webpage title:\n{self.title}\nWebpage content:\n{self.text}"

def get_link_prompts(website):
    system_prompt = """You are provided with a list of links found on webpage.
    You are able to decide which of the links would be most relevant to include in a brochure about the company,
    such as links to an About page, or a Company page, or Careeres/Job pages.\n
    You should respond in JSON as in this example
    {
        "links": [
            {"page": "about", "url": "https://full_url.com/goes/here/about"}
            {"page": "careers", "url": "https://full_url.com/careers"}
        ]
    }"""
    user_prompt = """Here is the list of links on the website of {url}, please decide which of these are relevant web links for a brochure about the company, respond with full https URL in JSON format.
    Do not include Terms of Service, Privacy, email links.
    Links:\n"""
    user_prompt += "\n".join(website.links)
    return (system_prompt, user_prompt)

def get_links(url):
    website = Website(url)
    system_prompt, user_prompt = get_link_prompts(website)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

def get_details(url):
    result = "Landing page:\n"
    result += Website(url).get_content()
    links = get_links(url)
    for link in links["links"]:
        result += "\n\nPage " + link["page"] + ":\n"
        result += Website(link["url"]).get_content()
    return result

def get_brochure_prompts(company_name, url):
    system_prompt = """You are an assistant that analyzes the contents of several pages from company website\
and create a short brochure about the company for prospective customers, investors and recurits. Respond in Markdown.\n
Include details of company culture, customers and careers/jobs if you have the information."""
    user_prompt = """You are looking at a compnany called: {company_name}
    Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in Markdown."""
    user_prompt += "\n" + get_details(url)
    return (system_prompt, user_prompt)

def create_brochure(company_name, url):
    system_prompt, user_prompt = get_brochure_prompts(company_name, url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": user_prompt}
        ]
    )
    result = response.choices[0].message.content
    return result

if __name__ == "__main__":
    print(create_brochure("Anthropic", "https://anthropic.com/"))