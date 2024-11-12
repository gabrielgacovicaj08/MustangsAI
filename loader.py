# file containing the functions to load and embed the text
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import  OpenAIEmbeddings
from dotenv import load_dotenv
import os
import bs4
import re

# Load environment variables from the .env file
load_dotenv(dotenv_path=r"C:\Users\gabri\Desktop\Rag_Project\.env.txt")

# Now you can access your API keys like this
openai_api_key = os.getenv("OPENAI_API_KEY")



'''
    This function recieve a list of key-value web pages ({'url':'...','class_name':'...'}), 
    and extract the text with the specific tag or classes ids.

    It returns a list a list of documents divede in meta-data and page content 
    for each URL
'''
def load_text(documents):
    # variable in which i store all the extracted text
    all_docs = []

    '''
        this loop will go through each document in the list and 
        extract the text with the specific tags and class name if existing
    '''
    for links in documents:
        url = links["url"]
        class_name = links["class_name"]
        
        # the tags which contain the most important part of the text 
        tags_to_target = ["div", "p", "h1", "h2", "h3", "td"]
        
        if class_name:
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs=dict(parse_only=bs4.SoupStrainer(tags_to_target, class_=class_name)),
            )
        else:
            # If no class names are provided, just target the tags
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs=dict(parse_only=bs4.SoupStrainer(tags_to_target)),
            )
        docs = loader.load()
        all_docs.extend(docs)

    return all_docs

'''
    this function recieves a list of documents and split
    the page_content sections in chunk of the size of 400 chars
    with a overlap of a 100 chars

    Paramaters:
    - docs: a list of documents divided in meta-data and page_content

    Returns:
    - a list of splitted documents
'''
def text_splitter(docs):
    text_splitter = text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    return splits


"""
    Load an existing Chroma vector store.

    Parameters:
    - collection_name: The name of the Chroma collection.
    - openai_api_key: The OpenAI API key for generating embeddings.
    - persist_directory: The directory where the vector store is persisted.

    Returns:
    - The loaded Chroma vector store instance.
 """
def load_existing_vector_store(collection_name, openai_api_key, persist_directory="./chroma_store2"):
    
    embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = Chroma(
        embedding_function=embeddings_model,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    return vector_store

if __name__ == "__main__":

    
    documents = [{"url":"https://msutexas.edu/about/", "class_name":["uk-width-medium-1-2"]},
                    {"url":"https://msutexas.edu/housing/meal-plan.php", "class_name":["uk-width-medium-1-1"]},
                   {"url":"https://msutexas.edu/academics/index.php","class_name":[]},
                   {"url":"https://catalog.msutexas.edu/content.php?catoid=41&navoid=2289", "class_name": ['block_content']},
                    {"url":"https://catalog.msutexas.edu/content.php?catoid=41&navoid=2311", "class_name": ['block_content']},
                    {"url":"https://catalog.msutexas.edu/preview_program.php?catoid=41&poid=5679", "class_name": ['block_content']},
                    {"url":"https://catalog.msutexas.edu/content.php?catoid=41&navoid=2254", "class_name": ['block_content']},
                    {"url":"https://catalog.msutexas.edu/content.php?catoid=41&navoid=2312", "class_name": ['block_content']},
                    {"url":"https://catalog.msutexas.edu/content.php?catoid=41&navoid=2282", "class_name": ['block_content']},
                    {"url":"https://catalog.msutexas.edu/content.php?catoid=41&navoid=2287", "class_name": ['block_content']},
                    {"url":"https://catalog.msutexas.edu/content.php?catoid=41&navoid=2316", "class_name": ['block_content']},
                    {"url":"https://catalog.msutexas.edu/content.php?catoid=41&navoid=2256", "class_name": ['block_content']},
                    {"url":"https://catalog.msutexas.edu/preview_program.php?catoid=41&poid=5679", "class_name": ['block_content']},
                    {"url":"https://catalog.msutexas.edu/preview_program.php?catoid=41&poid=5679", "class_name": ['block_content']},
                    {"url":"https://catalog.msutexas.edu/preview_program.php?catoid=41&poid=5679", "class_name": ['block_content']},
                    
                ]

    all_docs = load_text(documents)
    if not all_docs:
            raise ValueError("No documents were loaded. Check if the URLs and class names are correct.")
    
    for text in all_docs:
        # Replace hyphens with a blank space
        text.page_content = text.page_content.replace('-', ' ')

        # Remove unwanted characters
        unwanted_chars = ['•',"|"]  # Add other characters you want to remove
        for char in unwanted_chars:
            text.page_content = text.page_content.replace(char, '')

        phrases_to_remove = [
            "Print-Friendly Page (opens a new window)",
            "Facebook this Page (opens a new window)",
            "Tweet this Page (opens a new window)",
            "HELP",
            "Print Degree Planner (opens a new window)",
            "Print Friendly Page (opens a new window)"
        ]

        for phrase in phrases_to_remove:
            text.page_content = text.page_content.replace(phrase, '')

        # Replace multiple spaces with a single space
        text.page_content = re.sub(r'\s+', ' ', text.page_content)

        # Remove leading and trailing spaces if necessary
        text.page_content = text.page_content.strip()
    
    
    splits = text_splitter(all_docs)
    if not splits:
         raise ValueError("Check the format of the document")
    
    
    try:
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=OpenAIEmbeddings(api_key=openai_api_key),
                persist_directory='./chroma_store2',
                collection_name = "msu_collection"
            )
    except Exception as e:
            raise ValueError(f"Error setting up the vector store: {e}")

