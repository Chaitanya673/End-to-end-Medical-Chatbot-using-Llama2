from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as PineconeLang
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

#print(PINECONE_API_KEY)
#print(PINECONE_API_ENV)


extracted_data = load_pdf("C:/Users/chait/End-to-end-Medical-Chatbot-using-Llama2/Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone

pc = PineconeClient(api_key=PINECONE_API_KEY)

index_name="medical-bot"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=PineconeLang.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
