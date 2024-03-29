from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone as PineconeClient
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone

pc = PineconeClient(api_key=PINECONE_API_KEY)

index_name="medical-bot"

#Loading the index
docsearch=Pinecone.from_existing_index(index_name, embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])


llm=CTransformers(model="C:/Users/chait/End-to-end-Medical-Chatbot-using-Llama2/Model/llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",
                  config={'max_new_tokens':512,'temperature':0.8})

retriever=docsearch.as_retriever(search_kwargs={'k': 2})

qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,
                                 return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)