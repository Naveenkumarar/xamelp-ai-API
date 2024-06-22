from django.shortcuts import render
from decouple import config
from rest_framework.response import Response
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from .models import *

openai_api_key = config('openai_api_key')

import os
os.environ["OPENAI_API_KEY"] = openai_api_key
llm = ChatOpenAI(model="gpt-4")

def get_pdf_text(pdf_doc): 
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

conversation={}
@csrf_exempt
def pdf_upload(request):
    global conversation
    if request.method == "POST":
        pdf = request.FILES.get('pdf')
        name = request.POST.get("name")

        raw_text = get_pdf_text(pdf)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        conversation[name] = get_conversation_chain(vectorstore)
        
        # convo = Coversation.objects.create(name=name,chain=conversation)
        # print(convo)
        return JsonResponse({"status":True,"data":len(text_chunks)})
    
from sys import getsizeof

@csrf_exempt
def ask_question(request):
    global conversation
    if conversation==None:
        return JsonResponse({"status":False,"data":"Upload PDF"})
    if request.method == "GET":
        question = request.GET.get('question')
        name = request.GET.get('name')

        res= conversation[name]({'question': question})
        out=[]
        for i in res['chat_history']:
            out.append(i.content)
        return JsonResponse({"status":True,"data":{"answer":out}})