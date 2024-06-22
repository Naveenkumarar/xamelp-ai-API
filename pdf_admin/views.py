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

def get_conversation_chain(vectorstore,mem = None):
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    if mem:
        retrieved_chat_history = ChatMessageHistory(
        messages=mem.chat_memory.messages
    )
        memory = ConversationBufferMemory(
                chat_memory=retrieved_chat_history,memory_key='chat_history', return_messages=True)

    else:
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
    return conversation_chain

@csrf_exempt
def pdf_upload(request):
    if request.method == "POST":
        pdf = request.FILES.get('pdf')
        name = request.POST.get("name")
        
        convo = Coversation.objects.create(name=name,pdf=pdf)
        return JsonResponse({"status":True,"data":len(convo)})
    
from sys import getsizeof

@csrf_exempt
def ask_question(request):
    if request.method == "GET":
        question = request.GET.get('question')
        name = request.GET.get('name')

        convo_data = list(Coversation.objects.filter(name=name).values())
        if len(convo_data)==0:
            return JsonResponse({"status":False,"data":"No pdf found"})
        raw_text = get_pdf_text('media/'+convo_data[0]['pdf'])
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)

        chat = list(Chats.objects.filter(name=convo_data[0]['id']).order_by('timestamp').values())
        if len(chat)==0:
            conversation = get_conversation_chain(vectorstore)
        else:
            memory = ConversationBufferMemory()
            index=0
            while index < len(chat):
                inputs = {"input":chat[index]['message']}
                output = {"output":chat[index+1]['message']}
                memory.save_context(inputs,output)
                index=index+2
            conversation = get_conversation_chain(vectorstore,memory)
        
        res= conversation({'question': question})
        out=[]
        for i in res['chat_history']:
            out.append(i.content)
        
        convo = Coversation.objects.get(name=name)
        Chats.objects.create(name=convo,type="Human",message=out[-2])
        Chats.objects.create(name=convo,type="AI",message=out[-1])
        return JsonResponse({"status":True,"data":out[-1]})
    

@csrf_exempt
def get_mcq(request):
    if request.method == "GET":
        count = request.GET.get('count',5)
        name = request.GET.get('name')

        convo_data = list(Coversation.objects.filter(name=name).values())
        if len(convo_data)==0:
            return JsonResponse({"status":False,"data":"No pdf found"})
        raw_text = get_pdf_text('media/'+convo_data[0]['pdf'])
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)

        chat = list(Chats.objects.filter(name=convo_data[0]['id']).order_by('timestamp').values())
        if len(chat)==0:
            conversation = get_conversation_chain(vectorstore)
        else:
            memory = ConversationBufferMemory()
            index=0
            while index < len(chat):
                inputs = {"input":chat[index]['message']}
                output = {"output":chat[index+1]['message']}
                memory.save_context(inputs,output)
                index=index+2
            conversation = get_conversation_chain(vectorstore,memory)
        
        res= conversation({'question': f"give {count} mcq questions with answer"})
        out=[]
        for i in res['chat_history']:
            out.append(i.content)
        
        convo = Coversation.objects.get(name=name)
        Chats.objects.create(name=convo,type="Human",message=out[-2])
        Chats.objects.create(name=convo,type="AI",message=out[-1])
        mcq_format(out[-1])
        return JsonResponse({"status":True,"data":out[-1]})
    
def mcq_format(data):
    datas = data.split("\n")
    print(datas)