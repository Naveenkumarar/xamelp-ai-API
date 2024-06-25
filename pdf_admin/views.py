from django.shortcuts import render
from decouple import config
from rest_framework.response import Response
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import *
debug =  int(config('debug'))

if debug == 1:
    MEDIA_URL = 'media/'
else: 
    MEDIA_URL = '/mediadata/'

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


@csrf_exempt
def pdf_upload(request):
    if request.method == "POST":
        pdf = request.FILES.get('pdf')
        name = request.POST.get("name")
        
        convo = Coversation.objects.create(name=name,pdf=pdf)
        return JsonResponse({"status":True,"data":"succesfully added"})
    
def mcq_format(data):
    datas=[]
    questions = data.split("\n\n")
    index = 0
    print(len(questions))
    while index < len(questions):
        choice = questions[index].split("\n")
        temp={}
        temp["question"] = choice[0]
        temp["options"] = choice[1:-1]
        temp["answer"] = questions[index+1]
        index = index+2
        datas.append(temp)
    return(datas)

#####################LangChain Method#############################################
'''
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


openai_api_key = config('openai_api_key')

import os
os.environ["OPENAI_API_KEY"] = openai_api_key
llm = ChatOpenAI(model="gpt-3.5-turbo")

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key,model="text-embedding-3-small")
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
    
'''


################### OPEN AI Method #######################################################
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai

openai.api_key  = config('openai_api_key')


def get_text_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def embed_pdf_text(pdf_path):
    text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(text)
    embeddings = [get_text_embedding(chunk) for chunk in text_chunks]
    return text_chunks, embeddings

def get_query_embedding(query):
    return get_text_embedding(query)

def find_most_relevant_chunks(query_embedding, text_embeddings, text_chunks, top_k=3):
    similarities = cosine_similarity([query_embedding], text_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant_chunks = [text_chunks[i] for i in top_indices]
    return relevant_chunks

def ask_openai_question(question, context,mem=[]):
    messages = [
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    messages = mem+messages
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        # max_tokens=150
    )
    return response.choices[0].message.content.strip()

def get_answer_from_pdf(pdf_path, question,memory=[]):
    text_chunks, text_embeddings = embed_pdf_text(pdf_path)
    query_embedding = get_query_embedding(question)
    relevant_chunks = find_most_relevant_chunks(query_embedding, text_embeddings, text_chunks)
    context = " ".join(relevant_chunks)
    answer = ask_openai_question(question, context,memory)
    return answer

@csrf_exempt
def ask_question(request):
    if request.method == "GET":
        question = request.GET.get('question')
        name = request.GET.get('name')

        convo_data = list(Coversation.objects.filter(name=name).values())
        if len(convo_data)==0:
            return JsonResponse({"status":False,"data":"No pdf found"})
        pdf_path = MEDIA_URL+convo_data[0]['pdf']
        
        chat = list(Chats.objects.filter(name=convo_data[0]['id']).order_by('timestamp').values())
        if len(chat)==0:
            memory = []
        else:
            memory = []
            index=0
            while index < len(chat):
                memory.append({"role": "user", "content": chat[index]['message']})
                memory.append({"role": "system", "content": chat[index+1]['message']})
                index=index+2
        

        answer = get_answer_from_pdf(pdf_path, question,memory)

        convo = Coversation.objects.get(name=name)
        Chats.objects.create(name=convo,type="Human",message=question)
        Chats.objects.create(name=convo,type="AI",message=answer)
        return JsonResponse({"status":True,"data":answer})
    
@csrf_exempt
def get_mcq(request):
    if request.method == "GET":
        count = request.GET.get('count')
        name = request.GET.get('name')
        question = f"give {count} mcq questions with answer without referring to any image or table"
        
        convo_data = list(Coversation.objects.filter(name=name).values())
        if len(convo_data)==0:
            return JsonResponse({"status":False,"data":"No pdf found"})
        pdf_path = MEDIA_URL+convo_data[0]['pdf']
        
        chat = list(Chats.objects.filter(name=convo_data[0]['id']).order_by('timestamp').values())
        if len(chat)==0:
            memory = []
        else:
            memory = []
            index=0
            while index < len(chat):
                memory.append({"role": "user", "content": chat[index]['message']})
                memory.append({"role": "system", "content": chat[index+1]['message']})
                index=index+2
        

        answer = get_answer_from_pdf(pdf_path, question,memory)

        convo = Coversation.objects.get(name=name)
        Chats.objects.create(name=convo,type="Human",message=question)
        Chats.objects.create(name=convo,type="AI",message=answer)
        out = mcq_format(answer)
        return JsonResponse({"status":True,"data":out})