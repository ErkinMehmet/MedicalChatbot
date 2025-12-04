from flask import Flask,render_template,jsonify,request
from src.helper import download_hugging_face_embeddings,extract_data_pdf,text_split
import pinecone,os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import CTransformers
from dotenv import load_dotenv
from src.prompt import *
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app=Flask(__name__)
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

embeddings=download_hugging_face_embeddings()
# pinecone.init(api_key=PINECONE_API_KEY,environment=PINECONE_API_ENV) obsolete version
index_name="medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)       
PROMPT=PromptTemplate(template=prompt_template,input_variables=["context","question"])
chain_type_kwargs={"prompt": PROMPT}
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",config={
    'max_new_tokens': 512,'temperature': 0.8
})
retriever = docsearch.as_retriever(search_kwargs={"k": 2})
qa=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,
                               return_source_documents=True,chain_type_kwargs=chain_type_kwargs)
useOpenAI=False
if useOpenAI:
    llm=ChatOpenAI(model="gpt-4o")
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system",openai_template),
            ("human","{input}")]
    )
    qa=create_stuff_documents_chain(llm,prompt)
    rag_chain=create_retrieval_chain(retriever,qa)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/get',methods=["GET", "POST"])
def chat():
    user_input= request.form["msg"]#request.json.get('msg')
    print(user_input)
    if useOpenAI:
        response=rag_chain.run({"input":user_input})
    else:
        response=qa.invoke({"query":user_input})
    result=response["result"]
    print(result)
    return jsonify({'response':str(result)})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)