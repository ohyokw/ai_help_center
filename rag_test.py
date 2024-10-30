import os
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import torch
print(torch.cuda.is_available())  # True가 출력되면 CUDA가 정상적으로 설치된 것입니다.

# PDF 로드
loader = PyMuPDFLoader("laborlaw.pdf")
pages = loader.load()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(pages)

# 2. 임베딩 변환
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    #model_kwargs={'device':'cpu'},
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)

# 3. 벡터 저장소 경로 설정
## 현재 경로에 'vectorstore' 경로 생성
vectorstore_path = 'vectorstore'
os.makedirs(vectorstore_path, exist_ok=True)
# 벡터 저장소 생성 및 저장
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=vectorstore_path)
# 벡터스토어 데이터를 디스크에 저장
# vectorstore.persist() Chroma 0.4.x 버전부터는 수동으로 persist()를 호출할 필요가 없으며, 생성된 문서가 자동으로 저장됩니다. 따라서 vectorstore.persist() 줄은 제거해도 무방합니다:
# print("Vectorstore created and persisted")

# 3. RAG 모델 및 체인 구성
# Ollama 를 이용해 로컬에서 LLM 실행
model = ChatOllama(model="granite3-dense:8b", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# Prompt 템플릿 생성
template = '''친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘.":
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

# RAG Chain 연결
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Chain 실행
query = "연장근로수당에 대해 알려 줘"
answer = rag_chain.invoke(query)

print("Query:", query)
print("Answer:", answer)