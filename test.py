# 1. 필요한 라이브러리 설치
# pip install llama-index-llms-ollama llama-index-embeddings-ollama llama-index

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# 2. 분석할 문서들을 'data' 폴더에 넣어둡니다.
documents = SimpleDirectoryReader("data").load_data()

# 3. Ollama 설정 (모델 및 임베딩)
ollama_embedding = OllamaEmbedding(model_name="deepseek-r1:1.5b") # 문서 임베딩용 모델
llm = Ollama(model="deepseek-r1:1.5b", request_timeout=120.0) # 답변 생성용 모델

# 4. 문서 -> 벡터 변환 및 인덱스 생성 (오픈북 시험을 위한 '색인' 만들기)
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=ollama_embedding,
)

print("1. 문서 로드 완료")
print(f"- 문서 수: {len(documents)}")

# 5. 쿼리 엔진 생성 및 질문하기
print("\n2. 쿼리 엔진 생성 중...")
query_engine = index.as_query_engine(llm=llm)

print("\n3. 질문을 전송 중...")
query = "what is wintecautomation's main product?"
print(f"- 질문: {query}")

response = query_engine.query(query)
print("\n4. 응답:")
print(response)
