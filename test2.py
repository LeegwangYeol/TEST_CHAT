from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.memory import ChatMemoryBuffer
import os

# --- 1. 기본 설정: 모델 및 경로 ---
embed_model = OllamaEmbedding(model_name="Qwen3-Embedding-0.6B-f16:latest  ")
llm = Ollama(model="DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M", request_timeout=300.0)

Settings.llm = llm
Settings.embed_model = embed_model

DOCS_PATH = "c:\\source\\CNHRYPRS1019\\data"

# --- 2. 의미 기억 구현 (In-Memory) ---
print("메모리 기반 인덱스를 생성합니다...")

if not os.path.exists(DOCS_PATH):
    print(f"오류: '{DOCS_PATH}' 폴더를 찾을 수 없습니다.")
    exit(1)

try:
    reader = SimpleDirectoryReader(DOCS_PATH)
    documents = reader.load_data()
    
    if not documents:
        print(f"경고: '{DOCS_PATH}' 폴더에 분석 가능한 문서가 없습니다.")
        exit(1)
        
    print(f"총 {len(documents)}개의 문서를 찾았습니다. 인덱스를 생성 중입니다...")
    
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        show_progress=True
    )
    print("[OK] 메모리 인덱스 생성이 완료되었습니다.")
    
except Exception as e:
    print(f"\n[FAIL] 인덱스 생성 중 오류 발생: {e}")
    exit(1)

# --- 3. 문서 청크 설정 ---
from llama_index.core.node_parser import SentenceSplitter

node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=20,
)

# --- 4. 일화 기억 구현 (Episodic Memory) ---
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# --- 5. 채팅 프롬프트 템플릿 설정 ---
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

qa_prompt_template_messages = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are an AI assistant that answers user questions based ONLY on the provided 'context' information.\n"
            "If the answer is not found in the given context, you MUST answer 'I could not find the information in the document.'"
        ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "The context information is as follows.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Based on the context information above, please answer the following question: {query_str}"
        ),
    ),
]
qa_prompt_template = ChatPromptTemplate(message_templates=qa_prompt_template_messages)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    similarity_top_k=1,
    context_prompt=qa_prompt_template
)

# --- 6. 대화 시작 ---
import time

def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}초"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}분 {seconds:.1f}초"

print("\n장기 기억 모델과의 대화를 시작합니다. (종료하려면 'exit' 또는 'quit' 입력)")
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]:
        print("\n대화를 종료합니다.")
        break
    
    print("Assistant: ", end="", flush=True)
    
    start_time = time.time()
    response_stream = chat_engine.stream_chat(user_input)
    
    for token in response_stream.response_gen:
        print(token, end="", flush=True)
    
    end_time = time.time()
    print()
    
    response_time = end_time - start_time
    formatted_time = format_time(response_time)
    
    print(f"[응답 시간: {formatted_time}]")

    print("\n--- DEBUG: Retrieved Source Nodes ---")
    if response_stream.source_nodes:
        for i, node in enumerate(response_stream.source_nodes):
            if node.score is not None:
                print(f"Node {i+1} (Score: {node.score:.4f}):")
            else:
                print(f"Node {i+1} (Score: N/A):")
            print(node.get_content())
            print("-" * 38)
    else:
        print("No source nodes were retrieved for this query.")
    print("--- END DEBUG ---\n")
