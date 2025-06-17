import numpy as np
from llama_index.embeddings.ollama import OllamaEmbedding
import asyncio
import platform

# 코사인 유사도 계산 함수
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # 0으로 나누는 것을 방지
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

async def main():
    print("--- 임베딩 모델 직접 테스트 시작 ---")
    
    # 1. 테스트할 텍스트 정의 (data/test.txt에서 가져옴)
    query_text = "What is M-eYe-AI01?"
    # Markdown 문법(**)은 제거하여 순수 텍스트로 테스트
    document_text = "This is a new product, model name M-eYe-AI01, unveiled at the 'Smart Factory & Automation World 2024' (AW 2024) in March 2024."
    
    print(f"\n쿼리 텍스트: \"{query_text}\"")
    print(f"문서 텍스트: \"{document_text}\"")
    
    # 2. 임베딩 모델 초기화
    try:
        embed_model = OllamaEmbedding(model_name="nomic-embed-text-v1.5")
        print("\n[OK] 임베딩 모델 초기화 완료.")
    except Exception as e:
        print(f"\n[FAIL] 임베딩 모델 초기화 실패: {e}")
        return

    # 3. 각 텍스트에 대한 임베딩 생성
    print("\n임베딩 생성 중...")
    try:
        # 비동기 함수를 사용하여 임베딩 생성
        query_embedding = await embed_model.aget_query_embedding(query_text)
        doc_embedding = await embed_model.aget_text_embedding(document_text)
        print("[OK] 임베딩 생성 완료")
    except Exception as e:
        print(f"\n[FAIL] 임베딩 생성 실패: {e}")
        return

    # 4. 코사인 유사도 계산
    similarity = cosine_similarity(query_embedding, doc_embedding)
    
    # 5. 결과 출력
    print("\n--- 테스트 결과 ---")
    print(f"계산된 코사인 유사도 점수: {similarity:.4f}")
    
    if similarity > 0.7:
        print("\n판정: [성공] 임베딩 모델이 두 텍스트의 의미적 유사성을 올바르게 인식했습니다.")
        print("문제의 원인은 llama-index 또는 ChromaDB와의 연동 부분에 있을 가능성이 높습니다.")
    else:
        print("\n판정: [실패] 임베딩 모델이 의미적으로 유사한 텍스트를 구별하지 못하고 있습니다.")
        print("Ollama에서 사용하는 'nomic-embed-text-v1.5' 모델 자체에 문제가 있거나, Ollama 서버와의 통신에 문제가 있을 수 있습니다.")

if __name__ == "__main__":
    # numpy가 설치되어 있는지 확인
    try:
        import numpy
    except ImportError:
        print("오류: 'numpy' 라이브러리가 설치되지 않았습니다.")
        print("테스트를 실행하기 전에 'pip install numpy' 명령어로 설치해주세요.")
        exit()
        
    # Windows에서 asyncio를 원활하게 실행하기 위한 정책 설정
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
