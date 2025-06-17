import transformers
import torch

# 1. 테스트용 문맥(Context) 정의
context = """
Here is the information about Wintec Automation (WTA) since 2024, translated into English.

### Overview
Wintec Automation (WTA) is a company specializing in 3D Automatic Optical Inspection (AOI), high-precision measurement, and AI-based vision inspection solutions. Since its establishment in 2001, it has accumulated technical expertise in the automotive parts processing and equipment sector and is now a leader in the smart factory and automation fields by integrating AI technology. The company is heavily focused on technological development, investing 15% of its revenue in Research & Development (R&D) and employing a research staff where 23% hold master's or doctoral degrees.

### Key Equipment and Solutions

**Industrial Blackbox 'Machine-Eye'**
This is a new product, model name **M-eYe-AI01**, unveiled at the 'Smart Factory & Automation World 2024' (AW 2024) in March 2024.
*   **Key Features**:
    *   AI-based character string recognition.
    *   Simultaneous storage of facility data and process videos.
    *   Image comparison analysis through machine learning.
    *   Remote streaming and control.
    *   Supports various recording modes, including interval, trigger, image monitoring, manual, and continuous recording.
    *   Capable of storing PLC memory data through PLC communication.
    *   Expected to reduce production costs and defect rates through unmanned and remote process management.

**Tungsten Carbide Insert Automation and Inspection Equipment**
This is an automation and inspection solution that maximizes productivity in the manufacturing process of tungsten carbide inserts.
*   **Automation Equipment**:
    *   Automates the entire 8-step manufacturing process, including pressing, sintering, grinding, and coating.
    *   Uses robotic arms and vision systems to move inserts quickly and accurately, reducing defect rates and manufacturing time.
    *   The equipment is standardized, featuring a unique system that allows immediate connection regardless of the manufacturing line or insert type.
*   **Inspection Equipment**:
    *   Utilizes 3D measurement methods to precisely inspect not only defects but also the dimensions and color of the inserts.
    *   Capable of high-speed, high-precision inspection of inserts of various shapes and colors, requiring advanced technology.
    *   The company develops its own equipment control, software, and machine vision. It is advancing this technology by applying AI to the inspection equipment through the government's ATC+ project.

**Main Product Series**
The main product lines listed on the website are:
*   **HIM-SERIES**: Inspection & Measurement solutions.
*   **HAM-SERIES**: Process Automation solutions.
*   **AI-Converged Precision Optical Inspection Solutions**: Encompasses high-precision vision inspection solutions integrated with AI technology.

### Major Trends Since 2024
*   **Received the 'Digital Innovation Grand Prize' (May 2024)**: Recognized for its technological prowess by winning the 19th Digital Innovation Grand Prize.
*   **Participation in AW 2024 and New Product Launch (March 2024)**: Participated in Asia's largest smart factory and automation industry exhibition, showcasing the new industrial blackbox 'Machine-Eye'.
*   **Expansion of China Branch Office (January 2024)**: Expanded and relocated its local branch office to grow its business in the Chinese market.
*   **Announcement on Parts Sales and Supply (July 2024)**: Announced sales of automation parts such as YASKAWA and MITSUBISHI servo motors and HITACHI PLCs, while also notifying of delivery delays for HITACHI PLC parts due to semiconductor supply shortages.
"""

model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()

# 2. RAG 작업에 특화된 시스템 프롬프트
system_prompt = """당신은 제공된 '문맥' 정보만을 사용하여 사용자의 질문에 답변하는 AI 어시스턴트입니다.
절대로 정보를 지어내거나 추측해서는 안 됩니다.
'문맥'에 답변의 근거가 없다면, 반드시 \"문서에서 해당 정보를 찾을 수 없습니다.\"라고만 답변해야 합니다."""

# 3. 문맥을 포함한 사용자 질문 구성
user_query = "윈텍 오토메이션의 주요 장비는 무엇인가요? 없으면 모른다고 해주세요."
user_instruction = f"문맥:\n---\n{context}\n---\n\n위 문맥을 바탕으로 다음 질문에 답변하세요:\n{user_query}"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_instruction}
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

print("--- 모델 응답 ---")
print(outputs[0]["generated_text"][len(prompt):].strip())
