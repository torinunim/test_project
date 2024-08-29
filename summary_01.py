import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List

# 환경변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI()

def summarize_text(text: str, max_words: int = 100) -> str:
    instruction_message = f"""
    As an expert copy-writer, you will write a concise summary of the user provided content. The summary should be under {max_words} words.
    
    # Your Summarization Process
    - Read through the content and all the below sections to get an understanding of the task.
    - Write a new, denser summary of identical length which covers every detail from the previous summary plus any additional important information from the original content.
    
    # Guidelines
    - The summary should be dense and concise yet self-contained, e.g., easily understood without the original content.
    - Make every word count: use clear and precise language.
    - Focus on the most important information and key points from the original content.
    - Do not discuss the content itself, focus on the content's information and details.
    
    # IMPORTANT
    - Remember to keep the summary to max {max_words} words.
    - Do not discuss the content itself, focus on the content's information and details.
    - The summary should be written in the same language as the original content.

    """

    messages = [
        {"role": "system", "content": instruction_message},
        {"role": "user", "content": f"다음 텍스트를 요약해주세요:\n\n{text}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "요약 중 오류가 발생했습니다."

# 스트림릿 앱 설정
st.title("텍스트 요약기")

# 사용자 입력 필드
user_input = st.text_area("여기에 요약할 텍스트를 입력하세요:", height=300)

if st.button("요약하기"):
    if user_input:
        # 텍스트 요약 실행
        summary = summarize_text(user_input, max_words=100)
        st.write("요약 결과:")
        st.write(summary)
    else:
        st.write("텍스트를 입력해주세요.")