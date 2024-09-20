import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import chardet
import tiktoken
from PyPDF2 import PdfReader
from anthropic import Anthropic
import io
import pyperclip
import time

# 환경변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI()

anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')


def auto_decode(file_content):
    # PDF 파일 처리
    if file_content.startswith(b'%PDF'):
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    # 다른 형식의 파일 처리 (기존 코드)
    result = chardet.detect(file_content)
    detected_encoding = result['encoding']
    
    if not detected_encoding:
        detected_encoding = 'utf-8'
    
    common_encodings = ['utf-8', 'euc-kr', 'cp949', 'iso-8859-1']
    
    for encoding in [detected_encoding] + common_encodings:
        try:
            return file_content.decode(encoding)
        except UnicodeDecodeError:
            continue
    
    raise UnicodeDecodeError("Failed to decode with any encoding")

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def split_text(text: str, max_tokens: int = 100000) -> list:
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in text.split('. '):
        sentence_tokens = num_tokens_from_string(sentence)
        
        if current_size + sentence_tokens > max_tokens:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_size = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_size += sentence_tokens
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

def summarize_text(text: str, max_words: int, summary_type: str) -> str:
    token_count = num_tokens_from_string(text)
    if token_count > 100000:
        print(f"입력 텍스트가 너무 깁니다. 토큰 수: {token_count}")
        return "텍스트가 너무 길어 요약할 수 없습니다."

    if summary_type == "summary":
        instruction_message = f"""
        [Instructions]
        As an expert summarizer in the field of {text_category}, please provide a concise summary of the content provided by the user.
        
        [Context]
        Consider the following four aspects:

        1. Use of technical terminology
        2. Accurate wording regarding regulations and legal requirements
        3. Customer perspective
        4. Characteristics of products and services

        [Key considerations]
        1. Summarize concisely in {max_sentences} sentences.
        2. Use polite, formal language (equivalent to the Korean "-습니다" style).
        3. Summarize only the content present in the document.
        4. Ensure that the core content of the original text is preserved.
        5. Do not include information not explicitly mentioned in the source material.

        [Important]
        The final output must be in Korean.
        """
    elif summary_type == "document summary":
        instruction_message = f"""
        [Instructions]
        As a document summarization expert, please provide a comprehensive summary of the main content of the given document in the field of {text_category}.
        
        [Context]
        The summary should be appropriately crafted based on the number of pages and the amount of information in the document, and must include the document's structure, key arguments, and important details.
        Strictly adhere to the following four aspects:

        1. Use of technical terminology
        2. Accurate wording regarding regulations and legal requirements
        3. Customer perspective
        4. Characteristics of products and services

        [Key considerations]
        1. Summarize concisely in {max_sentences_doc} sentences.
        2. Use polite, formal language (equivalent to the Korean "-습니다" style).
        3. Summarize only the content present in the document.
        4. Ensure that the core content of the original text is preserved.
        5. Do not include information not explicitly mentioned in the source material.

        [Important]
        The final output must be in Korean.
        """
    elif summary_type == "bullet point summary":
        instruction_message = f"""
        [Instructions]
        As a specialist in structuring information for {text_category}, please summarize the key points of the given content in //bullet point// format.
        
        [Context]
        Strictly adhere to the following four aspects:

        1. Use of technical terminology
        2. Accurate wording regarding regulations and legal requirements
        3. Customer perspective
        4. Characteristics of products and services

        [Key considerations]
        1. Create a total of {num_points} bullet points.
        2. Each point should be concise and clear, encompassing the main ideas of the entire content.
        3. Start each bullet point with '-', and choose an appropriate number of points to summarize based on the amount of text.
        4. End each bullet point with a nominal ending.

        [Important]
        The final output must be in Korean.
        """
    else:
        instruction_message = f"""
        As a professional summarizer, please summarize the given content within {max_words} words.
The summary should accurately convey the essence of the original text.
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
        error_message = f"요약 중 오류가 발생했습니다: {str(e)}"
        print(error_message)
        return error_message
    
def summarize_long_text(text: str, max_words: int, summary_type: str) -> str:
    chunks = split_text(text)
    summaries = []
    
    for chunk in chunks:
        summary = summarize_text(chunk, max_words // len(chunks), summary_type)
        summaries.append(summary)
    
    final_summary = " ".join(summaries)
    
    if num_tokens_from_string(final_summary) > 100000:
        final_summary = summarize_text(final_summary, max_words, summary_type)
    
    return final_summary

def review_summary(original_text: str, summary: str, category: str) -> str:
    instruction_message = f"""
    [Instructions]
    {category} 분야의 전문 검수자로서, 원본 문서와 요약본을 비교 분석하여 요약의 품질을 평가해주세요.

    [Key considerations]
    1. 요약의 정확성: 원본 내용이 왜곡되지 않고 정확하게 전달되었는지 확인
    2. 핵심 정보 포함 여부: 원본의 주요 내용이 요약에 모두 포함되었는지 검토
    3. 간결성: 불필요한 정보나 중복된 내용 없이 간결하게 작성되었는지 평가
    4. 전문성: {category} 분야의 전문 용어와 개념이 적절히 사용되었는지 확인
    5. 개선점 제시: 요약의 품질을 높일 수 있는 구체적인 개선 방안 제안

    [Output Format]
    1. 전반적인 평가 (100점 만점)
    2. 장점
    3. 개선이 필요한 부분
    4. 구체적인 개선 제안
    """

    try:
        client = Anthropic(api_key=anthropic_api_key)
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.7,
            system=instruction_message,
            messages=[
                {"role": "user", "content": f"원본 문서:\n\n{original_text}\n\n요약본:\n\n{summary}"}
            ]
        )
        return response.content[0].text
    except Exception as e:
        error_message = f"검수 중 오류가 발생했습니다: {str(e)}"
        print(error_message)
        return error_message
    

page = st.sidebar.radio("기능 선택", ['텍스트(문장, 문단) 요약', '문서 요약', 'bullet point 요약', '검수'])

# # 디버그 모드 추가
# debug_mode = st.sidebar.checkbox("디버그 모드")

if page == '텍스트(문장, 문단) 요약':
    st.title("텍스트 요약기")
    text_category = st.sidebar.selectbox(
    "텍스트의 카테고리",
    ("보험", "은행", "카드", "증권"), index=0)
    user_input = st.text_area("여기에 요약할 텍스트를 입력하세요:", height=300)
    input_tokens = num_tokens_from_string(user_input)
    st.write(f"입력 텍스트 길이: {input_tokens} 토큰")
    max_sentences = st.sidebar.number_input("요약할 최대 문장 수 설정", min_value=1, max_value=10, value=3)
    st.sidebar.write("1~10문장 설정 가능")
    if st.button("요약하기"):
        if user_input:
            with st.spinner('요약 중...'):
                summary = summarize_long_text(user_input, max_sentences, "summary")
            st.write("요약 결과:")
            st.write(summary)
            summary_tokens = num_tokens_from_string(summary)
            st.write(f"요약 길이: {summary_tokens} 토큰")
            if st.button("결과 복사"):
                pyperclip.copy(summary)
                st.success("클립보드에 복사되었습니다!")
        else:
            st.write("텍스트를 입력해주세요.")

elif page == '문서 요약':
    st.title("문서 요약")
    text_category = st.sidebar.selectbox(
    "텍스트의 카테고리",
    ("보험", "은행", "카드", "증권"), index=0)
    doc_file = st.file_uploader("문서를 업로드하세요", type=['pdf', 'docx', 'txt'])
    max_sentences_doc = st.sidebar.number_input("문서 요약의 최대 문장 수 설정", min_value=1, max_value=20, value=5)
    st.sidebar.write("1~20문장 설정 가능")
    if st.button("문서 요약하기"):
        if doc_file is not None:
            try:
                with st.spinner('문서 요약 중...'):
                    file_content = doc_file.read()
                    text_content = auto_decode(file_content)
                    if debug_mode:
                        st.write("디코딩된 텍스트 (처음 500자):")
                        st.write(text_content[:500])
                        st.write(f"전체 텍스트 토큰 수: {num_tokens_from_string(text_content)}")
                    summary = summarize_long_text(text_content, max_sentences_doc, "document summary")
                st.write("문서 요약 결과:")
                st.write(summary)
                summary_tokens = num_tokens_from_string(summary)
                st.write(f"요약 길이: {summary_tokens} 토큰")
                if st.button("결과 복사"):
                    pyperclip.copy(summary)
                    st.success("클립보드에 복사되었습니다!")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                if debug_mode:
                    st.exception(e)
        else:
            st.write("문서 파일을 업로드 해주세요.")

elif page == 'bullet point 요약':
    st.title("Bullet Point 요약")
    text_category = st.sidebar.selectbox(
    "텍스트의 카테고리",
    ("보험", "은행", "카드", "증권"), index=0)
    user_input_bullet = st.text_area("여기에 요약할 텍스트를 입력하세요:", height=300)
    input_tokens = num_tokens_from_string(user_input_bullet)
    st.write(f"입력 텍스트 길이: {input_tokens} 토큰")
    num_points = st.sidebar.number_input("생성할 bullet point 수 설정", min_value=3, max_value=10, value=5)
    st.sidebar.write("3~10문장 설정 가능")
    if st.button("Bullet Point 요약하기"):
        if user_input_bullet:
            with st.spinner('Bullet Point 요약 중...'):
                summary = summarize_long_text(user_input_bullet, num_points, "bullet point summary")
            st.write("Bullet Point 요약 결과:")
            st.write(summary)
            summary_tokens = num_tokens_from_string(summary)
            st.write(f"요약 길이: {summary_tokens} 토큰")
            if st.button("결과 복사"):
                pyperclip.copy(summary)
                st.success("클립보드에 복사되었습니다!")
        else:
            st.write("텍스트를 입력해주세요.")
            
# 검수 기능
elif page == '검수':
    st.title("요약 검수")
    text_category = st.sidebar.selectbox(
    "텍스트의 카테고리",
    ("보험", "은행", "카드", "증권"), index=0)
    
    original_doc = st.file_uploader("원본 문서를 업로드하세요 (PDF)", type=['pdf'])
    summary_text = st.text_area("요약된 내용을 입력하세요:", height=200)
    
    if st.button("검수하기"):
        if original_doc is not None and summary_text:
            try:
                with st.spinner('검수 중...'):
                    file_content = original_doc.read()
                    original_text = auto_decode(file_content)
                    
                    if debug_mode:
                        st.write("원본 문서 내용 (처음 500자):")
                        st.write(original_text[:500])
                    
                    review_result = review_summary(original_text, summary_text, text_category)
                st.write("검수 결과:")
                st.write(review_result)
                if st.button("결과 복사"):
                    pyperclip.copy(review_result)
                    st.success("클립보드에 복사되었습니다!")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                if debug_mode:
                    st.exception(e)
        else:
            st.write("원본 문서와 요약된 내용을 모두 입력해주세요.")
