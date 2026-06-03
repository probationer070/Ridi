import requests
import re
import pandas as pd
import time
import random
from bs4 import BeautifulSoup

"""
북마크의 했던 url들을 크롤링하여 제목을 추출하고 CSV로 저장
- 입력 파일: bookmark.txt
- 출력 파일: crawled_data.csv
"""


def extract_subject(parsed_text: BeautifulSoup):
    """제목 추출"""
    # 제목 추출 (예: og:title 메타 태그)
    og_title = parsed_text.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title['content'].strip()

    # 제목 추출 (예: <title> 태그)
    title_tag = parsed_text.find("title")
    if title_tag:
        return title_tag.text.strip()
    
    # 제목 추출 (예: <h1> 태그)
    h1_title = parsed_text.find("h1")
    if h1_title:
        return h1_title.text.strip()
    
    return "제목을 찾을 수 없습니다."

def ErrResolve_429(url, headers, response):
    # 429 에러 발생 시 처리
    if response.status_code == 429:
        wait_time = int(response.headers.get("Retry-After", 5))
        print(f"[429 Error] 너무 많은 요청. {wait_time}초 대기 후 재시도...")
        time.sleep(wait_time)
        response = requests.get(url, headers=headers, timeout=15)
    return response

def get_article_title(url):
    """URL에 접속해 기사 제목과 내용 추출
    - 파싱이후에도 깨진 한글이 있는지 재확인
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.google.com/", # 구글에서 유입된 것처럼 속임
    }

    Accepts_List = [
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        
    ]

    soup = None

    try:
        
        response = requests.get(url, headers=headers, timeout=10)
        response = ErrResolve_429(url, headers, response)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생

        # 1. 우선 requests가 감지한 인코딩(apparent_encoding)으로 시도
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        title = extract_subject(soup)

        if title != "제목을 찾을 수 없습니다." and not is_broken(title):
            return title

        # 2. 깨졌다면 준비된 인코더 중 하나로 다시 시도
        try_encode = ['utf-8', 'cp949', 'euc-kr']
        for enc in try_encode:
            if response.encoding and enc.lower() == response.encoding.lower():
                continue
            attempt_text = response.content.decode(enc, errors='ignore')
            temp_soup = BeautifulSoup(attempt_text, 'html.parser')
            temp_title = extract_subject(temp_soup)
            if temp_title != "제목을 찾을 수 없습니다." and not is_broken(temp_title):
                return temp_title
        
        return title
    
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        print(f"[❌ Request Error] {code}/ {e}")
        return f"[❌ Request Error]"

    except Exception as e:
        soup_content = soup.prettify()[:100] if 'soup' in locals() and soup else 'N/A'
        print(f"[❌ Error] {e}, 원문 내용: {soup_content}")
        return f"[❌ Error]"

def is_broken(text):
    """텍스트에 한글 완성형이 포함되어 있는지 확인하여 깨짐 여부 판단"""
    if not text: return False
    # 한글(가-힣) 비율이 너무 낮으면 깨진 것으로 간주
    korean_chars = re.findall(r'[가-힣]', text)
    return len(korean_chars) == 0

def crawled_data2csv(input_file):
    """크롤링한 데이터를 CSV 파일로 저장
    
    입력된 txt 파일 양식
    - 그룹 구분: - 그룹명
    - URL: http(s)://....
    <예시>
    - 뉴스
    http://news.example.com/article1
    http://news.example.com/article2
    - 블로그
    http://blog.example.com/post1
    http://blog.example.com/post2   
    """
    res = []

    with open(input_file, "r",encoding="utf-8") as f:
        lines = f.readlines()
    
    for line in lines:
        url = line.strip()
        if not url:
            continue

        if line.startswith("-"):
            print(f"[📂 Current Group]: {line.strip()}")
            group = line.strip().replace("-", "").strip()
            continue

        # URL인 경우 기사 제목 추출
        if line.startswith("http"):
            print(f"[Crawling]: {url}")
            title = get_article_title(url)
            res.append({
                    "group": group,
                    "url": url,
                    "title": title
                })
            # 한 번 거절 당한 경우 (429 등) 대비 딜레이 추가후 다시시도
            if "429" in title:
                print("재시도 중...")
                time.sleep(random.uniform(1, 3))  # 1~3초 랜덤 딜레이
                title = get_article_title(url)
                res[-1]["title"] = title
            else:
                pass

    df = pd.DataFrame(res)
    df.to_csv("crawled_data.csv", index=True, encoding="utf-8-sig")
    print("CSV 파일로 저장 완료: crawled_data.csv")
    print(f"\n✨ 저장 완료! 총 {len(res)}개의 데이터가 저장되었습니다.")
    print(f"3번째 칼럼의 데이터 개수: {len(df['title'])}개")

if __name__ == "__main__":
    crawled_data2csv("bookmark.txt" )