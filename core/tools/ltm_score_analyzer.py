import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트 경로를 sys.path에 추가하여 configs 모듈을 임포트할 수 있도록 함
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from configs.system_config import SystemConfig

def visualize_ltm_scores(log_path: str, output_dir: str):
    """
    ltm_scoring_log.csv 파일을 읽어, 검색 쿼리별로 점수 기여도를 시각화하고 이미지 파일로 저장합니다.
    """
    # 1. 데이터 로드 및 전처리
    if not os.path.exists(log_path):
        print(f"오류: 로그 파일을 찾을 수 없습니다. 경로: {log_path}")
        return

    try:
        df = pd.read_csv(log_path)
        if df.empty:
            print("로그 파일이 비어있습니다. 분석할 데이터가 없습니다.")
            return
    except Exception as e:
        print(f"로그 파일 로딩 중 오류 발생: {e}")
        return

    # 숫자형으로 변환
    score_cols = ['final_score', 'semantic_norm', 'keyword_norm', 'recency_factor', 
                  'semantic_weight', 'keyword_weight', 'recency_weight']
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=score_cols, inplace=True)

    # 각 점수 요소의 가중치가 적용된 기여도 계산
    df['semantic_contribution'] = df['semantic_norm'] * df['semantic_weight']
    df['keyword_contribution'] = df['keyword_norm'] * df['keyword_weight']
    df['recency_contribution'] = df['recency_factor'] * df['recency_weight']

    # 2. 시각화
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 한글 폰트 설정 (맑은 고딕) - Windows 기준
    # 다른 OS의 경우, 적절한 폰트 이름으로 변경해야 합니다. (예: 'AppleGothic' for macOS)
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지
    except:
        print("경고: 'Malgun Gothic' 폰트를 찾을 수 없습니다. 시각화 결과에서 한글이 깨질 수 있습니다.")

    # 고유한 검색 쿼리별로 그룹화하여 처리
    for query, group in df.groupby('search_query'):
        print(f"\n--- 쿼리 분석 중: '{query}' ---")
        
        # 시각화를 위해 데이터 정렬
        group = group.sort_values('final_score', ascending=False).reset_index()
        
        if group.empty:
            continue

        plt.figure(figsize=(12, 8))
        
        # x축 레이블 설정 (item_id와 content preview)
        labels = [f"{row.item_id[:8]}...\n'{row.content_preview[:20]}...'" for index, row in group.iterrows()]

        # 누적 막대 그래프 생성
        plt.bar(labels, group['semantic_contribution'], color=sns.color_palette("pastel")[0], label='Semantic Contribution')
        plt.bar(labels, group['keyword_contribution'], bottom=group['semantic_contribution'], color=sns.color_palette("pastel")[1], label='Keyword Contribution')
        plt.bar(labels, group['recency_contribution'], bottom=group['semantic_contribution'] + group['keyword_contribution'], color=sns.color_palette("pastel")[2], label='Recency Contribution')

        # 각 막대 위에 최종 점수 표시
        for i, total_score in enumerate(group['final_score']):
            plt.text(i, total_score + 0.01, f'{total_score:.3f}', ha='center', fontsize=9)

        plt.ylabel('Final Score')
        plt.xlabel('Retrieved Memory Items')
        plt.title(f"LTM 검색 결과 분석\nQuery: '{query}'", fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.legend()
        plt.tight_layout()

        # 파일명으로 부적합한 문자 제거
        safe_query = "".join([c for c in query if c.isalnum() or c in " _-"]).rstrip()[:50]
        output_filename = os.path.join(output_dir, f"ltm_analysis_{safe_query}.png")
        
        plt.savefig(output_filename)
        print(f"분석 결과가 이미지 파일로 저장되었습니다: {output_filename}")
        plt.close()

def main():
    """
    스크립트의 메인 실행 함수
    """
    config = SystemConfig()
    log_file_path = config.ltm_scoring_log_path
    
    # 분석 결과를 저장할 디렉토리 설정
    analysis_output_dir = os.path.join(config.project_root, "analysis_output")
    
    visualize_ltm_scores(log_file_path, analysis_output_dir)

if __name__ == "__main__":
    main()