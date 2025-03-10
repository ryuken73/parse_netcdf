import json
import psycopg2
import os
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 데이터베이스 연결 설정
db_config = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# JSON 파일 경로
json_file_path = './jsonfiles/gk2a_ami_le1b_ir105_ea020lc_202503011340_step10.json'

# 파일명에서 메타데이터 추출 함수
def extract_metadata_from_filename(filename):
    basename = Path(filename).stem
    parts = basename.split('_')
    print(parts)
    observation_area = parts[4][:2]  # 'ea'
    print(observation_area)
    time_str = parts[5]              # '202503011340'
    print(time_str)
    step = int(parts[6].replace('step', ''))  # 'step10' -> 10
    print(step)
    print('extract metadata from file:', filename)

    # UTC 시간 변환
    observation_time_utc = datetime.strptime(time_str, '%Y%m%d%H%M')
    
    # KORST 시간 계산 (UTC + 9시간)
    observation_time_korst = observation_time_utc + timedelta(hours=9)
    print('meta data from fname:', observation_area, observation_time_utc, observation_time_korst, step)
    
    return observation_area, observation_time_utc, observation_time_korst, step

# JSON 데이터 삽입 함수
def insert_json_data(json_file_path, db_config):
    conn = None
    cur = None
    try:
        # 데이터베이스 연결
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # 파일명에서 메타데이터 추출
        observation_area, observation_time_utc, observation_time_korst, step = extract_metadata_from_filename(json_file_path)

        # JSON 파일 읽기
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)

        # 데이터 삽입 쿼리
        insert_query = """
        INSERT INTO ir105_json (observation_time_utc, observation_time_kor, observation_area, step, data)
        VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(insert_query, (
            observation_time_utc,
            observation_time_korst,
            observation_area,
            step,
            json.dumps(json_data)  # JSON 데이터를 문자열로 변환
        ))

        # 커밋
        conn.commit()
        print(f"Successfully inserted data from {json_file_path}")

    except psycopg2.errors.UniqueViolation as e:
        print(f"Error: Duplicate entry for observation_time_utc={observation_time_utc}, observation_area={observation_area}, step={step}")
        print(f"Details: {e}")
        if conn:
            conn.rollback()

    except Exception as e:
        print(f"Error: {e}")
        if conn:
            conn.rollback()

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# 실행
if __name__ == "__main__":
    insert_json_data(json_file_path, db_config)