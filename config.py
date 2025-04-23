import os
from dotenv import load_dotenv

# 환경 변수 로드
env = os.getenv("ENV", "dev")  # 기본값은 development
print('xxx', env)
load_dotenv(f".env.{env}")

class Config:
  ENV = os.getenv("ENV")
  OUT_PATH = os.getenv("OUT_PATH")
  WATCH_PATH = os.getenv("WATCH_PATH")
  OS_SEP = os.getenv("OS_SEP")

# 환경에 따른 설정 가져오기
def get_config():
  return Config()