import time
import os
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

class FileWatcher(FileSystemEventHandler):
    def __init__(self, callback):
        self.pending_files = {}
        self.callback = callback
        self.lock = threading.Lock()  # 스레드 안전성 보장

    def on_created(self, event):
        if event.is_directory:
            return
        file_path = os.path.normpath(event.src_path)
        with self.lock:
            if file_path not in self.pending_files:
                print(f"파일 생성 감지: {file_path} at {datetime.now()}")
                self.pending_files[file_path] = "created"
                # 별도 스레드에서 쓰기 완료 체크
                threading.Thread(target=self.check_file_completion, args=(file_path,), daemon=True).start()

    def on_modified(self, event):
        if event.is_directory:
            return
        file_path = os.path.normpath(event.src_path)
        with self.lock:
            if file_path in self.pending_files and self.pending_files[file_path] == "created":
                # 이미 스레드가 실행 중이니 추가 호출 불필요
                pass

    def check_file_completion(self, file_path):
        try:
            with self.lock:
                if self.pending_files.get(file_path) == "processing":
                    return
                self.pending_files[file_path] = "processing"

            last_size = -1
            stable_count = 0
            max_checks = 5
            check_interval = 1

            while stable_count < max_checks:
                try:
                    current_size = os.path.getsize(file_path)
                except FileNotFoundError:
                    print(f"파일이 삭제됨: {file_path}")
                    with self.lock:
                        del self.pending_files[file_path]
                    return
                except PermissionError:
                    print(f"파일 접근 불가: {file_path} - 잠김 또는 권한 문제")
                    with self.lock:
                        del self.pending_files[file_path]
                    return

                if current_size == last_size and last_size > 0:
                    stable_count += 1
                else:
                    stable_count = 0
                last_size = current_size
                time.sleep(check_interval)

            print(f"파일 쓰기 완료: {file_path} at {datetime.now()} (크기: {last_size} bytes)")
            self.process_file(file_path)
            with self.lock:
                del self.pending_files[file_path]

        except Exception as e:
            print(f"파일 확인 중 오류: {file_path} - {str(e)}")
            with self.lock:
                if file_path in self.pending_files:
                    del self.pending_files[file_path]

    def process_file(self, file_path):
        if self.callback:
            self.callback(file_path)
        else:
            print(f"작업 시작: {file_path} - 콜백이 없어요!")

def start_watching(path_to_watch, callback=None):
    event_handler = FileWatcher(callback)
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)
    observer.start()
    print(f"폴더 감시 시작 (하위 폴더 포함): {path_to_watch}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("감시 종료")
    observer.join()

# if __name__ == "__main__":
#     def my_callback(file_path):
#         print(f"콜백 실행: {file_path} - 여기서 원하는 작업을 하세요!")

#     watch_path = r"D:\002.Code\002.node\weather_api\kma_fetch\data\weather\gk2a"
#     start_watching(watch_path, my_callback)