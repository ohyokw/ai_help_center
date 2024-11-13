import os
import wave
import json
import vosk
import pyaudio
import threading

# 모델 경로 설정
MODEL_PATH = "vosk-model-small-ko-0.22"

# Vosk 모델 로드
if not os.path.exists(MODEL_PATH):
    raise ValueError("모델 경로가 올바르지 않습니다.")
model = vosk.Model(MODEL_PATH)

# PyAudio 설정
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
stream.start_stream()

# 음성 인식기 생성
rec = vosk.KaldiRecognizer(model, 16000)

# 종료 플래그 설정
is_running = True

# 키 입력 감지 함수
def listen_for_exit():
    global is_running
    while True:
        user_input = input()
        if user_input.lower() == 'q':
            is_running = False
            break

# 키 입력 감지 쓰레드 시작
input_thread = threading.Thread(target=listen_for_exit)
input_thread.start()

print("음성 인식 중... 'q'를 입력하여 종료하세요.")

try:
    while is_running:
        data = stream.read(4096, exception_on_overflow=False)
        
        if len(data) == 0:
            break
        
        # 음성 데이터 처리
        if rec.AcceptWaveform(data):
            result = rec.Result()
            result_json = json.loads(result)
            print("인식된 텍스트:", result_json.get("text", ""))
        
        else:
            partial_result = rec.PartialResult()
            partial_result_json = json.loads(partial_result)
            print("진행 중 텍스트:", partial_result_json.get("partial", ""))
            
except KeyboardInterrupt:
    print("음성 인식이 종료되었습니다.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
