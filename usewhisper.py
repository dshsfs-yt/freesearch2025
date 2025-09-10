import os
import tkinter as tk
from tkinter import filedialog
import whisper
import time
def choose_audio_file():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="MP3 파일 선택",
        filetypes=[("Audio", "*.mp3;*.wav;*.m4a;*.mp4;*.ogg;*.webm;*.flac")]
    )
    root.destroy()
    return path

def main():
    print("오디오 파일을 선택하세요.")
    audio_path = choose_audio_file()
    if not audio_path:
        print("파일이 선택되지 않았습니다."); return
    start_time= time.time()


    # 한국어라면 language="ko" 지정 시 속도/정확도 향상 가능
    model_size = "small"   # tiny/base/small/medium/large 중 선택
    print(f"[Whisper 로컬] 모델: {model_size}, 파일: {os.path.basename(audio_path)}")

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language=None)  # 한국어 위주면 language="ko" 권장

    end= time.time()
    print(f"처리 시간: {end - start_time:.2f}초")
    # 콘솔에도 텍스트 출력
    print("\n----- Transcript Preview -----")
    if "text" in result:
        print(result["text"][:2000])  # 너무 길면 앞부분만 미리보기

if __name__ == "__main__":
    main()