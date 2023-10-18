import cv2
from PIL import ImageGrab
import win32gui
import win32con
import pygetwindow as gw
import numpy as np
import torch

# YOLO model load
model_path = 'E:\ProjectGit\Detect_Inappropriate_Image\weights\exp1\\best.pt'  # Pre-trained weight pt file path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# 모델 설정
model.conf = 0.25  # confidence threshold
model.imgsz = 640  # image size

def get_window_list():
    windows = gw.getAllWindows()
    titled_windows = [window for window in windows if window.title.strip() != ""]
    return titled_windows

def select_window_by_number():
    windows = get_window_list()

    print("\n실행 중인 윈도우 목록:")
    for i, window in enumerate(windows, 1):
        print(f"{i}: {window.title}")

    while True:
        try:
            selection = int(input("\n캡쳐할 윈도우 번호를 선택하세요: "))
            if 1 <= selection <= len(windows):
                return windows[selection - 1].title
            else:
                print("잘못된 선택입니다. 목록에 있는 번호를 선택해주세요.")
        except ValueError:
            print("유효한 숫자를 입력해주세요.")


def capture_window(window_title):
    try:
        # Find windows
        target = gw.getWindowsWithTitle(window_title)[0]
        if not target:
            print(f"'{window_title}' 제목의 윈도우를 찾을 수 없습니다.")
            return

        if target.isMinimized:
            win32gui.ShowWindow(target._hWnd, win32con.SW_RESTORE)       

        # capture board update
        while True:
            x, y, x1, y1 = target.left, target.top, target.right, target.bottom

            screenshot = ImageGrab.grab(bbox=(x, y, x1, y1))
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            results = model(frame)  # predict
            frame = results.render()[0] # render results
            
            cv2.imshow('Captured Window', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    except Exception as e:
        print(e)
        cv2.destroyAllWindows()
        return

if __name__ == "__main__":
    selected_window_title = select_window_by_number()
    capture_window(selected_window_title)
