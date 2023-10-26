import os
import pygetwindow as gw
import pyautogui
import keyboard
from PIL import ImageDraw
import numpy as np
import time
import win32gui
import win32con
import pygetwindow as gw
import win32api
from datetime import datetime
from PIL import ImageFont
import keras_ocr

save_folder = "ocr_result"
pipeline = keras_ocr.pipeline.Pipeline()

def force_activate_window(window):
    try:
        hwnd = window._hWnd

        if window.isMinimized:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

        win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)  # Alt 키 누름
        time.sleep(0.1)  # 짧은 지연
        win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)  # Alt 키 뗌

        win32gui.SetForegroundWindow(hwnd)
        time.sleep(2)

        return True
    except Exception as e:
        print(f"An error occurred while trying to force activate the window: {e}")
        return False

def capture_screenshot(window):
    if not window.isActive:
        print("Window is not active, attempting to activate...")
        force_activate_window(window)

    if window.isActive:
        print(window.title)
        x, y, width, height = window.left, window.top, window.width, window.height
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        return screenshot
    else:
        print(f"The window is not active.")
        return None

def perform_ocr_and_save(screenshot, save_text=False, save_image=False):

    screenshot_np = np.array(screenshot)
    ocr_results = pipeline.recognize([screenshot_np])[0] 

    if save_text:
        save_text_results(ocr_results)
    elif save_image:
        save_image_with_boxes(screenshot, ocr_results)

def save_text_results(text_results):

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"ocr_result_text_{current_time}.txt"
    save_path = os.path.join(save_folder, filename)
    
    with open(save_path, 'w', encoding='utf-8') as text_file:
        for result in text_results:
            text = result[0]  # 이 부분이 문자열을 가리킵니다.
            line = text + '\n'
            text_file.write(line)
    print(f"Text results saved to {save_path}")

def save_image_with_boxes(image, ocr_results):
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("C:\Windows\Fonts\malgun", 12)
    except IOError:
        print("Could not find the 'Malgun Gothic' font on your system.")
        return

    for result in ocr_results:
        text, bbox  = result
        
        bbox_tuples = [tuple(point) for point in bbox]
        draw.polygon(bbox_tuples, outline="red")

        text_origin = (bbox[0][0], bbox[0][1])  
        draw.text(text_origin, text, fill="red", font=font)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    filename = f"ocr_result_image_{current_time}.png"
    save_path = os.path.join(save_folder, filename)
    
    image.save(save_path)
    print(f"Image saved to {save_path}")

def main():
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    all_windows = gw.getAllWindows()

    print("Available windows:")
    for i, window in enumerate(all_windows, 1):
        print(f"{i}: {window.title}")

    selected = int(input("Enter the number of the window you want to capture: "))
    selected_window = all_windows[selected - 1]

    print("Press Ctrl+P to capture and save text, Ctrl+I to capture and save image with bounding boxes, or 'q' to quit.")

    while True:
        event = keyboard.read_event()
        if event.event_type == 'down':
            if keyboard.is_pressed('ctrl+p'):
                screenshot = capture_screenshot(selected_window)
                if screenshot:
                    perform_ocr_and_save(screenshot, save_text=True)
                keyboard.read_event()
            elif keyboard.is_pressed('ctrl+i'):
                screenshot = capture_screenshot(selected_window)
                if screenshot:
                    perform_ocr_and_save(screenshot, save_image=True)
                keyboard.read_event()
            elif event.name == 'q':
                print("Program exited.")
                break

if __name__ == "__main__":
    main()
