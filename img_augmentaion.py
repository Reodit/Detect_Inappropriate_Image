import tkinter as tk
from tkinter import filedialog, Label, Menu, Canvas, Scrollbar, Frame, Toplevel, Listbox, OptionMenu, StringVar, Entry, Button
from PIL import Image, ImageTk
from imgaug import augmenters as iaa 
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import os
from datetime import datetime
import imgaug as ia
import itertools
import imageio
import cv2

class ImageAugmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Augmentation App")
        self.root.geometry("800x600")  # Set initial window size

        # List to store image data
        self.original_images = []  # Original image data
        self.original_images_np = [] # Original image data np
        self.display_images = []  # Images to be displayed on the screen
        self.image_frames = []  # Frames holding the images and the labels
        
        # Set up the menu
        menubar = Menu(root)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="이미지 파일 선택", command=self.load_images)
        file_menu.add_command(label="종료", command=root.quit)
        menubar.add_cascade(label="파일", menu=file_menu)

        augmentation_menu = Menu(menubar, tearoff=0)
        augmentation_menu.add_command(label="증강하기", command=self.open_augmentation_window)
        menubar.add_cascade(label="이미지 증강", menu=augmentation_menu)

        root.config(menu=menubar)

        # This label is placed outside the scrollable area, at the top of the window
        self.info_label = Label(self.root, text="이미지 파일을 선택해주세요.")
        self.info_label.pack()

        # Set up the scrollable canvas and the frame within it
        self.canvas = Canvas(root, bg='white')
        self.scrollable_frame = Frame(self.canvas)
        self.scrollbar = Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)  # Windows OS
        self.canvas.bind_all("<Button-4>", self.on_mousewheel)  # Linux, Mac OS (scroll up)
        self.canvas.bind_all("<Button-5>", self.on_mousewheel)  # Linux, Mac OS (scroll down)

        # Bind the event for window resize
        self.root.bind('<Configure>', self.on_window_resize)

    def on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))  # Update the scrollable area

    def on_mousewheel(self, event):
        # Handle the scroll event depending on the OS
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")  # Scroll up
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")  # Scroll down

    def on_window_resize(self, event=None):
        # Re-grid the images based on the new window size
        if self.image_frames:  # If there are images loaded
            num_images_width = self.root.winfo_width() // 260  # We assume image width of 250 with some padding
            for i, frame in enumerate(self.image_frames):
                frame.grid(row=i // num_images_width, column=i % num_images_width, padx=5, pady=5)

    def load_images(self):
        # Open the file selection dialog
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        
        if not file_paths:
            return

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.original_images.clear()
        self.display_images.clear()
        self.image_frames.clear()
        self.original_images_np.clear()
        
        num_images_width = self.root.winfo_width() // 260  # Assuming image width of 250 with some padding

        for i, file_path in enumerate(file_paths):
            # Load the image and save it to the list
            image = Image.open(file_path)
            self.original_images.append(image)
            image_np = imageio.v2.imread(file_path)
            self.original_images_np.append((image_np, (self.read_label_data(file_path))))
            
            # Resize the image for display
            display_image = image.resize((250, 250), Image.LANCZOS)  # or Image.ANTIALIAS
            photo_image = ImageTk.PhotoImage(display_image)
            self.display_images.append(photo_image)

            # Extract the file name
            file_name = file_path.split("/")[-1]

            # Create a new frame for each image and its name, then place it on the grid
            frame = Frame(self.scrollable_frame, bg='white')
            frame.grid(row=i // num_images_width, column=i % num_images_width, padx=5, pady=5)
            self.image_frames.append(frame)  # Save the frame reference for re-gridding on resize

            image_label = Label(frame, image=photo_image)
            image_label.image = photo_image  # keep a reference
            image_label.pack()

            name_label = Label(frame, text=file_name, wraplength=250, justify="center", bg='white')
            name_label.pack()

        # Update the instruction label
        self.info_label.config(text=f"{len(file_paths)}개의 이미지 파일이 선택되었습니다.")

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))  # Update the scrollable area
    
    def read_label_data(self, file_path):
        txt_file_path = os.path.splitext(file_path)[0] + ".txt"
        
        if not os.path.exists(txt_file_path):
            return None

        # 라벨 데이터를 저장할 리스트를 초기화합니다.
        labels = []

        # 텍스트 파일을 엽니다.
        with open(txt_file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                components = stripped_line.split()

                # 첫 번째 요소는 라벨이므로 정수로 변환합니다.
                label = int(components[0])

                # 나머지 요소는 바운딩 박스의 좌표이므로 float로 변환합니다.
                bbox_coords = [float(num) for num in components[1:]]

                # 라벨과 좌표를 결합한 리스트를 labels에 추가합니다.
                labels.append([label] + bbox_coords)

        return labels

    def open_augmentation_window(self):
        self.augmentation_window = Toplevel(self.root)
        self.augmentation_window.title("이미지 증강")
        self.augmentation_window.geometry("1600x800") 
        self.augmentation_window.resizable(False, False) 

        scrollbar = tk.Scrollbar(self.augmentation_window)
    
        self.image_listbox = tk.Listbox(self.augmentation_window, height=20, width=40, selectmode='extended', yscrollcommand=scrollbar.set)
        for i, img in enumerate(self.original_images):
            file_name = img.filename.split("/")[-1] 
            self.image_listbox.insert(i, file_name)
        scrollbar.config(command=self.image_listbox.yview)

        self.image_listbox.pack(side="left", fill="y")
        self.image_listbox.pack(side="left", fill="both", expand=True)  

        self.preview_label = Label(self.augmentation_window, text="이미지 미리보기")
        self.preview_label.pack(side="top")

        self.augmentation_options = ["직접 augmentation 함수를 지정", "모든 augmentation 함수를 실행", "모든 경우의 수의 augmentation 함수를 실행"]
        self.selected_option = StringVar()
        self.selected_option.set(self.augmentation_options[0])
        self.options_menu = OptionMenu(self.augmentation_window, self.selected_option, *self.augmentation_options)
        
        self.selection_info_label = Label(self.augmentation_window, text="")
        self.selection_info_label.pack(side="bottom")

        self.execution_count_label = Label(self.augmentation_window, text="함수 실행 수:")
        self.execution_count_entry = Entry(self.augmentation_window)
        
        self.augment_button = Button(self.augmentation_window, text="증강하기", command=self.perform_augmentation)
        
        self.augmentation_functions = {
            "좌우 반전": [],
            "상하 반전": [],
            "아핀 변환": ["scale_x", "scale_y", "translate_percent", "rotate", "shear"],
            "확대-축소 변환": ["percent"],
            "가우시안 노이즈": ["noise"],
            "명암 변경": ["alpha"],
            "가우시안 블러": ["sigma"],
            "선명도 변경": ["alpha", "lightness"],
            "JPEG 압축 품질 조정": ["quality"],
            "채도 변경": ["mul"]
        }
        self.function_check_vars = [tk.IntVar() for _ in self.augmentation_functions]
        
        self.checkbox_vars_mapping = {}

        self.checkboxes = []
        self.input_fields = {}
        self.frames = []
        
        for function, params in self.augmentation_functions.items():
            frame = tk.Frame(self.augmentation_window)
            frame.pack(fill=tk.X)
            self.frames.append(frame) 

            check_var = self.function_check_vars[len(self.checkboxes)]
            checkbox = tk.Checkbutton(frame, text=function, variable=check_var) 
            checkbox.pack(side=tk.LEFT) 
            self.checkboxes.append(checkbox)

            self.checkbox_vars_mapping[function] = check_var

            self.input_fields[function] = []
            param_entries = {}

            for param in params:
                label_min = tk.Label(frame, text=f"{param} (min)")
                label_min.pack(side=tk.LEFT)

                entry_min = tk.Entry(frame, width=5, name=f"{param}_min")
                entry_min.pack(side=tk.LEFT)

                label_max = tk.Label(frame, text=f"{param} (max)")
                label_max.pack(side=tk.LEFT)

                entry_max = tk.Entry(frame, width=5, name=f"{param}_max") 
                entry_max.pack(side=tk.LEFT)

                param_entries[param] = {"min": entry_min, "max": entry_max}

            self.input_fields[function] = param_entries

        self.label_creation_var = tk.IntVar() 
        self.label_creation_checkbox = tk.Checkbutton(self.augmentation_window, text="라벨 포함 생성", variable=self.label_creation_var)

        self.hide_menus()

        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        self.selected_option.trace("w", self.ui_packing)    

    def get_input_values(self):
        """입력 필드에서 사용자 입력값을 추출하고 반환합니다."""
        function_params = {}       

        for function, param_entries in self.input_fields.items():
            params = {}
            # 각 파라미터에 대해 min, max 값을 가져옵니다.
            for param_name, entries in param_entries.items():
                min_entry = entries["min"]  # 'min' 엔트리 위젯
                max_entry = entries["max"]  # 'max' 엔트리 위젯

                # 사용자가 값을 입력했는지 확인하고, 값이 있으면 float으로 변환합니다.
                min_value = min_entry.get()
                max_value = max_entry.get()

                min_value = float(min_value) if min_value.strip() else None  # strip()으로 양쪽 공백 제거
                max_value = float(max_value) if max_value.strip() else None  # strip()으로 양쪽 공백 제거

                # min, max 값을 튜플로 묶어서 해당 파라미터의 값으로 설정합니다.
                params[param_name + "_min"] = min_value
                params[param_name + "_max"] = max_value

            # 해당 함수의 파라미터 값을 저장합니다.
            function_params[function] = params 

        return function_params

    def hide_menus(self):
        for frame in self.frames:
            frame.pack_forget()
        self.label_creation_checkbox.pack_forget()
        self.options_menu.pack_forget()
        self.execution_count_label.pack_forget()
        self.execution_count_entry.pack_forget()
        self.augment_button.pack_forget()

    def ui_packing(self, *arg):
        # UI 기본 배치
        self.hide_menus()
        self.options_menu.pack(side="top", pady=5)

        selected = self.selected_option.get()
        
        if selected == "직접 augmentation 함수를 지정" or selected == "모든 경우의 수의 augmentation 함수를 실행":
            for frame in self.frames:
                frame.pack(side="top", anchor="w", fill=tk.X)
        else:
            for frame in self.frames:
                frame.pack_forget()

        self.execution_count_label.pack(side="top", pady=(10, 0))
        self.execution_count_entry.pack(side="top", pady=(0, 10))
        self.augment_button.pack(side="top", pady=20)
        self.label_creation_checkbox.pack(side="top", padx=10)

    def on_image_select(self, event):
        # 선택한 이미지 인덱스들 가져오기
        selection_indices = self.image_listbox.curselection()
        if selection_indices:
            first_selected_index = selection_indices[0]
            selected_image = self.original_images[first_selected_index]

            self.ui_packing()

            # 이미지의 미리보기 생성
            preview_image = selected_image.resize((250, 250), Image.BILINEAR)
            photo_image = ImageTk.PhotoImage(preview_image)

            # 미리보기 레이블에 이미지 설정
            self.preview_label.config(image=photo_image)
            self.preview_label.image = photo_image  # 참조 보관

            # 선택된 이미지 수에 따른 텍스트 정보 업데이트
            first_file_name = selected_image.filename.split("/")[-1]

            if len(selection_indices) > 1:
                additional_count = len(selection_indices) - 1
                selection_text = f"{first_file_name} 외 {additional_count}개의 이미지가 선택되었습니다."
            else:
                selection_text = f"{first_file_name}."

            self.selection_info_label.config(text=selection_text)

    def apply_fliplr(self, image, bbs, p=1.0):  # 좌우 반전
        aug = iaa.Fliplr(p)
        return aug(image=image, bounding_boxes=bbs)

    def apply_flipud(self, image, bbs, p=1.0):  # 상하 반전
        aug = iaa.Flipud(p)
        return aug(image=image, bounding_boxes=bbs)

    def apply_affine(self, image, bbs, scale_x=(0.8, 1.2), scale_y=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-45, 45), shear=(-16, 16)):  # 아핀 변환
        aug = iaa.Affine(
            scale={"x": scale_x, "y": scale_y},
            translate_percent={"x": translate_percent, "y": translate_percent},
            rotate=rotate,
            shear=shear
        )
        return aug(image=image, bounding_boxes=bbs)

    def apply_CropAndPad(self, image, bbs, percent=(-0.25, 0.25)): # 확대 및 축소
        aug = iaa.CropAndPad(percent=percent)
        return aug(image=image, bounding_boxes=bbs)

    def apply_gaussian_noise(self, image, bbs, noise=(0.01*255, 0.05*255)):  # 가우시안 노이즈 추가
        aug = iaa.AdditiveGaussianNoise(scale=noise)
        return aug(image=image, bounding_boxes=bbs)

    def apply_contrast_norm(self, image, bbs, alpha=(0.5, 2)):  # 명암 대비 조정
        aug = iaa.ContrastNormalization(alpha)
        return aug(image=image, bounding_boxes=bbs)

    def apply_gaussian_blur(self, image, bbs, sigma=(0.0, 3.0)):  # 가우시안 블러
        aug = iaa.GaussianBlur(sigma=sigma)
        return aug(image=image, bounding_boxes=bbs)

    def apply_sharpen(self, image, bbs, alpha=(0.0, 1.0), lightness=(0.75, 1.5)):  # 선명도 조절
        aug = iaa.Sharpen(alpha=alpha, lightness=lightness)
        return aug(image=image, bounding_boxes=bbs)

    def apply_jpeg_compression(self, image, bbs, quality=(70, 99)):  # JPEG 압축 품질 조정
        # 이미지가 2차원 배열인지 (즉, 그레이스케일 이미지인지) 확인
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)  # 이제 shape는 (height, width, 1)이 됩니다.

        # 이미지가 알파 채널을 포함하는 4채널 이미지인 경우 알파 채널 제거
        elif image.shape[2] == 4:
            image = self.remove_alpha_channel(image)

        aug = iaa.JpegCompression(compression=quality)
        return aug(image=image, bounding_boxes=bbs)

    def apply_saturation(self, image, bbs, mul=(0.5, 1.5)):  # 채도 조정
        if len(image.shape) == 2 or image.shape[2] == 1: 
            image = np.stack((image.squeeze(),) * 3, axis=-1) 

        elif image.shape[2] == 4:
            image = self.remove_alpha_channel(image)
        
        aug = iaa.MultiplySaturation(mul=mul)
        return aug(image=image, bounding_boxes=bbs)
    
    def remove_alpha_channel(self, image):
        # 알파 채널을 제거하고 RGB 이미지 반환
        return image[:, :, :3]  # 처음 세 채널 (R, G, B)만 유지
        
    def get_param_with_defaults(self, params, param_name, default_min, default_max):
        min_value = params.get(f"{param_name}_min")
        if min_value is not None:
            min_value = float(min_value)
        else:
            min_value = default_min  # min_value가 None이면 기본값 사용

        max_value = params.get(f"{param_name}_max")
        if max_value is not None:
            max_value = float(max_value)
        else:
            max_value = default_max  # max_value가 None이면 기본값 사용

        return (min_value, max_value)

    def convert_yolo_to_bbox(self, width, height, yolo_data):
        """
        YOLO 형식의 바운딩 박스 데이터를 전통적인 (x1, y1, x2, y2) 형식으로 변환합니다.

        :param yolo_data: YOLO 형식의 데이터 (라벨, x_center, y_center, width, height).
        :return: 라벨과 (x1, y1, x2, y2) 형식의 튜플.
        """
        # yolo_data 리스트에서 정보 추출
        label = yolo_data[0]
        abs_x_center = yolo_data[1] * width
        abs_y_center = yolo_data[2] * height
        abs_box_width = yolo_data[3] * width
        abs_box_height = yolo_data[4] * height

        # (x1, y1)은 왼쪽 상단 모서리를 나타냅니다.
        x1 = abs_x_center - (abs_box_width / 2)
        y1 = abs_y_center - (abs_box_height / 2)

        # (x2, y2)는 오른쪽 하단 모서리를 나타냅니다.
        x2 = abs_x_center + (abs_box_width / 2)
        y2 = abs_y_center + (abs_box_height / 2)

        return label, (x1, y1, x2, y2)

    def perform_augmentation(self):
        selection_indices = self.image_listbox.curselection()
        if not selection_indices:
            return
        
        try:
            execution_count = int(self.execution_count_entry.get())
        except ValueError:
            execution_count = 1 # exception일 때 강제로 값 1로 세팅
            return

        function_params = self.get_input_values()
        current_option = self.selected_option.get()
        
        folder_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(folder_name, exist_ok=True)

        for index in selection_indices:
            image = self.original_images[index]            
            image_name, image_extension = os.path.splitext(image.filename.split("/")[-1])
            image_np, label_data = self.original_images_np[index]
            image_height, image_width, channel = image_np.shape
            
            converted_bboxes = []
            bbs = None

            if label_data is not None:
                # 각 YOLO 데이터에 대해 변환 수행
                for yolo_data in label_data:
                    label, (x1, y1, x2, y2) = self.convert_yolo_to_bbox(image_width, image_height, yolo_data)
                    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label)
                    converted_bboxes.append(bbox)
                
                bbs = BoundingBoxesOnImage(converted_bboxes, shape=image_np.shape)

            j = 0
            for i in range(execution_count):  # 설정된 실행 횟수만큼 반복
                transformed_image_np = image_np.copy()
                transformed_bbs = bbs.copy()

                print(transformed_bbs)
                if current_option == "직접 augmentation 함수를 지정":
                    if self.function_check_vars[0].get():  
                        transformed_image_np, transformed_bbs = self.apply_fliplr(transformed_image_np, transformed_bbs)

                    if self.function_check_vars[1].get():  # "상하 반전"
                        transformed_image_np, transformed_bbs = self.apply_flipud(transformed_image_np, transformed_bbs)

                    if self.function_check_vars[2].get():  # "아핀 변환"
                        params = function_params.get("아핀 변환", {})
                        scale_x = self.get_param_with_defaults(params, "scale_x", 0.8, 1.2)
                        scale_y = self.get_param_with_defaults(params, "scale_y", 0.8, 1.2)
                        translate_percent = self.get_param_with_defaults(params, "translate_percent", -0.1, 0.1)
                        rotate = self.get_param_with_defaults(params, "rotate", -45, 45)
                        shear = self.get_param_with_defaults(params, "shear", -16, 16)
                        transformed_image_np, transformed_bbs = self.apply_affine(transformed_image_np, transformed_bbs, scale_x, scale_y, translate_percent, rotate, shear)

                    if self.function_check_vars[3].get():  # "Crop and Pad"
                        params = function_params.get("확대-축소 변환", {})
                        percent = self.get_param_with_defaults(params, "percent", -0.25, 0.25)
                        transformed_image_np, transformed_bbs = self.apply_CropAndPad(transformed_image_np, transformed_bbs, percent)

                    if self.function_check_vars[4].get():  # "가우시안 노이즈 추가"
                        params = function_params.get("가우시안 노이즈", {})
                        noise = self.get_param_with_defaults(params, "noise", 0.01 * 255, 0.05 * 255)
                        transformed_image_np, transformed_bbs = self.apply_gaussian_noise(transformed_image_np, transformed_bbs, noise)

                    if self.function_check_vars[5].get():  # "명암 대비 조정"
                        params = function_params.get("명암 변경", {})
                        alpha = self.get_param_with_defaults(params, "alpha", 0.5, 2)
                        transformed_image_np, transformed_bbs = self.apply_contrast_norm(transformed_image_np, transformed_bbs, alpha)

                    if self.function_check_vars[6].get():  # "가우시안 블러"
                        params = function_params.get("가우시안 블러", {})
                        sigma = self.get_param_with_defaults(params, "sigma", 0.0, 3.0)
                        transformed_image_np, transformed_bbs = self.apply_gaussian_blur(transformed_image_np, transformed_bbs, sigma)

                    if self.function_check_vars[7].get():  # "선명도 조절"
                        params = function_params.get("선명도 변경", {})
                        alpha = self.get_param_with_defaults(params, "alpha", 0.0, 1.0)
                        lightness = self.get_param_with_defaults(params, "lightness", 0.75, 1.5)
                        transformed_image_np, transformed_bbs = self.apply_sharpen(transformed_image_np, transformed_bbs, alpha, lightness)

                    if self.function_check_vars[8].get():  # "JPEG 압축 품질 조정"
                        params = function_params.get("JPEG 압축 품질 조정", {})
                        quality = self.get_param_with_defaults(params, "quality", 70, 99)
                        transformed_image_np, transformed_bbs = self.apply_jpeg_compression(transformed_image_np, transformed_bbs, quality)

                    if self.function_check_vars[9].get():  # "채도 조정"
                        params = function_params.get("채도 변경", {})
                        mul = self.get_param_with_defaults(params, "mul", 0.5, 1.5)
                        transformed_image_np, transformed_bbs = self.apply_saturation(transformed_image_np, transformed_bbs, mul)
                    
                    print(transformed_bbs)
                    self.save_image(transformed_image_np, folder_name, image_name, i, image_extension, transformed_bbs)
 
                elif current_option == "모든 augmentation 함수를 실행":
                    augmentations = [
                        (self.apply_fliplr, []),
                        (self.apply_flipud, []),
                        (self.apply_affine, [
                            self.get_param_with_defaults(function_params.get("아핀 변환", {}), "scale_x", 0.8, 1.2), 
                            self.get_param_with_defaults(function_params.get("아핀 변환", {}), "scale_y", 0.8, 1.2), 
                            self.get_param_with_defaults(function_params.get("아핀 변환", {}), "translate_percent", -0.1, 0.1), 
                            self.get_param_with_defaults(function_params.get("아핀 변환", {}), "rotate", -45, 45), 
                            self.get_param_with_defaults(function_params.get("아핀 변환", {}), "shear", -16, 16)
                            ]),
                        (self.apply_CropAndPad, [self.get_param_with_defaults(function_params.get("확대-축소 변환", {}), "percent", -0.25, 0.25)]),
                        (self.apply_gaussian_noise, [self.get_param_with_defaults(function_params.get("가우시안 노이즈", {}), "noise", 0.01 * 255, 0.05 * 255)]),
                        (self.apply_contrast_norm, [self.get_param_with_defaults(function_params.get("명암 변경", {}), "alpha", 0.5, 2)]),
                        (self.apply_gaussian_blur, [self.get_param_with_defaults(function_params.get("가우시안 블러", {}), "sigma", 0.0, 3.0)]),
                        (self.apply_sharpen, [
                            self.get_param_with_defaults(function_params.get("선명도 변경", {}), "alpha", 0.0, 1.0),
                            self.get_param_with_defaults(function_params.get("선명도 변경", {}), "lightness", 0.75, 1.5)
                            ]),
                        (self.apply_jpeg_compression, [self.get_param_with_defaults(function_params.get("JPEG 압축 품질 조정", {}), "quality", 70, 99)]),
                        (self.apply_saturation, [self.get_param_with_defaults(function_params.get("채도 변경", {}), "mul", 0.5, 1.5)])
                    ]

                    for aug_function, params in augmentations:
                        transformed_image_np = image_np.copy()
                        transformed_bbs = bbs.copy()
                        transformed_image_np, transformed_bbs = aug_function(transformed_image_np, transformed_bbs, *params)
                        self.save_image(transformed_image_np, folder_name, image_name, i + j + 1, image_extension, transformed_bbs) 
                        j += 1

                elif current_option == "모든 경우의 수의 augmentation 함수를 실행":
                    
                    augmentations = [
                        (self.apply_fliplr, [], self.function_check_vars[0].get()),
                        (self.apply_flipud, [], self.function_check_vars[1].get()),
                        (self.apply_affine, [
                            self.get_param_with_defaults(function_params.get("아핀 변환", {}), "scale_x", 0.8, 1.2), 
                            self.get_param_with_defaults(function_params.get("아핀 변환", {}), "scale_y", 0.8, 1.2), 
                            self.get_param_with_defaults(function_params.get("아핀 변환", {}), "translate_percent", -0.1, 0.1), 
                            self.get_param_with_defaults(function_params.get("아핀 변환", {}), "rotate", -45, 45), 
                            self.get_param_with_defaults(function_params.get("아핀 변환", {}), "shear", -16, 16)
                            ], self.function_check_vars[2].get()),
                        (self.apply_CropAndPad, [self.get_param_with_defaults(function_params.get("확대-축소 변환", {}), "percent", -0.25, 0.25)], self.function_check_vars[3].get()),
                        (self.apply_gaussian_noise, [self.get_param_with_defaults(function_params.get("가우시안 노이즈", {}), 
                                                                                  "noise", 0.01 * 255, 0.05 * 255)], 
                                                                                  self.function_check_vars[4].get()),
                        (self.apply_contrast_norm, [self.get_param_with_defaults(function_params.get("명암 변경", {}), "alpha", 0.5, 2)], self.function_check_vars[5].get()),
                        (self.apply_gaussian_blur, [self.get_param_with_defaults(function_params.get("가우시안 블러", {}), "sigma", 0.0, 3.0)], self.function_check_vars[6].get()),
                        (self.apply_sharpen, [
                            self.get_param_with_defaults(function_params.get("선명도 변경", {}), "alpha", 0.0, 1.0),
                            self.get_param_with_defaults(function_params.get("선명도 변경", {}), "lightness", 0.75, 1.5)
                            ], self.function_check_vars[7].get()),
                        (self.apply_jpeg_compression, [self.get_param_with_defaults(function_params.get("JPEG 압축 품질 조정", {}), "quality", 70, 99)], self.function_check_vars[8].get()),
                        (self.apply_saturation, [self.get_param_with_defaults(function_params.get("채도 변경", {}), "mul", 0.5, 1.5)], self.function_check_vars[9].get())
                    ]

                    checked_augmentations = [aug for aug in augmentations if aug[2]]  # 각 'aug'는 (function, params, is_checked) 구조를 가집니다.

                    # 가능한 모든 조합 생성 (빈 조합 포함)
                    all_combinations = list(itertools.chain.from_iterable(itertools.combinations(checked_augmentations, r) for r in range(len(checked_augmentations)+1)))
                    
                    # 각 조합에 대해 증강 작업 수행
                    for comb in all_combinations:
                        transformed_image_np_comb = image_np.copy()  # 원본 이미지 복사본 생성
                        transformed_bbs = bbs.copy()
                        for aug_function, params, _ in comb:  # 체크 상태는 여기서는 무시합니다 (_).
                            transformed_image_np_comb, transformed_bbs = aug_function(transformed_image_np_comb, transformed_bbs, *params)
                        
                        if comb:
                            self.save_image(transformed_image_np_comb, folder_name, image_name, i + j + 1, image_extension, transformed_bbs)
                            j += 1  

    def save_image(self, image_np, folder_name, index, i, image_extension, bbs):
        augmented_image_filename = f"{index}_{i}{image_extension}"  
        save_path = os.path.join(folder_name, augmented_image_filename)
        
        if image_extension == ".jpg":
            if image_np.ndim == 2:  # 그레이스케일 이미지
                image_to_save = Image.fromarray(image_np, mode='L')
            elif image_np.shape[2] == 1:  # 단일 채널 이미지
                image_np = image_np.squeeze(-1)  # 마지막 차원 제거
                image_to_save = Image.fromarray(image_np, mode='L')  # 'L': (8-bit pixels, black and white)
            elif image_np.shape[2] == 3:  # RGB 이미지
                image_to_save = Image.fromarray(image_np, mode='RGB')  # 'RGB': (3x8-bit pixels, true color)
            elif image_np.shape[2] == 4:  # RGBA 이미지
                image_np = image_np[:, :, :3]
                image_to_save = Image.fromarray(image_np, mode='RGB')  # 'RGBA': (4x8-bit pixels, true color with transparency mask)

        image_to_save.save(save_path) 
        
        if self.label_creation_var.get() == 1 and bbs is not None:
            img_height, img_width, channel = image_np.shape

            txt_filename = os.path.splitext(augmented_image_filename)[0] + ".txt"
            txt_save_path = os.path.join(folder_name, txt_filename)

            with open(txt_save_path, "w") as txt_file:
                for bb in bbs.bounding_boxes:

                    class_id = bb.label  

                    x, y, w, h = self.to_yolo_format(bb, img_width, img_height)

                    txt_file.write(f"{class_id} {x} {y} {w} {h}\n")

            # TEST
            # # 바운딩 박스를 그린 이미지를 저장하기 위한 파일명 생성
            # augmented_image_with_bb_filename = f"{index}_{i}_with_bb{image_extension}"
            # save_path_with_bb = os.path.join(folder_name, augmented_image_with_bb_filename)

            # # 이미 저장된 PIL 이미지 객체를 NumPy 배열로 변환합니다.
            # image_np_cv2 = np.array(image_to_save)

            # if bbs is not None:
            #     for bb in bbs.bounding_boxes:
            #         # 바운딩 박스 좌표가 이미지 경계 내에 있는지 확인하고, 필요한 경우 조정
            #         x1 = int(self.clamp(bb.x1, 0, img_width))
            #         y1 = int(self.clamp(bb.y1, 0, img_height))
            #         x2 = int(self.clamp(bb.x2, 0, img_width))
            #         y2 = int(self.clamp(bb.y2, 0, img_height))

            #         # 이미지에 바운딩 박스 그리기
            #         cv2.rectangle(image_np_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 녹색 바운딩 박스

            # # 바운딩 박스가 그려진 이미지를 파일로 저장
            # cv2.imwrite(save_path_with_bb, image_np_cv2)

    def clamp(self, value, min_value, max_value):
        """주어진 값이 특정 범위 내에 있는지 확인하고, 범위를 벗어나면 조정합니다."""
        return max(min(value, max_value), min_value)

    def to_yolo_format(self, bb, img_width, img_height):

        # 바운딩 박스 좌표가 음수인 경우 0으로 설정
        bb.x1 = max(0, bb.x1)
        bb.y1 = max(0, bb.y1)

        # 바운딩 박스 좌표가 이미지 크기를 초과하는 경우 이미지 크기로 설정
        bb.x2 = min(img_width, bb.x2)
        bb.y2 = min(img_height, bb.y2)

        # 바운딩 박스의 중심 좌표 계산
        x_center = bb.x1 + (bb.x2 - bb.x1) / 2.0
        y_center = bb.y1 + (bb.y2 - bb.y1) / 2.0

        # 바운딩 박스의 너비와 높이 계산
        width = bb.x2 - bb.x1
        height = bb.y2 - bb.y1

        # YOLO 형식으로 변환 (상대적인 값)
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        print(x_center, y_center, width, height)

        if x_center > 1.0 or y_center > 1.0 or width > 1.0 or height > 1.0:
            raise ValueError("YOLO format requires normalized values less than or equal to 1.0")

        return x_center, y_center, width, height
    
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAugmentationApp(root)
    root.mainloop()
