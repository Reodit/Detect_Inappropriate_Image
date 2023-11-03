# Detect_Inappropriate_Image
Using YOLOv5 to detect inappropriate images (Hamas symbols)

## Installation

**파이썬 3.7이상 버전이 필요합니다.
1. setup.py 를 실행하여 필요 패키지를 다운받습니다.

## Augmentation

1. setup.py 를 실행하여 필요 패키지를 다운 받은 후 img_augmentaion.py를 실행합니다.
2. 좌측 상단 파일 - 이미지 파일을 선택합니다 (다중 선택 가능)
3. 좌측 상단 이미지 증강 - 증강하기 버튼을 클릭합니다
4. 증강하려는 이미지를 좌측 박스에서 클릭한 후 증강하고자 하는 함수를 선택해서 증강합니다.
5. 레이블 데이터가 있을 경우, 이미지가 있는 폴더에 TXT 레이블 파일을 이미지와 동일한 위치에 놓은 후, 레이블 포함 생성하기에 체크하면 레이블 데이터도 자동으로 생성합니다.
6. 단, 레이블 데이터는 YOLO 형식이어야 합니다. (상대좌표)


## Object Detection

1. 모든 패키지를 설치했으면 screen_capture.py를 실행합니다
2. cmd 창에서 캡쳐하고자 하는 윈도우의 번호를 입력합니다. 단, 주 모니터에 있어야 합니다.
3. 현재 하마스 심볼만 학습이 되어있는 상태이므로, 하마스만 검출합니다다