import subprocess
import sys

def install(package_names):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *package_names])

def get_installed_packages():
    installed_packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    installed_packages = installed_packages.decode("utf-8")
    installed_packages = [r.split("==")[0] for r in installed_packages.split()]
    return installed_packages

def main():
    required_packages = ["opencv-python", "torch", "yolov5", "Pillow", "numpy", "pygetwindow", "pywin32"]
    installed_packages = get_installed_packages()
    packages_to_install = [pkg for pkg in required_packages if pkg not in installed_packages]

    if packages_to_install:
        print("필요한 패키지를 설치합니다:", ' '.join(packages_to_install))
        install(packages_to_install)
        print("모든 패키지가 설치되었습니다.")
    else:
        print("모든 필요한 패키지가 이미 설치되어 있습니다.")

if __name__ == "__main__":
    main()
