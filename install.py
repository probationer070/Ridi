import subprocess
import sys
import re

# 설치할 CUDA 버전을 12.4로 고정
TARGET_CUDA_VERSION = "12.4"
TARGET_CUDA_VERSION_STR = "124"

def check_cuda_version():
    """시스템에 설치된 CUDA 버전을 확인하고, 설치할 CUDA 버전과 비교합니다."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        output = result.stdout
        print(f"nvidia-smi output: \n{output}")

        # 정규 표현식을 사용하여 CUDA 버전 추출
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", output)
        if match:
            detected_cuda_version = match.group(1)
            print(f"[INSTALL.PY] Detected CUDA version: {detected_cuda_version}")

            if float(detected_cuda_version) < float(TARGET_CUDA_VERSION):
                print(f"[INSTALL.PY] Warning: Detected CUDA version ({detected_cuda_version}) is lower than the required CUDA version ({TARGET_CUDA_VERSION}).")
                print("[INSTALL.PY] Installation will be stopped. Please update your CUDA driver to version 12.4 or higher.")
                return False  # 설치 중단
            else:
                print(f"[INSTALL.PY] CUDA version {detected_cuda_version} is compatible with the required CUDA version {TARGET_CUDA_VERSION}.")
                return True  # 설치 진행
        else:
            print("[INSTALL.PY] CUDA version not found in nvidia-smi output.")
            return False

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[INSTALL.PY] CUDA not found. Installation will be stopped.")
        return False  # 설치 중단

def install_torch_packages():
    """torch, torchvision, torchaudio 패키지를 설치합니다."""
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        f"https://download.pytorch.org/whl/cu{TARGET_CUDA_VERSION_STR}"
    ]
    try:
        print(f"[INSTALL.PY] Installing torch, torchvision, torchaudio with CUDA {TARGET_CUDA_VERSION} support...")
        subprocess.run(command, check=True)
        print("[INSTALL.PY] torch, torchvision, torchaudio installation complete.")
        return True  # 설치 성공
    except subprocess.CalledProcessError as e:
        print(f"[INSTALL.PY] Error during torch, torchvision, torchaudio installation: {e}")
        return False  # 설치 실패

def main():
    """메인 함수: CUDA 버전 확인 및 torch 패키지 설치."""
    if not check_cuda_version():
        print("[INSTALL.PY] Installation aborted due to incompatible CUDA version.")
        sys.exit(1)  # 설치 실패 시 1을 반환

    if install_torch_packages():
        print("[INSTALL.PY] Installation process completed successfully.")
        sys.exit(0)  # 설치 성공 시 0을 반환
    else:
        print("[INSTALL.PY] Installation process failed.")
        sys.exit(1)  # 설치 실패 시 1을 반환

if __name__ == "__main__":
    main()
