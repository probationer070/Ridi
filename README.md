# NOTE
It is also a AI system that allows you to converse with people as a entity. code is designed to run on local llm with gguf.
Working on turning this into a package. Currently, I have only tested running it in conda.

This project is still incomplete. This code was written and executed in vscode, so it can be finally executed via `python infer_script.py`.

The gif below is just pretty, it has no meaning. (awesome chatgpt 4o img generation)

<!-- ![just pretty, it has no meaning](configs/ridiai.png) -->
<!-- ![AI](configs/ai1.png) -->

## Quick Install and Usage
I used Anaconda3 2024.10-1 (Python 3.10.16 64bit)

> [!CAUTION]
> Unzip the LangSegment library yourself and move it to site-packages.
> You need .gguf file in **model** folder

# Quick Start:

### Manual installation (old)

#### 1. Need to create conda virtual environment.

```
conda create -n <env_name> python>=3.10
conda activate <env_name>
```

#### 2. Install the required libs.
```
pip install -r requirements.txt
```

#### 3. Make sure torch is installed with CUDA enabled. Recommend to run `pip uninstall torch` to uninstall torch, then reinstall with the following. I chose 2.6.0+cu124:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 4. The **Langsegment** library currently only supports up to version 0.2.0, so you need to unzip `"LangSegment-0.3.5-py3-none-any.whl"` and manually insert it into the site-package file.
```
LangSegment-0.3.5-py3-none-any.whl
```

#### 5. Final execution
```
python infer_script.py
```


---


### Manual installation use .toml (new)

#### 1. Anaconda virtual environment must be set up.
```
conda create -n [env_name]
```

#### 2. Make sure build libs such as setuptools and wheel are installed. (latest)
```
pip install --upgrade pip build setuptools wheel
```

#### 3. perform package build
- When this command completes success, a folder called dist will be created under the project root directory, and two files will be created inside it:
    - source (e.g. Ridi-0.1.0.tar.gz - the version number is determined dynamically based on the Git tag)
    - wheel (e.g. Ridi-0.1.0-py3-none-any.whl or a platform-specific wheel)

```
python -m build
...
# when you success
Successfully built ridi-0.1.dev48+gc931b4a.d20250619.tar.gz and ridi-0.1.dev48+gc931b4a.d20250619-py3-none-any.whl
```

#### 4. special dependencies handling (Pytorch + LangSegment)
Unfortunately, it seems that req.bat is the only way to automatically configure the **conda** virtual environment and install pytorch.

> [!NOTE]
> Will be added in the future if Docker is supported

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
LangSegment requires you to manually overwrite the file after pip install or directly unpack the .whl file of a specific version into site-packages.
```
LangSegment-0.3.5-py3-none-any.whl
```
---



### Automatic installation

1. Just click on the `req.bat` file and various libs and pytorch CUDA version will be installed automatically. However, as above, **LangSegment** must be manually added to site-packages. (Automatically set the initial virtual environment name to mini from `environment.yml`, install the basic libs, install `torch+cu124`, `llama-cpp-python`, and finally reconfirm by installing `requirements.txt`)
```
LangSegment-0.3.5-py3-none-any.whl
```
```
conda activate mini
```

2. Final execution
```
python infer_script.py
```



## Credits

Special thanks to the RVC-Boss for getting this wonderful tool up and going, as well as all of the other attributions used to build it:

> [!Note]
> welcome any suggestions for improvement.

**Original Repo:** https://github.com/RVC-Boss/GPT-SoVITS
