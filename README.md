# Autogluon-GPU

This repo contain very basic scripts to get [Autogluon]() up an running on multiple 
CPU(s) and GPU(s). 

## Setup

### 🧕 Environment

1. Create the python virtual environment and activate it:
    ```
    python3.9 -m venv .venv
    source .venv/bin/activate
    ```

2. Install dependencies:
    ```
    poetry install
    ```
    In particular, this installs the prerequisite (`torch`) for running Autogluon with GPU:
    ```
    # in pyproject.toml
    torch = {url="https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp39-cp39-linux_x86_64.whl"}
    autogluon-tabular = {extras = ["all"], version = "~0.6.0"}
    ```
    You must check that `torch` is compatible with your cuda version. See [this](https://download.pytorch.org/whl/cu113/torch/) for available packages.

3. [Optional] Install LightGBM for GPU - the current LightGBM installation that
    comes with Autogluon doesn't have GPU capabilities. To enable this, we need need to
    uninstall this, and reinstall with `--install-option=--gpu`.
    ```
    # uninstall autogluon's lightgbm installation, and install the new build
    pip uninstall lightgbm -y
    pip install lightgbm --install-option=--gpu
    ```

## Run the script

### 🧕 Using [runner.sh](runner.sh)

To run the [basic-autogluon.py](basic-autogluon.py) script, do:

```
./runner.sh
```

[runner.sh](runner.sh) does a couple of things:

1. Set the `CUDA_VISIBLE_DEVICES` environment variable to assign GPUs to be used.
2. Set the CPU affininity for `taskset`, which is a linux utility that can assign CPUs
    to our python application. Autogluon does not limit the number of CPUs it uses
    by default - so to avoid hogging the server resources, we limit them.
3. Runs the [basic-autogluon.py](basic-autogluon.py) script that contains our main codes
    with `taskset`.

### 🧕 Using VSCode's Debugging Facilities

This repo comes with [.vscode/launch.json] that has `CUDA_VISIBLE_DEVICES` set up.

Just hit `F5` while you are on [basic-autogluon.py] to run the debugger.

> Idk how to allocate CPU with the debugger yet. If you know how to do that, feel
> free to submit a pull request :p


