import subprocess
import sys

# Define packages and whether to try installing via conda or pip
packages = {
    # Audio processing
    "joblib": "conda",
}

import importlib.util

def is_installed(pkg_name):
    return importlib.util.find_spec(pkg_name.replace("-", "_")) is not None

def install_package(name, method):
    print(f"Installing {name} via {method}...")
    try:
        if method == "conda":
            subprocess.check_call(["conda", "install", "-y", name])
        elif method == "pip":
            subprocess.check_call([sys.executable, "-m", "pip", "install", name])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {name}: {e}")

def main():
    for name, method in packages.items():
        module_name = name.split("==")[0]  # remove version pin if any
        if not is_installed(module_name):
            install_package(name, method)
        else:
            print(f"{name} already installed.")

if __name__ == "__main__":
    main()
