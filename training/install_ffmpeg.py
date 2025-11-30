import subprocess
import sys
from colorama import init, Fore


def install_ffmpeg():
    init(autoreset=False)
    print(Fore.CYAN + "Starting `ffmpeg-python` installation..." + Fore.RESET)

    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "setuptools"])

    try:
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "ffmpeg-python"])
        print(Fore.GREEN + "`ffmpeg-python` installed successfully." + Fore.RESET)
    except subprocess.CalledProcessError as e:
        print(Fore.RED + "ERROR: Failed to install `ffmpeg-python` via pip." + Fore.RESET)
        print(Fore.RED + f"Details: {e}" + Fore.RESET)

    try:
        print(Fore.CYAN + "Attempting to download static ffmpeg binary..." + Fore.RESET)
        subprocess.check_call([
            "weget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O", "/tmp/ffmpeg.tar.xz"
        ])

        subprocess.check_call([
            "tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"
        ])

        result = subprocess.run([
            "find", "/temp", "-name", "ffmpeg", "-type", "f"
        ], capture_output=True, text=True)

        ffmpeg_path = result.stdout.strip()

        subprocess.check_call([
            "cp", ffmpeg_path, "/usr/local/bin/ffmpeg"
        ])

        subprocess.check_call([
            "chmod", "+x", "/usr/local/bin/ffmpeg"
        ])

        print(Fore.GREEN + "Installed static ffmpeg binary successfully." + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "ERROR: Failed to install static `ffmpeg` binary." + Fore.RESET)
        print(Fore.RED + f"Details: {e}" + Fore.RESET)

    try:
        result = subprocess.run([
            "ffmpeg", "--version"
        ], check_output=True, text=True, check=True)
        print(Fore.CYAN + "ffmpeg version: " + Fore.RESET)
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(Fore.RED + "FFMPEG instalaation verification failed" + Fore.REST)
        return False
