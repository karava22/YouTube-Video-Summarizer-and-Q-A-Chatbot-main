import subprocess
import sys


def main() -> None:
    # Legacy launcher: keeps python youtube.py working while app.py is the single app file.
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=False)


if __name__ == "__main__":
    main()