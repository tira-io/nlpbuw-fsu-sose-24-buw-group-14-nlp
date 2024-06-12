import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from main import main

if __name__ == "__main__":
    main()
