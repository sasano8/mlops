import sys

args = sys.argv

from mlops import run

if __name__ == "__main__":
    run(args[1], "main")
