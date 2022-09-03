import sys
import importlib
args = sys.argv

if __name__ == "__main__":
    name = args[1] + ".main"
    print(f"run: {name}")
    mod = importlib.import_module(name)
    print(mod.init())
