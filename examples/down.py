"""将 src.down 作为 examples 入口运行。"""

import runpy

if __name__ == "__main__":
    runpy.run_module("src.down", run_name="__main__")
