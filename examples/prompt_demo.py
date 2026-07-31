"""将 src.rag.prompt_templates 作为 examples 入口运行。"""

import runpy

if __name__ == "__main__":
    runpy.run_module("src.rag.prompt_templates", run_name="__main__")
