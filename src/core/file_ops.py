from pathlib import Path

path = Path("data")
# 1. 创建目录 data/（如果不存在）
# 提示：Path("data").mkdir(...)
if not path.exists():
    path.mkdir(parents=True, exist_ok=True)


# 2. 写一个函数 save_note(filename, content)，把内容写入文件
def save_note(filename: str, content: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


save_note("data/note.txt", "这是第一行笔记\n第二行笔记")


# 3. 写一个函数 read_note(filename)，读取文件并返回内容，如果文件不存在返回 "文件不存在"
def read_note(filename) -> str:
    try:
        with open(filename, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "文件不存在"


# 4. 写一个函数 append_note(filename, content)，追加一行内容
def append_note(filename, content):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(content)


append_note("data/note.txt", "\n这是追加的第三行")

# 5. 读取完整文件内容，按行打印（每行前面加行号）
# 预期输出：
# 1: 这是第一行笔记
# 2: 第二行笔记
# 3: 这是追加的第三行


def read_step(filename):
    try:
        with open(filename, encoding="utf-8") as f:
            lines: list[str] = []
            for idx, line in enumerate(f, start=1):
                lines.append(f"{idx} : {line.strip()}")
                # print(f"{idx} : {line.strip()}")
            return lines
    except FileNotFoundError:
        return "文件不存在"


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("除数不能为0")
    return a / b


def file_exists(filename: str) -> bool:
    return Path(filename).exists()


if __name__ == "__main__":
    print(read_note("data/note.txt"))
    print(read_step("data/note.txt"))
    # 只有直接运行此文件时才执行
