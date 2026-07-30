"""
文本切分模块
支持固定大小切分和语义切分
"""

import re


class TextChunker:
    """文本切分器"""

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
    ):
        """
        初始化切分器

        Args:
            chunk_size: 每个chunk的目标字符
            chunk_overlap: chunk之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)

    def fixed_size_chunk(self, text: str) -> list[str]:
        """
        固定大小切分(优化版)

        Args:
            text: 输入文本

        Returns:
            切分后的chunk列表
        """
        # 清理文本：去除多余空白
        if not text:
            return []

        # 去掉文本首尾的空格、换行符。
        text = text.strip()

        # 如果文本长度小于 chunk_size, 直接返回整个文本
        if len(text) <= self.chunk_size:
            return [text]

        # 初始化变量
        chunks = []
        start = 0
        text_len = len(text)
        step = self.chunk_size - self.chunk_overlap  # 每次前进的步长

        # ✅ 安全阀：最大迭代次数 = 文本长度 / 步长 + 10
        max_iterations = (text_len // step + 10) * 2
        iteration = 0

        # 开始遍历整个文本
        while start < text_len and iteration < max_iterations:
            iteration += 1
            # 1. 计算结束位置
            end = min(start + self.chunk_size, text_len)
            # 2. 尝试在句子边界断开
            """
            1.如果 end 还没到文本结尾
            2.从 end 位置往前找（range(end, start, -1)）
            3.如果找到 。！？.!? 中的任意一个字符
            4.把 end 移到这个句子的结尾位置
            """
            if end < text_len:
                for i in range(end, start, -1):
                    if text[i - 1] in "。！？.!?":
                        end = i
                        break

            # 3. ✅ 安全保证：如果 end <= start，强制前进
            # ✅ 关键修复：确保 end > start
            # 强制前进 chunk_size 个字符，防止无限循环。
            if end <= start:
                end = min(start + self.chunk_size, text_len)

            # 4. 提取 chunk
            chunk = text[start:end].strip()
            if chunk:  # 只添加非空 chunk
                chunks.append(chunk)

            # 5. ✅ 关键：先前进 step，再回退 overlap
            # 这样确保 start 总体是递增的
            new_start = start + step

            # 6. ✅ 如果 new_start 超过了 end，但还没到结尾
            #    回退一点保持重叠，但不能小于 end - overlap
            if new_start > end and new_start < text_len:
                new_start = end - self.chunk_overlap

            # 7. ✅ 最终安全保证：new_start 必须 > start
            if new_start <= start:
                new_start = start + 1  # 至少前进 1 个字符

            start = new_start

        return chunks

    def semantic_chunk(self, text: str) -> list[str]:
        """
        语义切分：按段落和句子切分

        Args:
            text: 输入文本

        Returns:
            切分后的chunk列表
        """
        # 按段落切分
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk: list[str] = []

        for raw_para in paragraphs:
            para = raw_para.strip()  # 去除字符串两端的空白字符
            if not para:
                continue

            # 如果当前段落长度超过了 chunk_size, 递归处理
            if len(para) > self.chunk_size:
                # 按句子进一步切分
                sentence = re.split(r"(?<=[。！？.!?])\s*", para)
                for sent in sentence:
                    if not sent:
                        continue
                    if len("".join(current_chunk)) + len(sent) <= self.chunk_size:
                        current_chunk.append(sent)
                    else:
                        if current_chunk:
                            chunks.append("".join(current_chunk).strip())
                        current_chunk = [sent]
            else:
                if len("".join(current_chunk)) + len(para) <= self.chunk_size:
                    current_chunk.append(para)
                else:
                    if current_chunk:
                        chunks.append("".join(current_chunk).strip())
                    current_chunk = [para]

        if current_chunk:
            chunks.append("".join(current_chunk).strip())

        return chunks


def print_chunks(chunks: list[str], title: str = "切割结果"):
    """打印切分结果"""
    print(f"\n{'=' * 60}")
    print(f"📌 {title}")
    print(f"{'=' * 60}")
    print(f"共 {len(chunks)} 个 chunk\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1} (长度: {len(chunk)} 字符):")
        print(f"  {chunk[:100]}{'...' if len(chunk) > 100 else ''}")
        print()
