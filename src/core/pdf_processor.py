"""
PDF 处理模块
提取PDF文本、元数据、支持分页输出
"""

import json
from pathlib import Path

# from pypdf import PdfReader
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """从PDF提取文本, 按页返回

    Args:
        pdf_path: PDF 文件路径

    Returens:
        每页的文本和元数据列表
    """
    # reader = PdfReader(pdf_path)
    result: list = []
    with pdfplumber.open(pdf_path) as pdf:

        # 提取元数据
        metadata = pdf.metadata
        if metadata:
            metadata_dict = {
                "title": metadata.get("/Title", ""),
                "author": metadata.get("/Author", ""),
                "creator": metadata.get("Creator", ""),
                "producer": metadata.get("/Producer", ""),
                "pages": len(pdf.pages),
            }
        else:
            metadata_dict = {"Pages": len(pdf.pages)}

        # 逐页提取文本
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                result.append(
                    {"page": page_num, "text": text.strip(), "metadata": metadata_dict}
                )
    return result


def extract_full_text(pdf_path: str) -> str:
    """提取整个PDF完整文本"""
    full_text: list = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n\n".join(full_text)


def get_pdf_metadata(pdf_path: str) -> dict:
    """获取 PDF 元数据"""
    with pdfplumber.open(pdf_path) as pdf:
        meta = pdf.metadata
        if meta:
            return {
                "title": meta.get("/Title", ""),
                "author": meta.get("/Author", ""),
                "creator": meta.get("Creator", ""),
                "producer": meta.get("/Producer", ""),
                "pages": len(pdf.pages),
            }
    return {"pages": len(pdf.pages)}


def save_extracted_text(pdf_path: str, output_path: str | None = None) -> None:
    """提取 PDF 文本并保存到文件"""
    if output_path is None:
        pdf_name = Path(pdf_path).stem
        output_path = f"data/{pdf_name}_extracted.txt"

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    full_text = extract_full_text(pdf_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✅ 文本已保存到: {output_path}")
    print(f"📄 总字符数: {len(full_text)}")


def save_extracted_json(pdf_path: str, output_path: str | None = None) -> None:
    """提取 PDF 文本并保存为 JSON 格式（含元数据）"""
    if output_path is None:
        pdf_name = Path(pdf_path).stem
        output_path = f"data/{pdf_name}_extracted.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pages = extract_text_from_pdf(pdf_path)
    metadata = get_pdf_metadata(pdf_path)

    output = {
        "source": pdf_path,
        "metadata": metadata,
        "pages": pages,
        "total_pages": len(pages),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON 已保存到: {output_path}")
    print(f"📄 总页数: {len(pages)}")


if __name__ == "__main__":
    # 创建测试用的 PDF 文件（如果没有）
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    # 创建一个简单的测试 PDF
    test_pdf_path = "data/sample.pdf"

    if not Path(test_pdf_path).exists():
        Path("data").mkdir(exist_ok=True)
        c = canvas.Canvas(test_pdf_path, pagesize=A4)
        c.setFont("Helvetica", 24)
        c.drawString(100, 800, "Test PDF ")
        c.setFont("Helvetica", 14)
        c.drawString(
            100,
            750,
            "This is the content of page 1, used to test PDF extraction functionality.",
        )
        c.drawString(
            100, 720, "Page 1 also has more text to verify paginated extraction."
        )
        c.showPage()

        c.setFont("Helvetica", 24)
        c.drawString(100, 800, "Page 2")
        c.setFont("Helvetica", 14)
        c.drawString(100, 750, "This is the content of page 2.")
        c.drawString(
            100, 720, "Contains some sample text for testing multi-page extraction."
        )
        c.drawString(100, 690, "And some additional content.")
        c.showPage()
        c.save()
        print(f"✅ 测试 PDF 已创建: {test_pdf_path}")

    print("=" * 60)
    print("📄 PDF 处理测试")
    print("=" * 60)

    # 1. 获取元数据
    print("\n📋 元数据:")
    metadata = get_pdf_metadata(test_pdf_path)
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # 2. 提取完整文本
    print("\n📝 完整文本:")
    full_text = extract_full_text(test_pdf_path)
    print(f"  {full_text[:200]}...")
    print(f"  总字符数: {len(full_text)}")

    # 3. 按页提取
    print("\n📖 按页提取:")
    pages = extract_text_from_pdf(test_pdf_path)
    for page in pages:
        print(f"  第 {page['page']} 页: {page['text'][:100]}...")

    # 4. 保存为文本文件
    print("\n💾 保存结果:")
    save_extracted_text(test_pdf_path)
    save_extracted_json(test_pdf_path)

    print("\n✅ PDF 处理测试完成!")
