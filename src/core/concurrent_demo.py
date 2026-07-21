"""
并发处理演示
批量处理PDF文件,展示顺序 vs 并非的时间差异
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.core.pdf_processor import extract_full_text, get_pdf_metadata


def process_pdf(pdf_path: str) -> dict:
    """处理单个PDF: 提取文本和元数据"""
    print(f"⏳ 开始处理: {pdf_path}")
    start_time = time.time()

    try:
        metadata = get_pdf_metadata(pdf_path)
        text = extract_full_text(pdf_path)
        result = {
            "file": pdf_path,
            "metadata": metadata,
            "text_length": len(text),
            "pages": metadata.get("pages", 0),
            "status": "success",
        }
    except Exception as e:
        result = {
            "file": pdf_path,
            "status": "failed",
            "error": str(e),
        }

    elapsed = time.time() - start_time
    print(f"✅ 完成处理: {pdf_path} (耗时: {elapsed:.2f}s)")
    return result


def process_sequential(pdf_files: list[str]) -> list[dict]:
    """顺序处理"""
    results: list = []
    for pdf in pdf_files:
        results.append(process_pdf(pdf))
    return results


def process_concurrent(pdf_files: list[str], max_workers: int = 4) -> list[dict]:
    """并发处理"""
    results: list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_files}

        # 收集结果
        for future in as_completed(future_to_file):
            results.append(future.result())

    return results


def generate_sample_pdfs():
    """生成多个测试 PDF 文件（用于展示）"""

    Path("data/concurrent_test").mkdir(parents=True, exist_ok=True)

    for i in range(1, 6):
        pdf_path = f"data/concurrent_test/doc_{i}.pdf"
        if not Path(pdf_path).exists():
            c = canvas.Canvas(pdf_path, pagesize=A4)
            c.setFont("Helvetica", 20)
            c.drawString(100, 800, f"Wold {i}")
            c.setFont("Helvetica", 14)
            c.drawString(100, 750, f"This is the {i} XXX PDF ")
            c.drawString(100, 720, f"Include {i+1} Text。")
            c.showPage()
            c.save()
            print(f"✅ 创建: {pdf_path}")

    return [f"data/concurrent_test/doc_{i}.pdf" for i in range(1, 6)]


if __name__ == "__main__":
    print("=" * 60)
    print("📄 并发处理演示")
    print("=" * 60)

    # 生成测试文件
    pdf_files = generate_sample_pdfs()

    print(f"\n📁 共 {len(pdf_files)} 个 PDF 文件待处理\n")

    # 1. 顺序处理
    print("--- 顺序处理 ---")
    start = time.time()
    seq_results = process_sequential(pdf_files)
    seq_time = time.time() - start

    # 2. 并发处理
    print("\n--- 并发处理 ---")
    start = time.time()
    conc_results = process_concurrent(pdf_files, max_workers=4)
    conc_time = time.time() - start

    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 性能对比")
    print("=" * 60)
    print(f"  顺序处理耗时: {seq_time:.2f} 秒")
    print(f"  并发处理耗时: {conc_time:.2f} 秒")
    print(f"  速度提升: {seq_time / conc_time:.1f}x")

    # 验证结果正确性
    seq_success = sum(1 for r in seq_results if r.get("status") == "success")
    conc_success = sum(1 for r in conc_results if r.get("status") == "success")
    print(f"\n  顺序处理成功: {seq_success}/{len(pdf_files)}")
    print(f"  并发处理成功: {conc_success}/{len(pdf_files)}")

    print("\n✅ 并发处理演示完成!")
