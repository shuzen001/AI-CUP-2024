from pathlib import Path
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.settings import settings
import os
import concurrent.futures


# ------------------------------------------------------------
# Helper: convert scanned PDF to Markdown via Docling pipeline
# ------------------------------------------------------------
def pdf_to_markdown(pdf_path: str, md_output_path: str):
    """
    將單一 PDF 轉換為 Markdown。
    - 使用繁體中文 + 英文 OCR
    - 保留表格結構 (cell matching)
    - GPU 加速 (CUDA)
    轉換後寫入 md_output_path；若檔案已存在則直接返回。
    """
    if os.path.exists(md_output_path):
        return

    accelerator_options = AcceleratorOptions(
        num_threads=20, device=AcceleratorDevice.CUDA
    )
    ocr_options = TesseractCliOcrOptions(lang=["chi_tra", "eng"])

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        force_full_page_ocr=True,
        ocr_options=ocr_options,
    )
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    settings.debug.profile_pipeline_timings = False  # 關閉詳細 profiling

    conversion_result = converter.convert(Path(pdf_path))
    md_text = conversion_result.document.export_to_markdown()

    # 確保輸出資料夾存在
    os.makedirs(os.path.dirname(md_output_path), exist_ok=True)
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

# Helper: bulk PDF to Markdown conversion with parallelism
def bulk_pdf_to_markdown(tasks, max_workers: int = 2):
    """
    tasks: List[Tuple[str pdf_path, str md_output_path]]
    將多個 PDF 以 ProcessPoolExecutor 並行轉 Markdown。
    注意：若只有單張 GPU，並行數請≤2 以免 CUDA 記憶體衝突。
    """
    if not tasks:
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(lambda pair: pdf_to_markdown(*pair), tasks))
