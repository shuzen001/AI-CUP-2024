#!/usr/bin/env python3
"""
重試失敗的 PDF 轉換工具
讀取之前失敗的文件列表並重新嘗試轉換
"""

import os
import json
import logging
from typing import List, Dict
from bulk_convert_pdfs_enhanced import bulk_convert_pdfs, ConversionResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_failed_files_from_report(report_path: str) -> List[str]:
    """從轉換報告中加載失敗的文件列表"""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        failed_files = [item['pdf_path'] for item in report.get('failed_files', [])]
        logger.info(f"從報告中找到 {len(failed_files)} 個失敗文件")
        return failed_files
        
    except Exception as e:
        logger.error(f"無法讀取報告文件 {report_path}: {e}")
        return []

def load_failed_files_from_list(failed_list_path: str) -> List[str]:
    """從失敗文件列表中加載文件路徑"""
    try:
        failed_files = []
        with open(failed_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    # 取第一欄（文件路徑）
                    file_path = line.strip().split('\t')[0]
                    failed_files.append(file_path)
        
        logger.info(f"從列表中找到 {len(failed_files)} 個失敗文件")
        return failed_files
        
    except Exception as e:
        logger.error(f"無法讀取失敗文件列表 {failed_list_path}: {e}")
        return []

def create_temp_directory_structure(failed_files: List[str], input_dir: str, temp_dir: str):
    """創建臨時目錄結構，只包含失敗的文件"""
    import shutil
    
    os.makedirs(temp_dir, exist_ok=True)
    
    copied_files = []
    for pdf_path in failed_files:
        if os.path.exists(pdf_path):
            # 計算相對路徑
            rel_path = os.path.relpath(pdf_path, input_dir)
            temp_pdf_path = os.path.join(temp_dir, rel_path)
            
            # 創建必要的目錄
            os.makedirs(os.path.dirname(temp_pdf_path), exist_ok=True)
            
            # 創建符號連結而不是複製文件（節省空間）
            try:
                os.symlink(pdf_path, temp_pdf_path)
                copied_files.append(temp_pdf_path)
            except OSError:
                # 如果符號連結失敗，則複製文件
                shutil.copy2(pdf_path, temp_pdf_path)
                copied_files.append(temp_pdf_path)
        else:
            logger.warning(f"文件不存在，跳過: {pdf_path}")
    
    logger.info(f"創建了包含 {len(copied_files)} 個文件的臨時目錄")
    return copied_files

def retry_failed_conversions(
    failed_source: str,
    input_dir: str,
    output_dir: str,
    max_workers: int = 2,
    cleanup_temp: bool = True
) -> Dict:
    """
    重試失敗的轉換
    
    Args:
        failed_source: 失敗文件來源（報告文件或失敗列表文件）
        input_dir: 原始輸入目錄
        output_dir: 輸出目錄
        max_workers: 工作進程數（重試時建議較少）
        cleanup_temp: 是否清理臨時目錄
    """
    logger.info(f"開始重試失敗的轉換...")
    logger.info(f"失敗文件來源: {failed_source}")
    
    # 載入失敗文件列表
    if failed_source.endswith('.json'):
        failed_files = load_failed_files_from_report(failed_source)
    else:
        failed_files = load_failed_files_from_list(failed_source)
    
    if not failed_files:
        logger.warning("沒有找到需要重試的文件")
        return {'error': 'No failed files found'}
    
    # 檢查文件是否存在
    existing_files = [f for f in failed_files if os.path.exists(f)]
    if len(existing_files) != len(failed_files):
        missing_count = len(failed_files) - len(existing_files)
        logger.warning(f"{missing_count} 個文件不存在，將跳過")
    
    if not existing_files:
        logger.error("所有失敗文件都不存在")
        return {'error': 'All failed files missing'}
    
    # 創建臨時目錄
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix='retry_conversion_')
    
    try:
        # 創建臨時目錄結構
        create_temp_directory_structure(existing_files, input_dir, temp_dir)
        
        # 執行重試轉換
        logger.info(f"使用 {max_workers} 個工作進程重試轉換...")
        
        result = bulk_convert_pdfs(
            input_dir=temp_dir,
            output_dir=output_dir,
            max_workers=max_workers,
            force_convert=True,  # 強制重新轉換
            save_reports=True,
            log_file=os.path.join(output_dir, f'retry_{os.path.basename(failed_source)}.log')
        )
        
        return result
        
    finally:
        # 清理臨時目錄
        if cleanup_temp:
            import shutil
            try:
                shutil.rmtree(temp_dir)
                logger.info("已清理臨時目錄")
            except Exception as e:
                logger.warning(f"清理臨時目錄失敗: {e}")

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="重試失敗的 PDF 轉換")
    parser.add_argument("failed_source", help="失敗文件來源（conversion_report.json 或 failed_files.txt）")
    parser.add_argument("input_dir", help="原始輸入PDF目錄")
    parser.add_argument("output_dir", help="輸出Markdown目錄")
    parser.add_argument("--workers", "-w", type=int, default=2, help="工作進程數 (預設: 2，重試時建議較少)")
    parser.add_argument("--keep-temp", action="store_true", help="保留臨時目錄（用於調試）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.failed_source):
        print(f"❌ 錯誤: 失敗文件來源不存在 {args.failed_source}")
        return
    
    if not os.path.exists(args.input_dir):
        print(f"❌ 錯誤: 輸入目錄不存在 {args.input_dir}")
        return
    
    # 執行重試
    result = retry_failed_conversions(
        failed_source=args.failed_source,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.workers,
        cleanup_temp=not args.keep_temp
    )
    
    if 'error' not in result:
        print(f"\n🎉 重試完成！")
        print(f"成功: {result.get('successful', 0)} 個文件")
        print(f"失敗: {result.get('failed', 0)} 個文件")
        print(f"成功率: {result.get('success_rate', 0):.1f}%")

if __name__ == "__main__":
    main() 