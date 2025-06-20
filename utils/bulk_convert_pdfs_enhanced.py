#!/usr/bin/env python3
"""
增強版 PDF 批量轉換工具
提供更好的進度追蹤、錯誤處理、統計報告和恢復機制
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import multiprocessing
from pdf_to_md import pdf_to_markdown
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# 設置多進程啟動方法為 'spawn' 以解決 CUDA 問題
multiprocessing.set_start_method('spawn', force=True)

# 設置增強日誌
def setup_logging(log_file: Optional[str] = None):
    """設置詳細的日誌系統"""
    log_format = '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    
    # 控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    handlers = [console_handler]
    
    # 文件處理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=handlers
    )

@dataclass
class ConversionResult:
    """轉換結果數據類"""
    pdf_path: str
    md_path: str
    success: bool
    error_message: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    file_size: int = 0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0.0

class ProgressTracker:
    """進度追蹤器"""
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = time.time()
        self.failed_files = []
        self.success_files = []
        
    def update(self, result: ConversionResult):
        """更新進度"""
        self.completed += 1
        if result.success:
            self.successful += 1
            self.success_files.append(result)
        else:
            self.failed += 1
            self.failed_files.append(result)
    
    def get_stats(self) -> Dict:
        """獲取統計信息"""
        elapsed = time.time() - self.start_time
        avg_time = sum(r.duration for r in self.success_files) / len(self.success_files) if self.success_files else 0
        
        return {
            'total': self.total_files,
            'completed': self.completed,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate': self.successful / self.completed * 100 if self.completed > 0 else 0,
            'elapsed_time': elapsed,
            'estimated_remaining': (self.total_files - self.completed) * avg_time if avg_time > 0 else 0,
            'avg_processing_time': avg_time,
            'files_per_minute': self.completed / (elapsed / 60) if elapsed > 0 else 0
        }

def find_pdf_files(input_dir: str) -> List[str]:
    """遞迴搜尋目錄中的所有PDF文件，並排序"""
    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    # 按文件名排序以便預測處理順序
    return sorted(pdf_files)

def create_output_path(pdf_path: str, input_dir: str, output_base_dir: str) -> str:
    """根據PDF路徑創建對應的Markdown輸出路徑（修復路徑問題）"""
    # 獲取相對於輸入目錄的路徑
    rel_path = os.path.relpath(pdf_path, input_dir)
    # 將.pdf替換為.md
    md_path = rel_path.replace('.pdf', '.md')
    # 組合完整的輸出路徑
    output_path = os.path.join(output_base_dir, md_path)
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    return output_path

def should_skip_file(pdf_path: str, md_path: str, force_convert: bool = False) -> bool:
    """檢查是否應該跳過已轉換的文件"""
    if force_convert:
        return False
    
    if os.path.exists(md_path):
        # 檢查文件大小（避免空文件）
        if os.path.getsize(md_path) > 100:  # 至少100字節
            return True
    
    return False

def convert_single_pdf(args: Tuple[str, str, str, bool]) -> ConversionResult:
    """單個PDF轉換任務的包裝函數（增強版）"""
    pdf_path, md_path, input_dir, force_convert = args
    start_time = time.time()
    
    result = ConversionResult(
        pdf_path=pdf_path,
        md_path=md_path,
        success=False,
        start_time=start_time
    )
    
    try:
        # 獲取文件大小
        result.file_size = os.path.getsize(pdf_path)
        
        # 檢查是否跳過
        if should_skip_file(pdf_path, md_path, force_convert):
            result.success = True
            result.error_message = "已存在，跳過轉換"
            result.end_time = time.time()
            return result
        
        # 執行轉換
        pdf_to_markdown(pdf_path, md_path)
        
        # 驗證輸出文件
        if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
            result.success = True
        else:
            result.error_message = "輸出文件為空或不存在"
            
    except FileNotFoundError as e:
        result.error_message = f"文件未找到: {str(e)}"
    except PermissionError as e:
        result.error_message = f"權限錯誤: {str(e)}"
    except Exception as e:
        result.error_message = f"轉換錯誤: {str(e)}"
    
    result.end_time = time.time()
    return result

def save_progress_report(output_dir: str, tracker: ProgressTracker, failed_results: List[ConversionResult]):
    """保存進度報告"""
    report_path = os.path.join(output_dir, 'conversion_report.json')
    
    stats = tracker.get_stats()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'statistics': stats,
        'failed_files': [
            {
                'pdf_path': r.pdf_path,
                'error': r.error_message,
                'duration': r.duration,
                'file_size': r.file_size
            }
            for r in failed_results
        ],
        'successful_files': [
            {
                'pdf_path': r.pdf_path,
                'md_path': r.md_path,
                'duration': r.duration,
                'file_size': r.file_size
            }
            for r in tracker.success_files
        ]
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report_path

def save_failed_list(output_dir: str, failed_results: List[ConversionResult]):
    """保存失敗文件列表，用於重試"""
    failed_list_path = os.path.join(output_dir, 'failed_files.txt')
    
    with open(failed_list_path, 'w', encoding='utf-8') as f:
        for result in failed_results:
            f.write(f"{result.pdf_path}\t{result.error_message}\n")
    
    return failed_list_path

def bulk_convert_pdfs(
    input_dir: str, 
    output_dir: str, 
    max_workers: int = 6,
    force_convert: bool = False,
    save_reports: bool = True,
    log_file: Optional[str] = None
) -> Dict:
    """
    增強版批量轉換PDF文件到Markdown
    
    Args:
        input_dir: 輸入PDF文件目錄
        output_dir: 輸出Markdown文件目錄
        max_workers: 並行處理的工作進程數
        force_convert: 是否強制重新轉換已存在的文件
        save_reports: 是否保存詳細報告
        log_file: 日誌文件路徑
    
    Returns:
        包含統計信息的字典
    """
    # 確保輸出目錄存在（必須在設置日誌之前）
    os.makedirs(output_dir, exist_ok=True)
    
    # 設置日誌
    if log_file is None:
        log_file = os.path.join(output_dir, f'conversion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    # 獲取所有PDF文件
    logger.info("🔍 掃描PDF文件...")
    pdf_files = find_pdf_files(input_dir)
    if not pdf_files:
        logger.warning(f"在 {input_dir} 中未找到PDF文件")
        return {'error': 'No PDF files found'}
    
    logger.info(f"找到 {len(pdf_files)} 個PDF文件")
    
    # 創建轉換任務列表
    conversion_tasks = []
    skipped_count = 0
    
    for pdf_path in pdf_files:
        md_path = create_output_path(pdf_path, input_dir, output_dir)
        
        # 預檢查跳過的文件
        if not force_convert and should_skip_file(pdf_path, md_path, False):
            skipped_count += 1
            logger.debug(f"跳過已存在文件: {pdf_path}")
            continue
            
        conversion_tasks.append((pdf_path, md_path, input_dir, force_convert))
    
    if skipped_count > 0:
        logger.info(f"跳過 {skipped_count} 個已轉換的文件")
    
    if not conversion_tasks:
        logger.info("所有文件都已轉換完成！")
        return {'message': 'All files already converted', 'skipped': skipped_count}
    
    logger.info(f"將轉換 {len(conversion_tasks)} 個文件，使用 {max_workers} 個工作進程")
    
    # 初始化進度追蹤
    tracker = ProgressTracker(len(conversion_tasks))
    failed_results = []
    
    # 使用增強的進度條
    with tqdm(
        total=len(conversion_tasks), 
        desc="轉換進度",
        unit="files",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任務
            future_to_task = {
                executor.submit(convert_single_pdf, task): task 
                for task in conversion_tasks
            }
            
            # 處理完成結果
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    tracker.update(result)
                    
                    if result.success:
                        if "跳過" not in result.error_message:
                            logger.info(f"✅ 成功: {os.path.basename(result.pdf_path)} ({result.duration:.1f}s)")
                        else:
                            logger.debug(f"⏭️  跳過: {os.path.basename(result.pdf_path)}")
                    else:
                        failed_results.append(result)
                        logger.error(f"❌ 失敗: {os.path.basename(result.pdf_path)} - {result.error_message}")
                    
                    # 更新進度條描述
                    stats = tracker.get_stats()
                    pbar.set_postfix({
                        'Success': f"{stats['successful']}/{stats['completed']}",
                        'Rate': f"{stats['success_rate']:.1f}%",
                        'Speed': f"{stats['files_per_minute']:.1f}/min"
                    })
                    
                except Exception as e:
                    logger.error(f"處理任務時發生錯誤: {str(e)}")
                    # 創建錯誤結果
                    error_result = ConversionResult(
                        pdf_path=task[0],
                        md_path=task[1],
                        success=False,
                        error_message=f"任務處理錯誤: {str(e)}"
                    )
                    failed_results.append(error_result)
                    tracker.update(error_result)
                finally:
                    pbar.update(1)
    
    # 生成最終統計
    final_stats = tracker.get_stats()
    
    logger.info("=" * 60)
    logger.info("🎉 轉換完成！")
    logger.info(f"📊 總計: {final_stats['total']} 個文件")
    logger.info(f"✅ 成功: {final_stats['successful']} 個")
    logger.info(f"❌ 失敗: {final_stats['failed']} 個")
    logger.info(f"📈 成功率: {final_stats['success_rate']:.1f}%")
    logger.info(f"⏱️  總耗時: {final_stats['elapsed_time']:.1f} 秒")
    logger.info(f"🚀 平均速度: {final_stats['files_per_minute']:.1f} 文件/分鐘")
    
    # 保存報告
    if save_reports:
        report_path = save_progress_report(output_dir, tracker, failed_results)
        logger.info(f"📋 詳細報告已保存: {report_path}")
        
        if failed_results:
            failed_list_path = save_failed_list(output_dir, failed_results)
            logger.info(f"📋 失敗文件列表: {failed_list_path}")
    
    # 失敗文件摘要
    if failed_results:
        logger.warning(f"\n❌ {len(failed_results)} 個文件轉換失敗:")
        error_summary = {}
        for result in failed_results:
            error_type = result.error_message.split(':')[0]
            error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        for error_type, count in error_summary.items():
            logger.warning(f"   {error_type}: {count} 個文件")
    
    return final_stats

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="增強版 PDF 批量轉換工具")
    parser.add_argument("input_dir", help="輸入PDF目錄")
    parser.add_argument("output_dir", help="輸出Markdown目錄")
    parser.add_argument("--workers", "-w", type=int, default=6, help="工作進程數 (預設: 6)")
    parser.add_argument("--force", "-f", action="store_true", help="強制重新轉換已存在的文件")
    parser.add_argument("--no-reports", action="store_true", help="不保存詳細報告")
    parser.add_argument("--log-file", "-l", help="指定日誌文件路徑")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ 錯誤: 輸入目錄不存在 {args.input_dir}")
        return
    
    # 執行轉換
    bulk_convert_pdfs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.workers,
        force_convert=args.force,
        save_reports=not args.no_reports,
        log_file=args.log_file
    )

if __name__ == "__main__":
    # 如果作為腳本直接運行
    if len(os.sys.argv) == 1:
        # 使用預設設定
        input_directory = "競賽資料集/reference/finance"
        output_directory = "競賽資料集/reference/finance_md"
        
        print(f"使用預設設定:")
        print(f"輸入目錄: {input_directory}")
        print(f"輸出目錄: {output_directory}")
        print(f"Workers: 6")
        
        bulk_convert_pdfs(
            input_dir=input_directory,
            output_dir=output_directory,
            max_workers=6,
            force_convert=False,
            save_reports=True
        )
    else:
        main() 