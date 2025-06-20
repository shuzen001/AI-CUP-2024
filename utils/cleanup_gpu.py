#!/usr/bin/env python3
"""
GPU 清理工具
清理遺留的 PDF 轉換進程和釋放 GPU 記憶體
"""

import subprocess
import sys
import logging
import os
import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gpu_processes():
    """獲取使用 GPU 的進程列表"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                if len(parts) >= 3:
                    processes.append({
                        'pid': int(parts[0]),
                        'name': parts[1],
                        'memory': int(parts[2])
                    })
        return processes
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def find_multiprocessing_processes():
    """尋找 multiprocessing 相關進程"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True)
        multiproc_pids = []
        
        for line in result.stdout.split('\n'):
            if 'multiprocessing' in line and 'python' in line:
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pid = int(parts[1])
                        multiproc_pids.append(pid)
                    except ValueError:
                        continue
        
        return multiproc_pids
    except subprocess.CalledProcessError:
        return []

def find_pdf_processes():
    """尋找 PDF 轉換相關進程"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True)
        pdf_pids = []
        
        for line in result.stdout.split('\n'):
            if any(keyword in line for keyword in ['bulk_convert', 'pdf_to_md', 'docling']):
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pid = int(parts[1])
                        pdf_pids.append(pid)
                    except ValueError:
                        continue
        
        return pdf_pids
    except subprocess.CalledProcessError:
        return []

def get_gpu_memory_usage():
    """獲取當前 GPU 記憶體使用量"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            used, total = result.stdout.strip().split(', ')
            return int(used), int(total)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return 0, 0

def cleanup_processes(pids, process_type="進程", force=False):
    """清理指定的進程"""
    if not pids:
        logger.info(f"沒有找到需要清理的{process_type}")
        return True
    
    logger.info(f"找到 {len(pids)} 個{process_type}: {pids}")
    
    for pid in pids:
        try:
            # 先嘗試優雅關閉
            if not force:
                logger.info(f"嘗試優雅關閉 PID {pid}...")
                os.kill(pid, signal.SIGTERM)
            else:
                logger.info(f"強制終止 PID {pid}...")
                os.kill(pid, signal.SIGKILL)
                
        except ProcessLookupError:
            logger.info(f"PID {pid} 已經不存在")
        except PermissionError:
            logger.error(f"沒有權限終止 PID {pid}")
            return False
        except Exception as e:
            logger.error(f"終止 PID {pid} 時發生錯誤: {e}")
    
    return True

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU 清理工具")
    parser.add_argument("--force", "-f", action="store_true", 
                       help="強制終止所有相關進程")
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="只顯示會被清理的進程，不實際執行")
    
    args = parser.parse_args()
    
    logger.info("🔍 掃描 GPU 使用情況...")
    
    # 獲取 GPU 記憶體使用情況
    used_memory, total_memory = get_gpu_memory_usage()
    logger.info(f"GPU 記憶體使用: {used_memory} MB / {total_memory} MB ({used_memory/total_memory*100:.1f}%)")
    
    # 獲取 GPU 進程
    gpu_processes = get_gpu_processes()
    if gpu_processes:
        logger.info("使用 GPU 的進程:")
        for proc in gpu_processes:
            logger.info(f"  PID {proc['pid']}: {proc['name']} ({proc['memory']} MB)")
    
    # 尋找相關進程
    multiproc_pids = find_multiprocessing_processes()
    pdf_pids = find_pdf_processes()
    
    all_pids = list(set(multiproc_pids + pdf_pids))
    
    if args.dry_run:
        logger.info("🔍 Dry-run 模式，以下進程會被清理:")
        logger.info(f"Multiprocessing 進程: {multiproc_pids}")
        logger.info(f"PDF 轉換進程: {pdf_pids}")
        logger.info(f"總共: {len(all_pids)} 個進程")
        return
    
    if not all_pids:
        logger.info("✅ 沒有找到需要清理的進程")
        return
    
    logger.info(f"🧹 開始清理 {len(all_pids)} 個進程...")
    
    # 先嘗試優雅關閉
    if not args.force:
        logger.info("步驟 1: 嘗試優雅關閉...")
        cleanup_processes(all_pids, "相關進程", force=False)
        
        # 等待一下
        import time
        time.sleep(3)
        
        # 檢查是否還有進程存在
        remaining_multiproc = find_multiprocessing_processes()
        remaining_pdf = find_pdf_processes()
        remaining_pids = list(set(remaining_multiproc + remaining_pdf))
        
        if remaining_pids:
            logger.warning(f"仍有 {len(remaining_pids)} 個進程未關閉，將強制終止...")
            cleanup_processes(remaining_pids, "殘留進程", force=True)
    else:
        logger.info("強制模式: 直接終止所有相關進程...")
        cleanup_processes(all_pids, "相關進程", force=True)
    
    # 等待 GPU 記憶體釋放
    import time
    time.sleep(2)
    
    # 檢查清理結果
    final_used, final_total = get_gpu_memory_usage()
    released = used_memory - final_used
    
    logger.info("🎉 清理完成!")
    logger.info(f"GPU 記憶體使用: {final_used} MB / {final_total} MB ({final_used/final_total*100:.1f}%)")
    if released > 0:
        logger.info(f"釋放了 {released} MB GPU 記憶體")
    
    # 最終檢查
    remaining_gpu_processes = get_gpu_processes()
    python_processes = [p for p in remaining_gpu_processes if 'python' in p['name'].lower()]
    
    if python_processes:
        logger.warning("⚠️  仍有 Python 進程在使用 GPU:")
        for proc in python_processes:
            logger.warning(f"  PID {proc['pid']}: {proc['name']} ({proc['memory']} MB)")
    else:
        logger.info("✅ 所有 Python GPU 進程已清理完畢")

if __name__ == "__main__":
    main() 