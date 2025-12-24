import argparse
import os
import time
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Check if server task is completed')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to check for .npz files')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Starting to check for .npz files in: {args.log_dir}")
    
    while True:
        # 使用 glob 查找目錄下所有的 .npz 檔案
        npz_files = glob.glob(os.path.join(args.log_dir, '*.npz'))
        
        if npz_files:
            print(f"Found .npz file(s) in {args.log_dir}:")
            for file in npz_files:
                print(f"  - {os.path.basename(file)}")
            break
        
        time.sleep(10)
        
    time.sleep(10)
        
if __name__ == '__main__':
    main()