import os
import shutil

def main():
    # 取得目前腳本所在的目錄 (專案根目錄)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定義目標資料夾與要移動的檔案清單
    dirs = {
        "model": ["keras_model.h5", "labels.txt"],
        "assets": ["ui.png", "01.png", "02.png", "03.png"]
    }
    
    print(f"開始整理專案資料夾: {base_dir}")
    
    for folder_name, files in dirs.items():
        # 建立資料夾路徑
        folder_path = os.path.join(base_dir, folder_name)
        
        # 如果資料夾不存在則建立
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"已建立資料夾: {folder_name}")
            
        # 移動檔案
        for file_name in files:
            src_path = os.path.join(base_dir, file_name)
            dst_path = os.path.join(folder_path, file_name)
            
            if os.path.exists(src_path):
                try:
                    # 如果目標已有檔案，shutil.move 預設可能會報錯或覆蓋，視作業系統而定
                    # 這裡直接移動
                    shutil.move(src_path, dst_path)
                    print(f"已移動: {file_name} -> {folder_name}/{file_name}")
                except Exception as e:
                    print(f"移動失敗 {file_name}: {e}")
            elif os.path.exists(dst_path):
                print(f"略過: {file_name} 已經在 {folder_name} 中")
            else:
                print(f"找不到檔案: {file_name} (可能不存在或名稱錯誤)")

    print("整理完成！")

if __name__ == "__main__":
    main()