import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, train_dir, valid_dir, test_dir, train_ratio=0.7, valid_ratio=0.15, seed=42):
    # 設定隨機種子，確保每次執行分割結果一致 (方便實驗重現)
    random.seed(seed)
    
    # 定義支援的圖片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 建立路徑物件
    src_path = Path(source_dir)
    train_path = Path(train_dir)
    valid_path = Path(valid_dir)
    test_path = Path(test_dir)

    # 確保目的地目錄存在
    train_path.mkdir(parents=True, exist_ok=True)
    valid_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # 1. 遞迴蒐集所有圖片路徑
    all_images = []
    for file in src_path.rglob('*'):
        if file.is_file() and file.suffix.lower() in image_extensions:
            all_images.append(file)

    if not all_images:
        print("在來源目錄中找不到任何圖片。")
        return

    # 2. 隨機打亂檔案順序
    random.shuffle(all_images)

    # 3. 計算分割索引點
    train_split_index = int(len(all_images) * train_ratio)
    
    train_files = all_images[:train_split_index]
    valid_test_files = all_images[train_split_index:]
    
    remain_valid_ratio = valid_ratio / ( 1 - train_ratio )
    valid_split_index = int(len(valid_test_files) * remain_valid_ratio)
    valid_files = valid_test_files[:valid_split_index]
    test_files = valid_test_files[valid_split_index:]

    # 4. 執行複製
    def copy_files(files, destination_folder):
        count = 0
        for f in files:
            # 為了避免不同子資料夾有同名檔案，目的地檔名加上原始路徑特徵或保持唯一
            # 這裡簡單處理：如果目的地已有同名檔案，則加上計數器
            dest_file = destination_folder / f.name
            if dest_file.exists():
                dest_file = destination_folder / f"{f.stem}_{count}{f.suffix}"
            
            shutil.copy2(f, dest_file)
            count += 1
        return count

    print(f"開始分配圖片（總數: {len(all_images)}）...")
    train_count = copy_files(train_files, train_path)
    valid_count = copy_files(valid_files, valid_path)
    test_count = copy_files(test_files, test_path)

    print("-" * 30)
    print(f"任務完成！")
    print(f"訓練集 (Train): {train_count} 張 (約 {train_ratio*100:.1f}%)")
    print(f"訓練集 (Valid): {valid_count} 張 (約 {valid_ratio*100:.1f}%)")
    print(f"測試集 (Test): {test_count} 張 (約 {(1-train_ratio-valid_ratio)*100:.1f}%)")

# --- 設定區域 ---
TRAIN_SET_RATIO = 0.7                   # 分配給 Train 的比例 (0.0 ~ 1.0)
VALID_SET_RATIO = 0.15

if __name__ == "__main__":
    SOURCE_DIR = "/home/lab702/POSTER/data/5_classes_single/0"      # 來源目錄
    TRAIN_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/train_set/3"        # 目的地 Train
    VALID_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/valid_set/3"          # 目的地 valid
    TEST_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/test_set/3"          # 目的地 test
    split_dataset(SOURCE_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR, TRAIN_SET_RATIO, VALID_SET_RATIO)

    SOURCE_DIR = "/home/lab702/POSTER/data/5_classes_single/1"      # 來源目錄
    TRAIN_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/train_set/4"        # 目的地 Train
    VALID_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/valid_set/4"          # 目的地 valid
    TEST_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/test_set/4"          # 目的地 test
    split_dataset(SOURCE_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR, TRAIN_SET_RATIO, VALID_SET_RATIO)

    SOURCE_DIR = "/home/lab702/POSTER/data/5_classes_single/2"      # 來源目錄
    TRAIN_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/train_set/5"        # 目的地 Train
    VALID_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/valid_set/5"          # 目的地 valid
    TEST_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/test_set/5"          # 目的地 test
    split_dataset(SOURCE_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR, TRAIN_SET_RATIO, VALID_SET_RATIO)

    SOURCE_DIR = "/home/lab702/POSTER/data/5_classes_single/3"      # 來源目錄
    TRAIN_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/train_set/6"        # 目的地 Train
    VALID_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/valid_set/6"          # 目的地 valid
    TEST_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/test_set/6"          # 目的地 test
    split_dataset(SOURCE_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR, TRAIN_SET_RATIO, VALID_SET_RATIO)

    SOURCE_DIR = "/home/lab702/POSTER/data/5_classes_single/4"      # 來源目錄
    TRAIN_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/train_set/7"        # 目的地 Train
    VALID_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/valid_set/7"          # 目的地 valid
    TEST_DIR = "/home/lab702/POSTER/data/AffectNet8c_Shopping_test/test_set/7"          # 目的地 test
    split_dataset(SOURCE_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR, TRAIN_SET_RATIO, VALID_SET_RATIO)
