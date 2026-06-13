import os
import shutil
import random

# ---------------- 配置區塊 ----------------
# 設定隨機種子，確保每次執行的切分結果一致
random.seed(42)

# 基礎目錄與檔案路徑 (請確認與您的實際路徑相符)
base_dir = './data/AffectNet'
train_txt = os.path.join(base_dir, 'train_set', 'train_annotations_8class.txt')
valid_txt = os.path.join(base_dir, 'valid_set', 'valid_annotations_8class.txt')
train_img_dir = os.path.join(base_dir, 'train_set', 'images')
valid_img_dir = os.path.join(base_dir, 'valid_set', 'images')

# 輸出目錄與切分比例
output_dir = './data/AffectNet8c_Shopping_test'
splits = ['train', 'valid', 'test']
split_ratios = {'train': 0.7, 'valid': 0.15, 'test': 0.15} # 加總必須為 1.0

# ------------------------------------------

def main():
    # 1. 建立輸出目錄結構 (train, valid, test 底下各包含 0~7 的資料夾)
    print("正在建立目標資料夾結構...")
    for split in splits:
        for i in range(8):
            os.makedirs(os.path.join(output_dir, split, str(i)), exist_ok=True)

    # 2. 準備一個字典來收集每個類別的所有圖片路徑
    # 資料結構: { '0': [path1, path2...], '1': [...], ... }
    class_images = {str(i): [] for i in range(8)}

    # 輔助函式：讀取標註檔並將路徑加入字典
    def parse_annotations(txt_path, img_dir):
        if not os.path.exists(txt_path):
            print(f"[警告] 找不到標註檔: {txt_path}")
            return
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_id = parts[1]
                    
                    if class_id in class_images:
                        full_img_path = os.path.join(img_dir, img_name)
                        # 記錄完整的來源路徑
                        class_images[class_id].append(full_img_path)

    # 3. 讀取並合併 train 和 valid 的標註
    print("正在讀取並解析標註檔...")
    parse_annotations(train_txt, train_img_dir)
    parse_annotations(valid_txt, valid_img_dir)

    print("-" * 30)
    print("開始打亂順序並分配檔案至 train/valid/test...")

    total_copied = 0
    missing_count = 0

    # 4. 針對每個類別進行打亂和切分
    for class_id, img_paths in class_images.items():
        # 打亂該類別的所有圖片，避免同一個影片序列的截圖過度集中
        random.shuffle(img_paths)
        
        # 過濾掉實際上不存在的圖片路徑
        valid_img_paths = [p for p in img_paths if os.path.exists(p)]
        missing_count += (len(img_paths) - len(valid_img_paths))
        
        total_imgs = len(valid_img_paths)
        if total_imgs == 0:
            print(f"[警告] 類別 {class_id} 沒有找到任何有效圖片。")
            continue
            
        # 計算切分索引
        train_end = int(total_imgs * split_ratios['train'])
        valid_end = train_end + int(total_imgs * split_ratios['valid'])
        
        # 將路徑分配給不同的 split
        split_dict = {
            'train': valid_img_paths[:train_end],
            'valid': valid_img_paths[train_end:valid_end],
            'test': valid_img_paths[valid_end:]
        }
        
        # 複製檔案到對應目錄
        for split_name, paths in split_dict.items():
            for src_path in paths:
                img_name = os.path.basename(src_path)
                dst_path = os.path.join(output_dir, split_name, class_id, img_name)
                shutil.copy(src_path, dst_path)
                total_copied += 1
                
        print(f"類別 {class_id} 處理完成: 總共 {total_imgs} 張 -> Train: {len(split_dict['train'])}, Valid: {len(split_dict['valid'])}, Test: {len(split_dict['test'])}")

    print("-" * 30)
    print("資料集重新切分完成！")
    print(f"成功複製: {total_copied} 張圖片")
    if missing_count > 0:
        print(f"[提示] 有 {missing_count} 張圖片在標註檔中有紀錄，但實體檔案遺失。")
    print(f"輸出目錄位在: {os.path.abspath(output_dir)}")

if __name__ == '__main__':
    main()