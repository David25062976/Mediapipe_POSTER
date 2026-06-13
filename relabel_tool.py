# [0_browsing, 1_interested, 2_thinking, 3_buy, 4_pass]
# csv_path = 'checkpoint/window_20260521-120629/valid_misclassified_samples.csv'

import os
import cv2
import pandas as pd
import numpy as np
import shutil
import re

# 抑制部分 OpenCV/Qt 在終端機的煩人警告 (VNC 環境友善)
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

def get_frame_num(filename):
    """
    從檔名提取幀編號，例如 'frame_000026.jpg' -> 26
    """
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else -1

def auto_detect_classes(base_dir):
    """自動掃描基礎目錄底下的所有類別資料夾"""
    folders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    class_map = {}
    for folder in folders:
        try:
            key = str(int(folder.split('_')[0]))
            class_map[key] = folder
        except ValueError:
            continue
    return class_map

def create_grid_image_from_chunk(video_dir, chunk_frames, num_frames=8):
    """從給定的幀列表中讀取影像，並組合成 2x4 網格"""
    imgs = []
    for f in chunk_frames:
        img = cv2.imread(os.path.join(video_dir, f))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            imgs.append(img)
            
    # 如果不足 8 幀，用黑圖補齊
    while len(imgs) < num_frames:
        imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
    row1 = np.hstack(imgs[:4])
    row2 = np.hstack(imgs[4:8])
    return np.vstack([row1, row2])

def main():
    csv_path = 'checkpoint/window_20260521-120629/valid_misclassified_samples.csv' # 請確認你的 CSV 路徑
    base_dir = './data/5_classes'
    window_size = 8 # 每段顯示的幀數
    
    if not os.path.exists(csv_path):
        print(f"找不到 CSV 檔案: {csv_path}")
        return

    print("正在統整 CSV 數據...")
    df = pd.read_csv(csv_path)
    df['video_path'] = df['file_path'].apply(os.path.dirname)
    
    numeric_cols = [col for col in df.columns if col == 'total_errors' or col.startswith('pred_as_')]
    df_video = df.groupby(['video_path', 'true_label'])[numeric_cols].sum().reset_index()
    df_video = df_video.sort_values(by='total_errors', ascending=False)
    
    class_map = auto_detect_classes(base_dir)
    print(f"\n偵測到的類別映射: {class_map}")
    print("================ 操作說明 ================")
    print("[0-4]   : 分類並移動影像 (連續幀會自動群組)")
    print("[Space] : 保留這 8 幀，看下一段")
    print("[S]     : 略過這部影片剩下的所有片段，直接看下一部影片")
    print("[Q]     : 儲存並退出")
    print("==========================================\n")

    # 改為 WINDOW_NORMAL，允許使用者在 VNC 自由縮放視窗
    cv2.namedWindow('Segment Relabeling Tool', cv2.WINDOW_NORMAL)
    
    # 用來追蹤每部影片「最後一次搬移」的狀態，以判斷是否連續
    # 格式: {"video_dir": {"class_key": str, "dest_dir": str, "last_frame": int}}
    video_track_state = {}
    moved_operations_count = 0
    
    for idx, row in df_video.iterrows():
        video_dir = row['video_path']
        true_label = row['true_label']
        total_errors = row['total_errors']
        
        if not os.path.exists(video_dir):
            continue

        all_frames = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(('.jpg', '.png'))])
        if not all_frames:
            continue

        pred_counts = pd.to_numeric(row[numeric_cols[1:]], errors='coerce').fillna(0)
        max_pred_col = pred_counts.idxmax() if not pred_counts.empty else "N/A"
        max_pred_count = pred_counts.max() if not pred_counts.empty else 0

        skip_rest_of_video = False
        
        for i in range(0, len(all_frames), window_size):
            if skip_rest_of_video:
                break
                
            chunk_frames = all_frames[i : i + window_size]
            start_frame = get_frame_num(chunk_frames[0])
            end_frame = get_frame_num(chunk_frames[-1])
            
            # 建立影像網格 (預設寬度為 896)
            grid_img = create_grid_image_from_chunk(video_dir, chunk_frames)
            grid_h, grid_w = grid_img.shape[:2]
            
            # --- 畫面加寬邏輯 (1350像素) 確保長檔名顯示 ---
            canvas_w = max(1350, grid_w) 
            header_h = 180
            canvas_h = header_h + grid_h
            
            display_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            x_offset = (canvas_w - grid_w) // 2
            display_img[header_h:header_h+grid_h, x_offset:x_offset+grid_w] = grid_img
            
            # 寫入文字資訊
            video_name = os.path.basename(video_dir)
            info_text1 = f"Path: {video_name} (Frames {start_frame} to {end_frame})"
            info_text2 = f"Orig Label: [{true_label}] | Video Total Errors: {total_errors}"
            info_text3 = f"Most Confused as: {str(max_pred_col).replace('pred_as_', '')} ({max_pred_count} times)"
            instruction = "[0_browsing, 1_interested, 2_thinking, 3_buy, 4_pass]: Relabel & Move | [Space]: Keep | [S]: Skip Video | [Q]: Quit"
            
            cv2.putText(display_img, info_text1, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display_img, info_text2, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(display_img, info_text3, (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_img, instruction, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            
            # 強制設定視窗大小並顯示
            cv2.resizeWindow('Segment Relabeling Tool', canvas_w, canvas_h)
            cv2.imshow('Segment Relabeling Tool', display_img)
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                char_key = chr(key).lower()
                
                if char_key == 'q':
                    print("\n[退出] 使用者中斷工具。")
                    cv2.destroyAllWindows()
                    print(f"總結: 成功執行了 {moved_operations_count} 次搬移操作。")
                    return
                    
                elif key == 32: # Space
                    print(f"  [Keep] 保留片段 Frames {start_frame}~{end_frame}")
                    break
                    
                elif char_key == 's':
                    print(f"  [Skip Video] 略過 {video_name} 剩餘片段。")
                    skip_rest_of_video = True
                    break
                    
                elif char_key in class_map:
                    target_folder_name = class_map[char_key]
                    target_dir = os.path.join(base_dir, target_folder_name)
                    
                    # --- 連續性判斷邏輯 ---
                    state = video_track_state.get(video_dir)
                    append_to_existing = False
                    
                    if state is not None:
                        # 條件 1: 類別與上一次移動的一致
                        if state["class_key"] == char_key:
                            # 條件 2: 幀號差小於等於 3 (例如 8 - 7 = 1 <= 3)
                            if (start_frame - state["last_frame"]) <= 3:
                                append_to_existing = True
                                
                    if append_to_existing:
                        # 連續幀，沿用剛剛的資料夾
                        new_chunk_dir = state["dest_dir"]
                    else:
                        # 建立全新的資料夾 (使用起始幀號作為資料夾後綴，避免重複)
                        new_chunk_dir_name = f"{video_name}_part{start_frame}"
                        new_chunk_dir = os.path.join(target_dir, new_chunk_dir_name)
                    
                    if os.path.normpath(video_dir) == os.path.normpath(new_chunk_dir):
                        print(f"  [Info] 片段已經屬於 {target_folder_name}。")
                        break
                        
                    try:
                        os.makedirs(new_chunk_dir, exist_ok=True)
                        for f in chunk_frames:
                            src_path = os.path.join(video_dir, f)
                            dst_path = os.path.join(new_chunk_dir, f)
                            shutil.move(src_path, dst_path)
                            
                        # 更新終端機輸出訊息，讓使用者清楚知道是「附加」還是「新建」
                        if append_to_existing:
                            print(f"  [Append] 連續幀整合！附加 Frames {start_frame}~{end_frame} -> {target_folder_name}/{os.path.basename(new_chunk_dir)}")
                        else:
                            print(f"  [New Folder] 建立新片段 Frames {start_frame}~{end_frame} -> {target_folder_name}/{os.path.basename(new_chunk_dir)}")
                            
                        moved_operations_count += 1
                        
                        # 更新此影片的最後操作狀態
                        video_track_state[video_dir] = {
                            "class_key": char_key,
                            "dest_dir": new_chunk_dir,
                            "last_frame": end_frame
                        }
                    except Exception as e:
                        print(f"  [Error] 搬移失敗: {e}")
                    break

    cv2.destroyAllWindows()
    print(f"\n✅ 檢視完畢！共執行了 {moved_operations_count} 次搬移操作。")

if __name__ == "__main__":
    main()