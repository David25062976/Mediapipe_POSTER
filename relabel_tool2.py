import os
import cv2
import pandas as pd
import numpy as np
import shutil
import re

# 抑制部分 OpenCV/Qt 在終端機的煩人警告 (VNC 環境友善)
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

def get_frame_num(filename):
    """從檔名提取幀編號，例如 'frame_000026.jpg' -> 26"""
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
        img_path = os.path.join(video_dir, f)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            imgs.append(img)
            
    # 如果不足 8 幀，或是圖檔已遺失(被移走)，用黑圖補齊
    while len(imgs) < num_frames:
        imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
    row1 = np.hstack(imgs[:4])
    row2 = np.hstack(imgs[4:8])
    return np.vstack([row1, row2])

def main():
    csv_path = 'checkpoint/window_20260521-120629/train_misclassified_samples.csv' # 請確認你的 CSV 路徑
    base_dir = './data/5_classes'
    window_size = 8 # 每段顯示的幀數
    
    if not os.path.exists(csv_path):
        print(f"找不到 CSV 檔案: {csv_path}")
        return

    print("正在統整 CSV 數據並建立任務清單...")
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

    # 1. 預先將所有需要檢視的任務攤平為一個 Queue
    tasks = []
    for idx, row in df_video.iterrows():
        video_dir = row['video_path']
        if not os.path.exists(video_dir):
            continue

        all_frames = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(('.jpg', '.png'))])
        if not all_frames:
            continue

        pred_counts = pd.to_numeric(row[numeric_cols[1:]], errors='coerce').fillna(0)
        max_pred_col = pred_counts.idxmax() if not pred_counts.empty else "N/A"
        max_pred_count = pred_counts.max() if not pred_counts.empty else 0

        for i in range(0, len(all_frames), window_size):
            chunk_frames = all_frames[i : i + window_size]
            tasks.append({
                "video_dir": video_dir,
                "video_name": os.path.basename(video_dir),
                "true_label": row['true_label'],
                "total_errors": row['total_errors'],
                "max_pred_col": max_pred_col,
                "max_pred_count": max_pred_count,
                "chunk_frames": chunk_frames,
                "start_frame": get_frame_num(chunk_frames[0]),
                "end_frame": get_frame_num(chunk_frames[-1])
            })

    cv2.namedWindow('Segment Relabeling Tool', cv2.WINDOW_NORMAL)
    
    video_track_state = {}
    moved_operations_count = 0
    curr_idx = 0
    
    # 2. 使用 While 迴圈走訪佇列，方便我們使用 Next 預覽
    while curr_idx < len(tasks):
        task = tasks[curr_idx]
        
        # 如果因為某些原因這部影片已經不存在了，跳過
        if not os.path.exists(task["video_dir"]):
            curr_idx += 1
            continue
            
        # 尋找下一個有效的任務作為 Preview
        next_task = None
        lookahead = curr_idx + 1
        while lookahead < len(tasks):
            if os.path.exists(tasks[lookahead]["video_dir"]):
                next_task = tasks[lookahead]
                break
            lookahead += 1

        # 建立主要影像網格
        grid_img = create_grid_image_from_chunk(task["video_dir"], task["chunk_frames"])
        grid_h, grid_w = grid_img.shape[:2]
        
        # 建立預覽影像網格 (縮小 50%)
        preview_img = None
        if next_task:
            preview_img = create_grid_image_from_chunk(next_task["video_dir"], next_task["chunk_frames"])
            preview_img = cv2.resize(preview_img, (0, 0), fx=0.5, fy=0.5)

        # --- 畫面佈局邏輯 ---
        canvas_w = max(1350, grid_w) 
        header_h = 180
        preview_header_h = 60 if preview_img is not None else 0
        preview_h = preview_img.shape[0] if preview_img is not None else 0
        
        canvas_h = header_h + grid_h + preview_header_h + preview_h + 20
        
        display_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # 繪製主圖 (置中)
        x_offset = (canvas_w - grid_w) // 2
        display_img[header_h:header_h+grid_h, x_offset:x_offset+grid_w] = grid_img
        
        # 寫入文字資訊
        info_text1 = f"Path: {task['video_name']} (Frames {task['start_frame']} to {task['end_frame']})"
        info_text2 = f"Orig Label: [{task['true_label']}] | Video Total Errors: {task['total_errors']}"
        info_text3 = f"Most Confused as: {str(task['max_pred_col']).replace('pred_as_', '')} ({task['max_pred_count']} times)"
        instruction = "[0_browsing, 1_interested, 2_thinking, 3_buy, 4_pass]: Relabel & Move | [Space]: Keep | [S]: Skip Video | [Q]: Quit"
        
        cv2.putText(display_img, info_text1, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display_img, info_text2, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(display_img, info_text3, (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_img, instruction, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        # 繪製預覽圖 (如果有)
        if preview_img is not None:
            y_prev_txt = header_h + grid_h + 35
            prev_txt = f"--- NEXT PREVIEW: {next_task['video_name']} (Frames {next_task['start_frame']}~{next_task['end_frame']}) ---"
            cv2.putText(display_img, prev_txt, (15, y_prev_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
            
            prev_h, prev_w = preview_img.shape[:2]
            prev_x_offset = (canvas_w - prev_w) // 2
            y_prev_img = header_h + grid_h + preview_header_h
            display_img[y_prev_img:y_prev_img+prev_h, prev_x_offset:prev_x_offset+prev_w] = preview_img

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
                print(f"  [Keep] 保留片段 Frames {task['start_frame']}~{task['end_frame']}")
                curr_idx += 1
                break
                
            elif char_key == 's':
                print(f"  [Skip Video] 略過 {task['video_name']} 剩餘片段。")
                current_video_dir = task["video_dir"]
                # 略過屬於同一部影片的所有後續 Task
                while curr_idx < len(tasks) and tasks[curr_idx]["video_dir"] == current_video_dir:
                    curr_idx += 1
                break
                
            elif char_key in class_map:
                target_folder_name = class_map[char_key]
                target_dir = os.path.join(base_dir, target_folder_name)
                
                # --- 連續性判斷邏輯 ---
                state = video_track_state.get(task["video_dir"])
                append_to_existing = False
                
                if state is not None:
                    if state["class_key"] == char_key:
                        if (task["start_frame"] - state["last_frame"]) <= 3:
                            append_to_existing = True
                            
                if append_to_existing:
                    new_chunk_dir = state["dest_dir"]
                else:
                    new_chunk_dir_name = f"{task['video_name']}_part{task['start_frame']}"
                    new_chunk_dir = os.path.join(target_dir, new_chunk_dir_name)
                
                if os.path.normpath(task["video_dir"]) == os.path.normpath(new_chunk_dir):
                    print(f"  [Info] 片段已經屬於 {target_folder_name}。")
                    curr_idx += 1
                    break
                    
                try:
                    os.makedirs(new_chunk_dir, exist_ok=True)
                    for f in task["chunk_frames"]:
                        src_path = os.path.join(task["video_dir"], f)
                        dst_path = os.path.join(new_chunk_dir, f)
                        if os.path.exists(src_path):
                            shutil.move(src_path, dst_path)
                        
                    if append_to_existing:
                        print(f"  [Append] 連續幀整合！附加 Frames {task['start_frame']}~{task['end_frame']} -> {target_folder_name}/{os.path.basename(new_chunk_dir)}")
                    else:
                        print(f"  [New Folder] 建立新片段 Frames {task['start_frame']}~{task['end_frame']} -> {target_folder_name}/{os.path.basename(new_chunk_dir)}")
                        
                    moved_operations_count += 1
                    
                    video_track_state[task["video_dir"]] = {
                        "class_key": char_key,
                        "dest_dir": new_chunk_dir,
                        "last_frame": task["end_frame"]
                    }
                except Exception as e:
                    print(f"  [Error] 搬移失敗: {e}")
                
                curr_idx += 1
                break

    cv2.destroyAllWindows()
    print(f"\n✅ 檢視完畢！共執行了 {moved_operations_count} 次搬移操作。")

if __name__ == "__main__":
    main()