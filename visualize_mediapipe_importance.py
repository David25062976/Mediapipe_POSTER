import cv2
import mediapipe as mp
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def visualize_landmarks_importance(image_path, pt_path, output_path="importance_vis.jpg"):
    # 1. 檢查檔案是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到影像檔案: {image_path}")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"找不到權重排序檔案: {pt_path}")

    # 2. 載入第一階段提取的全域重要性排序表
    # 這是一個 1D Tensor，裡面存放的是依照重要性由高到低排序的 landmark index (0~477)
    importance_order = torch.load(pt_path, weights_only=True)

    # 建立一個反向對應字典：得知每個點位的「名次」 (0是第一名，477是最後一名)
    # 例如 rank_dict[33] = 0 代表左眼角的 index 33 是第1名
    rank_dict = {landmark_id.item(): rank for rank, landmark_id in enumerate(importance_order)}
    max_rank = len(importance_order) - 1

    # 3. 初始化 MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,      # 靜態影像模式
        max_num_faces=1,
        refine_landmarks=True,       # 設為 True 才會輸出包含瞳孔的 478 點
        min_detection_confidence=0.5
    )

    # 4. 讀取與處理影像
    image = cv2.imread(image_path)
    h, w, c = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print("影像中未偵測到臉部！")
        return

    face_landmarks = results.multi_face_landmarks[0]
    output_image = image.copy()
    
    # 5. 準備繪圖參數
    # 使用 matplotlib 的 jet colormap 產生熱像圖效果 (紅->黃->綠->藍)
    cmap = plt.colormaps.get_cmap('jet')

    # 6. 遍歷所有 478 個特徵點並繪製
    for idx, landmark in enumerate(face_landmarks.landmark):
        # 轉換正規化座標為像素座標
        cx = int(landmark.x * w)
        cy = int(landmark.y * h)
        
        # 取得該點位的名次
        rank = rank_dict.get(idx, max_rank)
        
        # 將名次轉換為 0.0 ~ 1.0 的分數 (名次越靠前，分數越接近 1.0)
        # 這樣 Colormap 才會是：1.0 (最重要) = 紅色, 0.0 (最不重要) = 藍色
        importance_score = 1.0 - (rank / max_rank) 
        
        # 將分數映射為 RGBA 顏色，再轉換為 OpenCV 使用的 BGR 格式 (0~255)
        rgba = cmap(importance_score)
        r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
        color_bgr = (b, g, r) 
        
        radius = int( 0.5 + (importance_score * 5) )
        thickness = -1  # 不重要的點畫小小的即可
            
        cv2.circle(output_image, (cx, cy), radius, color_bgr, thickness)
        
    # 7. 儲存結果並顯示
    cv2.imwrite(output_path, output_image)
    print(f"✅ 視覺化結果已成功儲存為: {output_path}")
    
    # 若在可顯示介面的環境執行，可解除註解下面這段來直接跳出視窗觀看
    # cv2.imshow('Attention Importance Visualization', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # 請修改為你實際的圖片路徑
    TEST_IMAGE_PATH = "face_image2.jpg" 
    ORDER_PT_PATH = "./checkpoint/20260605-012234/global_points_importance_order.pt"    # symmetrized_
    
    visualize_landmarks_importance(
        image_path=TEST_IMAGE_PATH, 
        pt_path=ORDER_PT_PATH, 
        output_path="attention_heatmap.jpg"
    )