import cv2

def read_video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)  # 開啟影片
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 沒有讀到畫面就跳出迴圈
        frames.append(frame)  # 將每一幀加入 list

    cap.release()
    return frames

# 使用範例
video_path = 'data/vivit/video_001.mp4'
frame_list = read_video_to_frames(video_path)

print(f"總共讀取了 {len(frame_list)} 幀")
# 顯示第一幀確認結果（可選）
cv2.imshow("First Frame", frame_list[0])
cv2.waitKey(0)
cv2.imshow("First Frame", frame_list[1])
cv2.waitKey(0)
cv2.imshow("First Frame", frame_list[2])
cv2.waitKey(0)
cv2.imshow("First Frame", frame_list[3])
cv2.waitKey(0)
cv2.imshow("First Frame", frame_list[4])
cv2.waitKey(0)
cv2.imshow("First Frame", frame_list[5])
cv2.waitKey(0)
cv2.destroyAllWindows()
