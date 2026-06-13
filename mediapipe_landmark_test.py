import sys
import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QCheckBox, QScrollArea, QLabel, 
                             QPushButton, QSlider)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

class ZoomableLandmarkApp(QMainWindow):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("MediaPipe 點位篩選器 (支援縮放)")
        self.image_path = image_path
        self.zoom_factor = 1.0  # 初始縮放比例
        
        # 初始化 MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        
        # 讀取圖片
        self.original_cv_image = cv2.imread(image_path)
        if self.original_cv_image is None:
            print("錯誤：找不到圖片")
            sys.exit()
            
        self.landmarks = self.get_landmarks()
        self.checkboxes = []
        
        self.init_ui()
        self.update_display()

    def get_landmarks(self):
        rgb_img = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        return results.multi_face_landmarks[0].landmark if results.multi_face_landmarks else []

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # --- 左側：控制面板 ---
        control_panel = QVBoxLayout()
        
        # 縮放控制區
        zoom_label = QLabel("縮放倍率:")
        control_panel.addWidget(zoom_label)
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(50)   # 50%
        self.zoom_slider.setMaximum(500)  # 500%
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_change)
        control_panel.addWidget(self.zoom_slider)

        # 全選/全不選按鈕
        btn_layout = QHBoxLayout()
        btn_all = QPushButton("全選")
        btn_none = QPushButton("全不選")
        btn_all.clicked.connect(lambda: self.set_all_checkboxes(True))
        btn_none.clicked.connect(lambda: self.set_all_checkboxes(False))
        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_none)
        control_panel.addLayout(btn_layout)

        # 點位清單
        scroll_list = QScrollArea()
        scroll_widget = QWidget()
        self.check_layout = QVBoxLayout(scroll_widget)
        for i in range(len(self.landmarks)):
            cb = QCheckBox(f"點位 {i}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_display)
            self.check_layout.addWidget(cb)
            self.checkboxes.append(cb)
        scroll_list.setWidget(scroll_widget)
        scroll_list.setWidgetResizable(True)
        scroll_list.setFixedWidth(160)
        control_panel.addWidget(scroll_list)
        
        main_layout.addLayout(control_panel)

        # --- 右側：圖片顯示區 (使用 ScrollArea 支援溢出顯示) ---
        self.image_scroll_area = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.image_scroll_area.setWidget(self.image_label)
        self.image_scroll_area.setWidgetResizable(True) # 允許內容大於視窗
        main_layout.addWidget(self.image_scroll_area)

        self.setCentralWidget(main_widget)
        self.resize(1200, 800)

    def on_zoom_change(self, value):
        self.zoom_factor = value / 100.0
        self.update_display()

    def set_all_checkboxes(self, state):
        for cb in self.checkboxes:
            cb.blockSignals(True) # 暫時阻斷訊號，避免頻繁重繪
            cb.setChecked(state)
            cb.blockSignals(False)
        self.update_display()

    def update_display(self):
        if not self.landmarks:
            return

        # 根據縮放倍率計算新尺寸
        h, w, _ = self.original_cv_image.shape
        new_w, new_h = int(w * self.zoom_factor), int(h * self.zoom_factor)
        
        # 先縮放底圖
        resized_img = cv2.resize(self.original_cv_image, (new_w, new_h))

        # 在縮放後的圖上繪製點位
        for i, lm in enumerate(self.landmarks):
            if self.checkboxes[i].isChecked():
                # 使用 MediaPipe 的比例座標 (0~1) 乘以縮放後的新寬高
                cx, cy = int(lm.x * new_w), int(lm.y * new_h)
                
                # 繪製點 (根據縮放調整點的大小，避免放大後點太小)
                radius = max(1, int(2 * self.zoom_factor))
                cv2.circle(resized_img, (cx, cy), radius, (0, 255, 0), -1)
                
                # 繪製文字 (縮放倍率大時字體也要調整)
                font_scale = 0.3 * self.zoom_factor
                cv2.putText(resized_img, str(i), (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1)

        # 轉成 QImage 並顯示
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(resized_img.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(new_w, new_h) # 必須設定固定大小，ScrollArea 才知道要出捲動軸

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 記得把 "test.jpg" 換成你的圖檔
    window = ZoomableLandmarkApp("face_image.jpg") 
    window.show()
    sys.exit(app.exec())