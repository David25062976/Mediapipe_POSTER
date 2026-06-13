import sys
import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QScrollArea, QLabel, QSlider, 
                             QGroupBox, QTextEdit, QPushButton)
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent
from PyQt6.QtCore import Qt, pyqtSignal, QSize

# --- 1. 自定義的圖片顯示元件 ---
class LandmarkLabel(QLabel):
    landmarkToggled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        
        self.raw_landmarks = []      
        self.current_zoom = 1.0
        self.image_size = QSize(0, 0) 
        self.id_visibility_states = [] 
        self.hit_radius = 10 # 稍微加大感應範圍，更好點擊

    def set_data(self, landmarks, states, zoom):
        self.raw_landmarks = landmarks
        self.id_visibility_states = states
        self.current_zoom = zoom

    def update_image_size(self, w, h):
        self.image_size = QSize(w, h)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.raw_landmarks and not self.image_size.isEmpty():
            click_pos = event.position()
            ex, ey = click_pos.x(), click_pos.y()
            w, h = self.image_size.width(), self.image_size.height()

            clicked_index = -1
            for i, lm in enumerate(self.raw_landmarks):
                px, py = lm.x * w, lm.y * h
                distance = np.sqrt((ex - px)**2 + (ey - py)**2)
                if distance <= self.hit_radius:
                    clicked_index = i
                    break 

            if clicked_index != -1:
                self.id_visibility_states[clicked_index] = not self.id_visibility_states[clicked_index]
                self.landmarkToggled.emit()
        super().mousePressEvent(event)

# --- 2. 主應用程式視窗 ---
class AdvancedLandmarkApp(QMainWindow):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("MediaPipe 點位互動分析工具")
        self.resize(1500, 900)
        
        self.image_path = image_path
        self.zoom_factor = 1.0
        
        # 初始化 MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        
        self.original_cv_image = cv2.imread(image_path)
        if self.original_cv_image is None:
            print("錯誤：找不到圖片")
            sys.exit()
            
        self.landmarks = self.get_landmarks()
        # 預設全部關閉編號，讓畫面乾淨，讓使用者點選開啟
        self.id_visibility_states = [False] * len(self.landmarks)
        
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
        control_panel.setContentsMargins(10, 10, 10, 10)
        
        # 1. 縮放控制
        zoom_group = QGroupBox("檢視控制")
        zoom_layout = QVBoxLayout()
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(20, 500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_change)
        self.zoom_label = QLabel("縮放: 100%")
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        
        # 快速操作按鈕
        btn_layout = QHBoxLayout()
        btn_all = QPushButton("全部顯示")
        btn_none = QPushButton("全部隱藏")
        btn_all.clicked.connect(lambda: self.set_all_states(True))
        btn_none.clicked.connect(lambda: self.set_all_states(False))
        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_none)
        zoom_layout.addLayout(btn_layout)
        
        zoom_group.setLayout(zoom_layout)
        control_panel.addWidget(zoom_group)
        
        # 2. 目前顯示的點位清單
        list_group = QGroupBox("目前顯示編號的點 (點擊圖片切換)")
        list_layout = QVBoxLayout()
        self.active_points_text = QTextEdit()
        self.active_points_text.setReadOnly(True) # 唯讀
        self.active_points_text.setStyleSheet("background-color: #f0f0f0; font-family: Consolas; font-size: 12px;")
        list_layout.addWidget(self.active_points_text)
        list_group.setLayout(list_layout)
        control_panel.addWidget(list_group)

        main_layout.addLayout(control_panel, 1)

        # --- 右側：圖片顯示區 ---
        self.image_scroll_area = QScrollArea()
        self.image_label = LandmarkLabel()
        self.image_label.landmarkToggled.connect(self.update_display)
        self.image_scroll_area.setWidget(self.image_label)
        self.image_scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.image_scroll_area, 4)

        self.setCentralWidget(main_widget)

    def on_zoom_change(self, value):
        self.zoom_factor = value / 100.0
        self.zoom_label.setText(f"縮放: {value}%")
        self.update_display()

    def set_all_states(self, state):
        self.id_visibility_states = [state] * len(self.landmarks)
        self.update_display()

    def update_display(self):
        if not self.landmarks:
            return

        h, w, _ = self.original_cv_image.shape
        new_w, new_h = int(w * self.zoom_factor), int(h * self.zoom_factor)
        
        self.image_label.set_data(self.landmarks, self.id_visibility_states, self.zoom_factor)
        self.image_label.update_image_size(new_w, new_h)

        resized_img = cv2.resize(self.original_cv_image, (new_w, new_h))

        active_indices = []

        # 繪圖邏輯
        for i, lm in enumerate(self.landmarks):
            cx, cy = int(lm.x * new_w), int(lm.y * h * self.zoom_factor)
            
            if self.id_visibility_states[i]:
                # 狀態：顯示編號 (亮綠色)
                active_indices.append(str(i))
                radius = max(3, int(4 * self.zoom_factor))
                cv2.circle(resized_img, (cx, cy), radius, (0, 255, 0), -1)
                # 藍色數字
                if self.zoom_factor >= 0.3:
                    cv2.putText(resized_img, str(i), (cx + 3, cy - 3), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.zoom_factor, (255, 50, 50), 1)
            else:
                # 狀態：不顯示編號 (亮紅色，更明顯)
                radius = max(2, int(3 * self.zoom_factor))
                cv2.circle(resized_img, (cx, cy), radius, (0, 0, 255), -1) # 紅色

        # 更新左側文字清單
        self.active_points_text.setText(", ".join(active_indices))

        # 顯示圖片
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(resized_img.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
        self.image_label.setFixedSize(new_w, new_h)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedLandmarkApp("face_image.jpg") # 請更換為你的檔名
    window.show()
    sys.exit(app.exec())