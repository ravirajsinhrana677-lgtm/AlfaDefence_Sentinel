import sys
import os
import time
import json
import csv
import winsound
import ctypes
from datetime import datetime

import cv2
import numpy as np
import mss

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QFont

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
WDA_EXCLUDEFROMCAPTURE = 0x00000011

CONFIG_FILE = "sentinel_config.json"
CSV_FILE = "sentinel_alerts.csv"
SNAPSHOT_DIR = "sentinel_snapshots"

class SentinelWorker(QThread):
    alert_detected = pyqtSignal(int, str, object, list)

    def __init__(self, monitor_idx=1, rows=3, cols=3, mode="Motion", sensitivity=30):
        super().__init__()
        self.monitor_idx = monitor_idx
        self.rows = rows
        self.cols = cols
        self.mode = mode
        self.sensitivity = sensitivity
        self.running = False
        self.subtractors = {}
        for r in range(self.rows):
            for c in range(self.cols):
                cam_id = r * self.cols + c + 1
                self.subtractors[cam_id] = cv2.createBackgroundSubtractorMOG2(
                    history=150, varThreshold=self.sensitivity, detectShadows=False
                )

    def run(self):
        self.running = True
        with mss.mss() as sct:
            monitors = sct.monitors
            target_monitor = monitors[min(self.monitor_idx, len(monitors) - 1)]

            while self.running:
                loop_start = time.time()
                raw_frame = np.array(sct.grab(target_monitor))
                frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2BGR)
                h, w, _ = frame.shape

                cell_h = h // self.rows
                cell_w = w // self.cols

                for r in range(self.rows):
                    for c in range(self.cols):
                        cam_id = r * self.cols + c + 1
                        y1, y2 = r * cell_h, (r + 1) * cell_h
                        x1, x2 = c * cell_w, (c + 1) * cell_w
                        tile = frame[y1:y2, x1:x2]

                        th = y2 - y1
                        masked_tile = tile.copy()
                        masked_tile[0:int(th * 0.12), :] = 0
                        masked_tile[int(th * 0.90):, :] = 0

                        if cam_id not in self.subtractors:
                            self.subtractors[cam_id] = cv2.createBackgroundSubtractorMOG2(
                                history=150, varThreshold=self.sensitivity, detectShadows=False
                            )

                        fg = self.subtractors[cam_id].apply(masked_tile)
                        _, thresh = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        boxes = []
                        min_area = max(500, 2500 - (self.sensitivity * 50))

                        for cnt in contours:
                            if cv2.contourArea(cnt) > min_area:
                                bx, by, bw, bh = cv2.boundingRect(cnt)
                                boxes.append((x1 + bx, y1 + by, bw, bh))

                        if boxes:
                            label = "PERSON/VEHICLE" if self.mode == "AI" else "MOTION"
                            self.alert_detected.emit(cam_id, label, tile, boxes)

                elapsed = time.time() - loop_start
                sleep_dur = max(0.01, 0.12 - elapsed)
                time.sleep(sleep_dur)

    def stop(self):
        self.running = False
        self.wait()

class SentinelOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.alerts = {}
        self.boxes = []
        self.overlay_enabled = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.decrement_ttl)
        self.timer.start(80)

    def enable_click_through(self):
        hwnd = int(self.winId())
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
        try:
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass

    def trigger(self, cam_id, tile_rect, boxes):
        if not self.overlay_enabled:
            return
        self.alerts[cam_id] = {"rect": tile_rect, "ttl": 15}
        self.boxes = boxes
        self.update()

    def decrement_ttl(self):
        active = {}
        for cam, v in self.alerts.items():
            if v["ttl"] > 1:
                v["ttl"] -= 1
                active[cam] = v
        self.alerts = active
        if not self.alerts:
            self.boxes = []
        self.update()

    def paintEvent(self, event):
        if not self.overlay_enabled or not self.alerts:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for cam_id, val in self.alerts.items():
            x, y, w, h = val["rect"]
            pen = QPen(QColor(200, 42, 29, 230), 4)
            painter.setPen(pen)
            painter.drawRect(x, y, w, h)

            painter.fillRect(x, y, 170, 36, QColor(200, 42, 29, 240))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            painter.drawText(x + 12, y + 24, f"CAM {cam_id} ALERT")

        box_pen = QPen(QColor(0, 255, 128, 240), 2)
        painter.setPen(box_pen)
        for (bx, by, bw, bh) in self.boxes:
            painter.drawRect(bx, by, bw, bh)

class SentinelControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AlfaDefence Sentinel - Enterprise Suite")
        self.resize(800, 580)
        self.worker = None
        self.last_alert_times = {}

        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        self.init_csv()
        self.load_config()
        self.init_ui()
        self.apply_branding()

        self.overlay = SentinelOverlay()
        self.realign_overlay()
        self.overlay.show()
        self.overlay.enable_click_through()

    def apply_branding(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0E0E10; color: #FFFFFF; font-family: 'Segoe UI', sans-serif; }
            QLabel { color: #E0E0E0; font-weight: 600; font-size: 13px; }
            QComboBox, QSlider { background-color: #1A1A1D; color: #FFF; border: 1px solid #333338; padding: 4px; border-radius: 4px; }
            QPushButton { 
                background-color: #C82A1D; 
                color: #FFFFFF; 
                font-weight: 700; 
                border-radius: 4px; 
                padding: 9px 18px; 
                font-size: 13px; 
            }
            QPushButton:hover { background-color: #E23627; }
            QTableWidget { 
                background-color: #141416; 
                border: 1px solid #2B2B30; 
                color: #F0F0F0; 
                gridline-color: #232326; 
                selection-background-color: #C82A1D; 
            }
            QHeaderView::section { 
                background-color: #1C1C1F; 
                color: #DDD; 
                font-weight: 700; 
                border: 1px solid #2B2B30; 
                padding: 5px; 
            }
        """)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)

        title_box = QHBoxLayout()
        logo_label = QLabel("ALFADEFENCE SENTINEL")
        logo_label.setStyleSheet("color: #C82A1D; font-size: 20px; font-weight: 900; letter-spacing: 2px;")
        sub_label = QLabel("| 24/7 VIDEO WALL SENTINEL")
        sub_label.setStyleSheet("color: #888888; font-size: 12px; margin-top: 5px;")
        title_box.addWidget(logo_label)
        title_box.addWidget(sub_label)
        title_box.addStretch()
        layout.addLayout(title_box)

        ctrl_grid = QHBoxLayout()
        ctrl_grid.addWidget(QLabel("Monitor:"))
        self.mon_combo = QComboBox()
        with mss.mss() as sct:
            for idx in range(1, len(sct.monitors)):
                self.mon_combo.addItem(f"Display {idx}")
        ctrl_grid.addWidget(self.mon_combo)

        ctrl_grid.addWidget(QLabel("Grid:"))
        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["3x3 (9 Feeds)", "4x4 (16 Feeds)", "2x2 (4 Feeds)", "1x1 (Full)"])
        self.grid_combo.setCurrentText(self.config.get("grid", "3x3 (9 Feeds)"))
        ctrl_grid.addWidget(self.grid_combo)

        ctrl_grid.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Motion Mode", "AI (Person/Vehicle)"])
        self.mode_combo.setCurrentText(self.config.get("mode", "Motion Mode"))
        ctrl_grid.addWidget(self.mode_combo)
        layout.addLayout(ctrl_grid)

        sub_ctrl = QHBoxLayout()
        sub_ctrl.addWidget(QLabel("Sensitivity:"))
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(5, 50)
        self.sens_slider.setValue(self.config.get("sensitivity", 25))
        sub_ctrl.addWidget(self.sens_slider)

        self.audio_chk = QCheckBox("Sound Alert")
        self.audio_chk.setChecked(self.config.get("sound", True))
        sub_ctrl.addWidget(self.audio_chk)

        self.hide_overlay_btn = QPushButton("Hide Overlay")
        self.hide_overlay_btn.setStyleSheet("background-color: #2D2D32;")
        self.hide_overlay_btn.clicked.connect(self.toggle_overlay)
        sub_ctrl.addWidget(self.hide_overlay_btn)

        self.start_btn = QPushButton("START MONITORING")
        self.start_btn.clicked.connect(self.toggle_worker)
        sub_ctrl.addWidget(self.start_btn)
        layout.addLayout(sub_ctrl)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Camera", "Event", "Snapshot Path"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        bottom_bar = QHBoxLayout()
        self.status_lbl = QLabel("System Standby")
        self.status_lbl.setStyleSheet("color: #00FF88; font-size: 11px;")
        bottom_bar.addWidget(self.status_lbl)
        bottom_bar.addStretch()

        self.export_btn = QPushButton("Export Log (.csv)")
        self.export_btn.setStyleSheet("background-color: #242428; border: 1px solid #3F3F46; padding: 6px 14px;")
        self.export_btn.clicked.connect(self.export_csv)
        bottom_bar.addWidget(self.export_btn)
        layout.addLayout(bottom_bar)

    def realign_overlay(self):
        with mss.mss() as sct:
            idx = self.mon_combo.currentIndex() + 1
            idx = min(idx, len(sct.monitors) - 1)
            m = sct.monitors[idx]
            self.overlay.setGeometry(m["left"], m["top"], m["width"], m["height"])

    def toggle_overlay(self):
        self.overlay.overlay_enabled = not self.overlay.overlay_enabled
        self.hide_overlay_btn.setText("Hide Overlay" if self.overlay.overlay_enabled else "Show Overlay")
        self.overlay.update()

    def toggle_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.start_btn.setText("START MONITORING")
            self.start_btn.setStyleSheet("background-color: #C82A1D;")
            self.status_lbl.setText("Monitoring Stopped")
            self.status_lbl.setStyleSheet("color: #E23627;")
        else:
            grid_text = self.grid_combo.currentText()
            rows, cols = (4, 4) if "4x4" in grid_text else (2, 2) if "2x2" in grid_text else (1, 1) if "1x1" in grid_text else (3, 3)
            mode = "AI" if "AI" in self.mode_combo.currentText() else "Motion"
            sens = self.sens_slider.value()
            mon_idx = self.mon_combo.currentIndex() + 1

            self.realign_overlay()
            self.worker = SentinelWorker(mon_idx, rows, cols, mode, sens)
            self.worker.alert_detected.connect(self.handle_alert)
            self.worker.start()

            self.start_btn.setText("STOP MONITORING")
            self.start_btn.setStyleSheet("background-color: #1A1A1E; border: 1px solid #C82A1D;")
            self.status_lbl.setText(f"Active Monitoring: {rows}x{cols} Wall ({mode} Mode)")
            self.status_lbl.setStyleSheet("color: #00FF88;")

    def handle_alert(self, cam_id, label, tile_img, boxes):
        now = time.time()
        if now - self.last_alert_times.get(cam_id, 0) < 1.5:
            return
        self.last_alert_times[cam_id] = now

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snap_path = os.path.join(SNAPSHOT_DIR, f"cam{cam_id}_{file_ts}.jpg")

        cv2.imwrite(snap_path, tile_img)

        if self.audio_chk.isChecked():
            try:
                winsound.Beep(1200, 150)
            except Exception:
                pass

        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(timestamp_str))
        self.table.setItem(row, 1, QTableWidgetItem(f"CAM {cam_id}"))
        self.table.setItem(row, 2, QTableWidgetItem(label))
        self.table.setItem(row, 3, QTableWidgetItem(snap_path))
        self.table.scrollToBottom()

        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([timestamp_str, f"CAM {cam_id}", label, snap_path])

        with mss.mss() as sct:
            idx = self.mon_combo.currentIndex() + 1
            m = sct.monitors[min(idx, len(sct.monitors) - 1)]
            w, h = m["width"], m["height"]

        rows, cols = self.worker.rows, self.worker.cols
        cw, ch = w // cols, h // rows
        r = (cam_id - 1) // cols
        c = (cam_id - 1) % cols
        tile_rect = (c * cw, r * ch, cw, ch)

        self.overlay.trigger(cam_id, tile_rect, boxes)

    def init_csv(self):
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, "w", newline="") as f:
                csv.writer(f).writerow(["Timestamp", "Camera", "Detection_Type", "Snapshot_File"])

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Report", "AlfaDefence_Shift_Report.csv", "CSV (*.csv)")
        if path:
            import shutil
            shutil.copy(CSV_FILE, path)
            QMessageBox.information(self, "Export Successful", f"Shift report saved to:\n{path}")

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    self.config = json.load(f)
            except Exception:
                self.config = {}
        else:
            self.config = {}

    def closeEvent(self, event):
        cfg = {
            "grid": self.grid_combo.currentText(),
            "mode": self.mode_combo.currentText(),
            "sensitivity": self.sens_slider.value(),
            "sound": self.audio_chk.isChecked()
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=4)

        if self.worker and self.worker.isRunning():
            self.worker.stop()
        self.overlay.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    panel = SentinelControlPanel()
    panel.show()
    sys.exit(app.exec())
