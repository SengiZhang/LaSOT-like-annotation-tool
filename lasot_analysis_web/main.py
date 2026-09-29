from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, QSettings, Qt
from PySide6.QtGui import QAction, QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from analysis_data import (
    SEARCH_MODE_INDEPENDENT,
    SEARCH_MODE_SYNC_GROUND_TRUTH,
    SEARCH_MODE_SYNC_PREDICTION,
    DataError,
    SequenceAnalysis,
    SequenceSource,
    build_analyses,
    scan_dataset,
    scan_predictions,
    search_reference_boxes,
)
from widgets import GREEN, ORANGE, RED, YELLOW, ImageCanvas, MetricPlot, PlotController, PreviewImage


APP_NAME = "LaSOT 测试集分析工具"


def button(text: str, object_name: str = "") -> QPushButton:
    result = QPushButton(text)
    if object_name:
        result.setObjectName(object_name)
    result.setCursor(Qt.CursorShape.PointingHandCursor)
    return result


class DetailDialog(QDialog):
    def __init__(self, analysis: SequenceAnalysis, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.analysis = analysis
        self.images = analysis.source.load_images()
        self.frame_count = min(analysis.frame_count, len(self.images))
        if self.frame_count <= 0:
            raise DataError("该序列没有可读取的图像帧")
        self.setWindowTitle(f"{analysis.name} · 详细分析")
        self.resize(1450, 920)
        self.controller = PlotController(self.frame_count, self)
        self.controller.changed.connect(self.update_frame)
        self._shown_frame = -1

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        topbar = QHBoxLayout()
        title = QLabel(analysis.name)
        title.setObjectName("detailTitle")
        self.frame_label = QLabel()
        self.frame_label.setObjectName("framePill")
        mode_label = QLabel("搜索区域模式")
        mode_label.setObjectName("hint")
        self.search_mode = QComboBox()
        self.search_mode.addItem("不同步", SEARCH_MODE_INDEPENDENT)
        self.search_mode.addItem("同步真值区域", SEARCH_MODE_SYNC_GROUND_TRUTH)
        self.search_mode.addItem("同步预测值区域", SEARCH_MODE_SYNC_PREDICTION)
        self.search_mode.setToolTip("控制真值和预测搜索区域使用哪一个上一帧 BBOX")
        self.search_mode.currentIndexChanged.connect(self.search_mode_changed)
        hint = QLabel("图像：Ctrl+拖动平移 · Ctrl+滚轮缩放 · Ctrl+空格复位    图表：←/→逐帧 · Ctrl+滚轮缩放横轴")
        hint.setObjectName("hint")
        topbar.addWidget(title)
        topbar.addWidget(self.frame_label)
        topbar.addSpacing(12)
        topbar.addWidget(mode_label)
        topbar.addWidget(self.search_mode)
        topbar.addStretch()
        topbar.addWidget(hint)
        root.addLayout(topbar)

        vertical = QSplitter(Qt.Orientation.Vertical)
        image_container = QWidget()
        image_layout = QHBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.full_canvas = ImageCanvas()
        self.full_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        truth_label = QLabel("真值搜索区域")
        truth_label.setObjectName("canvasLabel")
        prediction_label = QLabel("预测搜索区域")
        prediction_label.setObjectName("canvasLabel")
        self.truth_canvas = ImageCanvas()
        self.prediction_canvas = ImageCanvas()
        side_layout.addWidget(truth_label)
        side_layout.addWidget(self.truth_canvas, 1)
        side_layout.addWidget(prediction_label)
        side_layout.addWidget(self.prediction_canvas, 1)
        image_layout.addWidget(self.full_canvas, 2)
        image_layout.addWidget(side, 1)

        plots = QWidget()
        plot_layout = QVBoxLayout(plots)
        plot_layout.setContentsMargins(0, 3, 0, 0)
        self.iou_plot = MetricPlot(
            analysis.iou[: self.frame_count],
            analysis.full_occlusion[: self.frame_count],
            analysis.out_of_view[: self.frame_count],
            "IoU",
            self.controller,
            fixed_max=1.0,
        )
        self.distance_plot = MetricPlot(
            analysis.center_distance[: self.frame_count],
            analysis.full_occlusion[: self.frame_count],
            analysis.out_of_view[: self.frame_count],
            "中心距离 / px",
            self.controller,
        )
        plot_layout.addWidget(self.iou_plot)
        plot_layout.addWidget(self.distance_plot)
        vertical.addWidget(image_container)
        vertical.addWidget(plots)
        vertical.setSizes([500, 390])
        root.addWidget(vertical, 1)
        self.update_frame()

    @staticmethod
    def _search_rect(box: np.ndarray) -> QRectF:
        x, y, width, height = [float(value) for value in box]
        side = max(2.0, 4.0 * np.sqrt(max(0.25, width * height)))
        center_x, center_y = x + width / 2, y + height / 2
        return QRectF(center_x - side / 2, center_y - side / 2, side, side)

    def search_mode_changed(self) -> None:
        self._shown_frame = -1
        self.update_frame()

    def update_frame(self) -> None:
        frame = self.controller.frame
        if frame == self._shown_frame:
            return
        self._shown_frame = frame
        path = self.images[frame]
        gt = self.analysis.ground_truth[frame]
        prediction = self.analysis.prediction[frame]
        truth_reference, prediction_reference = search_reference_boxes(
            self.analysis.ground_truth,
            self.analysis.prediction,
            frame,
            self.search_mode.currentData(),
        )
        truth_search = self._search_rect(truth_reference)
        prediction_search = self._search_rect(prediction_reference)
        self.frame_label.setText(f"第 {frame + 1} / {self.frame_count} 帧")
        self.full_canvas.set_content(path, [(gt, RED), (prediction, GREEN)])
        self.truth_canvas.set_content(path, [(gt, RED)], truth_search)
        self.prediction_canvas.set_content(path, [(prediction, GREEN)], prediction_search)


class SequenceRow(QFrame):
    def __init__(self, analysis: SequenceAnalysis, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.analysis = analysis
        self.setObjectName("sequenceRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("双击进入详细分析")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(14)
        info = QWidget()
        info.setFixedWidth(225)
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(0, 0, 0, 0)
        name = QLabel(analysis.name)
        name.setObjectName("sequenceName")
        preview = PreviewImage(analysis.preview_image, analysis.ground_truth[0], analysis.prediction[0])
        stats = QLabel(f"{analysis.frame_count} 帧  ·  平均 IoU {np.mean(analysis.iou):.3f}")
        stats.setObjectName("hint")
        info_layout.addWidget(name)
        info_layout.addWidget(preview)
        info_layout.addWidget(stats)
        plots = QWidget()
        plot_layout = QVBoxLayout(plots)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(6)
        self.controller = PlotController(analysis.frame_count, self)
        plot_layout.addWidget(
            MetricPlot(analysis.iou, analysis.full_occlusion, analysis.out_of_view, "IoU", self.controller, 1.0, compact=True)
        )
        plot_layout.addWidget(
            MetricPlot(analysis.center_distance, analysis.full_occlusion, analysis.out_of_view, "中心距离 / px", self.controller, compact=True)
        )
        layout.addWidget(info)
        layout.addWidget(plots, 1)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                dialog = DetailDialog(self.analysis, self)
            except (OSError, DataError) as exc:
                QMessageBox.critical(self, "详细视图打开失败", str(exc))
                return
            finally:
                QApplication.restoreOverrideCursor()
            dialog.exec()
        else:
            super().mouseDoubleClickEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.dataset: dict[str, SequenceSource] = {}
        self.predictions: dict[str, Path] = {}
        self.dataset_path: Path | None = None
        self.prediction_path: Path | None = None
        self.settings = QSettings("MyLabel", "LaSOTAnalysis")
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 920)
        self.setMinimumSize(1024, 680)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 14, 18, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        subtitle = QLabel("导入 LaSOT 类数据集和跟踪结果，逐序列比较目标框、IoU 与中心距离")
        subtitle.setObjectName("hint")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        self.dataset_button = button("打开数据集", "primaryButton")
        self.file_button = button("打开单个预测 TXT")
        self.folder_button = button("打开预测结果文件夹")
        self.dataset_button.clicked.connect(self.open_dataset)
        self.file_button.clicked.connect(self.open_prediction_file)
        self.folder_button.clicked.connect(self.open_prediction_folder)
        header.addLayout(title_box)
        header.addStretch()
        header.addWidget(self.dataset_button)
        header.addWidget(self.file_button)
        header.addWidget(self.folder_button)
        root.addLayout(header)

        paths = QFrame()
        paths.setObjectName("pathBar")
        paths_layout = QHBoxLayout(paths)
        paths_layout.setContentsMargins(12, 8, 12, 8)
        self.dataset_label = QLabel("数据集：未打开")
        self.prediction_label = QLabel("预测结果：未打开")
        paths_layout.addWidget(self.dataset_label, 1)
        paths_layout.addWidget(self.prediction_label, 1)
        root.addWidget(paths)

        legend = QHBoxLayout()
        self.summary_label = QLabel("请先打开数据集和预测结果")
        self.summary_label.setObjectName("summary")
        legend.addWidget(self.summary_label)
        legend.addStretch()
        legend.addWidget(self._legend_item(RED, "真值框"))
        legend.addWidget(self._legend_item(GREEN, "预测框"))
        legend.addWidget(self._legend_item(YELLOW, "完全遮挡"))
        legend.addWidget(self._legend_item(ORANGE, "出视野"))
        root.addLayout(legend)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.rows_widget = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(10)
        self.rows_layout.addStretch()
        self.scroll.setWidget(self.rows_widget)
        root.addWidget(self.scroll, 1)
        self.setStatusBar(QStatusBar())

        quit_action = QAction("退出", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

    @staticmethod
    def _legend_item(color: QColor, text: str) -> QLabel:
        label = QLabel(f"● {text}")
        label.setStyleSheet(f"color: {color.name()}; font-weight: 600")
        return label

    def _initial_dir(self) -> str:
        return self.settings.value("lastDirectory", str(Path.home()), str)

    def open_dataset(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择单序列或多序列数据集文件夹", self._initial_dir())
        if not selected:
            return
        try:
            dataset, warnings = scan_dataset(Path(selected))
        except (OSError, DataError) as exc:
            QMessageBox.critical(self, "数据集打开失败", str(exc))
            return
        self.dataset, self.dataset_path = dataset, Path(selected)
        self.settings.setValue("lastDirectory", selected)
        self.dataset_label.setText(f"数据集：{selected}")
        self.statusBar().showMessage(f"已读取 {len(dataset)} 个序列", 5000)
        self._show_warnings("数据集读取提示", warnings)
        self.refresh_results()

    def open_prediction_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "选择单序列预测结果", self._initial_dir(), "文本文件 (*.txt)")
        if selected:
            self._load_predictions(Path(selected))

    def open_prediction_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择多序列预测结果文件夹", self._initial_dir())
        if selected:
            self._load_predictions(Path(selected))

    def _load_predictions(self, path: Path) -> None:
        try:
            predictions, warnings = scan_predictions(path)
        except (OSError, DataError) as exc:
            QMessageBox.critical(self, "预测结果打开失败", str(exc))
            return
        self.predictions, self.prediction_path = predictions, path
        self.settings.setValue("lastDirectory", str(path.parent if path.is_file() else path))
        self.prediction_label.setText(f"预测结果：{path}")
        self.statusBar().showMessage(f"已读取 {len(predictions)} 个预测文件", 5000)
        self._show_warnings("预测结果读取提示", warnings)
        self.refresh_results()

    def _clear_rows(self) -> None:
        while self.rows_layout.count() > 1:
            item = self.rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def refresh_results(self) -> None:
        self._clear_rows()
        if not self.dataset or not self.predictions:
            self.summary_label.setText(f"数据集 {len(self.dataset)} 个序列 · 预测结果 {len(self.predictions)} 个文件")
            return
        analyses, missing, extra, warnings = build_analyses(self.dataset, self.predictions)
        for analysis in analyses:
            self.rows_layout.insertWidget(self.rows_layout.count() - 1, SequenceRow(analysis))
        self.summary_label.setText(f"成功匹配 {len(analyses)} 个序列 · 双击任一行查看详情")
        messages = []
        if missing:
            messages.append("数据集中缺少预测结果：\n" + "、".join(missing))
        if extra:
            messages.append("预测结果中没有对应数据序列：\n" + "、".join(extra))
        messages.extend(warnings)
        self._show_warnings("匹配与解析提示", messages)
        if not analyses:
            self.summary_label.setText("没有名称相同且可分析的序列")

    def _show_warnings(self, title: str, warnings: list[str]) -> None:
        if warnings:
            text = "\n\n".join(warnings[:30])
            if len(warnings) > 30:
                text += f"\n\n……另有 {len(warnings) - 30} 项"
            QMessageBox.warning(self, title, text)


def stylesheet() -> str:
    return """
        QWidget { font-family: "Microsoft YaHei UI"; font-size: 13px; color: #111827; }
        QMainWindow, QDialog { background: #f3f5f8; }
        QLabel#appTitle { font-size: 25px; font-weight: 700; color: #0f172a; }
        QLabel#detailTitle { font-size: 21px; font-weight: 700; }
        QLabel#sequenceName { font-size: 17px; font-weight: 700; }
        QLabel#hint { color: #64748b; }
        QLabel#summary { font-weight: 600; color: #334155; }
        QLabel#framePill { background: #dbeafe; color: #1d4ed8; border-radius: 10px; padding: 4px 10px; font-weight: 600; }
        QLabel#canvasLabel { color: #475569; font-weight: 600; }
        QPushButton { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 7px; padding: 8px 13px; }
        QPushButton:hover { border-color: #2563eb; color: #1d4ed8; }
        QPushButton#primaryButton { background: #2563eb; color: white; border-color: #2563eb; font-weight: 600; }
        QPushButton#primaryButton:hover { background: #1d4ed8; }
        QFrame#pathBar { background: white; border: 1px solid #e2e8f0; border-radius: 8px; }
        QFrame#sequenceRow { background: white; border: 1px solid #e2e8f0; border-radius: 10px; }
        QFrame#sequenceRow:hover { border-color: #93c5fd; }
        QScrollArea { background: transparent; }
        QStatusBar { background: white; }
        QSplitter::handle { background: #cbd5e1; height: 4px; }
    """


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("MyLabel")
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    app.setStyleSheet(stylesheet())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
