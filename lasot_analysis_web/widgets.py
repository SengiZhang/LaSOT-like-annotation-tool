from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal, QObject
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget


RED = QColor("#ef4444")
GREEN = QColor("#22c55e")
YELLOW = QColor(250, 204, 21, 95)
ORANGE = QColor(249, 115, 22, 95)


def load_image(path: Path) -> QImage:
    image = QImage(str(path))
    if image.isNull():
        try:
            image.loadFromData(path.read_bytes())
        except OSError:
            pass
    return image


class PlotController(QObject):
    changed = Signal()

    def __init__(self, count: int, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.count = max(1, count)
        self.frame = 0
        self.view_start = 0
        self.view_end = self.count - 1

    def set_frame(self, frame: int, ensure_visible: bool = True) -> None:
        frame = max(0, min(self.count - 1, int(frame)))
        span = self.view_end - self.view_start
        changed = frame != self.frame
        self.frame = frame
        if ensure_visible and frame < self.view_start:
            self.view_start = frame
            self.view_end = min(self.count - 1, frame + span)
            changed = True
        elif ensure_visible and frame > self.view_end:
            self.view_end = frame
            self.view_start = max(0, frame - span)
            changed = True
        if changed:
            self.changed.emit()

    def set_view(self, start: int, end: int) -> None:
        start, end = int(start), int(end)
        if self.count <= 1:
            start = end = 0
        else:
            start = max(0, min(self.count - 2, start))
            end = max(start + 1, min(self.count - 1, end))
        if (start, end) != (self.view_start, self.view_end):
            self.view_start, self.view_end = start, end
            if self.frame < start:
                self.frame = start
            elif self.frame > end:
                self.frame = end
            self.changed.emit()

    def zoom(self, anchor: int, factor: float) -> None:
        old_span = self.view_end - self.view_start + 1
        new_span = max(2, min(self.count, int(round(old_span * factor))))
        if new_span == old_span:
            return
        ratio = 0.5 if old_span <= 1 else (anchor - self.view_start) / (old_span - 1)
        start = round(anchor - ratio * (new_span - 1))
        start = max(0, min(self.count - new_span, start))
        self.set_view(start, start + new_span - 1)


class MetricPlot(QWidget):
    def __init__(
        self,
        values: np.ndarray,
        occlusion: np.ndarray,
        out_of_view: np.ndarray,
        title: str,
        controller: PlotController,
        fixed_max: float | None = None,
        compact: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.values = np.asarray(values, dtype=float)
        self.occlusion = np.asarray(occlusion, dtype=np.uint8)
        self.out_of_view = np.asarray(out_of_view, dtype=np.uint8)
        self.title = title
        self.fixed_max = fixed_max
        self.controller = controller
        self.compact = compact
        self.dragging = False
        self.panning_view = False
        self.panning_track = False
        self.pan_origin_x = 0.0
        self.pan_origin_start = 0
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(126 if compact else 190)
        controller.changed.connect(self.update)

    def enterEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().enterEvent(event)

    def _plot_rect(self) -> QRectF:
        return QRectF(48, 27, max(40, self.width() - 62), max(36, self.height() - 70))

    def _track_rect(self) -> QRectF:
        return QRectF(48, self.height() - 29, max(40, self.width() - 62), 12)

    def _viewport_rect(self) -> QRectF:
        track = self._track_rect()
        full_span = max(1, self.controller.count - 1)
        left = track.left() + self.controller.view_start / full_span * track.width()
        right = track.left() + self.controller.view_end / full_span * track.width()
        return QRectF(left, track.top() + 1, max(8, right - left), track.height() - 2)

    def _x_for_frame(self, frame: int, rect: QRectF | None = None) -> float:
        rect = rect or self._plot_rect()
        span = max(1, self.controller.view_end - self.controller.view_start)
        return rect.left() + (frame - self.controller.view_start) / span * rect.width()

    def _frame_for_x(self, x: float, visible_only: bool = True) -> int:
        rect = self._plot_rect()
        start = self.controller.view_start if visible_only else 0
        end = self.controller.view_end if visible_only else self.controller.count - 1
        ratio = (x - rect.left()) / max(1.0, rect.width())
        return round(start + max(0.0, min(1.0, ratio)) * (end - start))

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#ffffff"))
        plot = self._plot_rect()
        track = self._track_rect()
        start, end = self.controller.view_start, self.controller.view_end
        visible = self.values[start : end + 1]
        ymax = self.fixed_max if self.fixed_max is not None else max(1.0, float(np.nanmax(visible)) * 1.08 if len(visible) else 1.0)

        p.setPen(QPen(QColor("#e5e7eb"), 1))
        for i in range(6):
            y = plot.top() + i * plot.height() / 5
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        for i in range(9):
            x = plot.left() + i * plot.width() / 8
            p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

        frame_width = plot.width() / max(1, end - start + 1)
        for frame in range(start, end + 1):
            x1 = self._x_for_frame(frame, plot) - frame_width / 2
            area = QRectF(x1, plot.top(), frame_width + 1, plot.height())
            if frame < len(self.occlusion) and self.occlusion[frame]:
                p.fillRect(area, YELLOW)
            if frame < len(self.out_of_view) and self.out_of_view[frame]:
                p.fillRect(area, ORANGE)

        if len(visible):
            path = QPainterPath()
            for offset, value in enumerate(visible):
                x = self._x_for_frame(start + offset, plot)
                y = plot.bottom() - max(0.0, min(ymax, float(value))) / ymax * plot.height()
                path.moveTo(x, y) if offset == 0 else path.lineTo(x, y)
            p.setPen(QPen(QColor("#2563eb"), 1.6))
            p.drawPath(path)

        current_x = self._x_for_frame(self.controller.frame, plot)
        p.setPen(QPen(QColor("#111827"), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(current_x, plot.top()), QPointF(current_x, plot.bottom()))
        p.setPen(QColor("#111827"))
        value = self.values[self.controller.frame]
        p.drawText(8, 18, f"{self.title}  第 {self.controller.frame + 1} 帧：{value:.4g}")
        if len(self.values):
            p.setPen(QColor("#6b7280"))
            p.drawText(int(plot.right() - 175), 18, f"最小 {np.nanmin(self.values):.4g}   最大 {np.nanmax(self.values):.4g}")
        p.setPen(QColor("#6b7280"))
        p.drawText(5, int(plot.top() + 5), f"{ymax:.3g}")
        p.drawText(25, int(plot.bottom()), "0")
        p.drawText(int(plot.left()), int(plot.bottom() + 14), str(start + 1))
        p.drawText(int(plot.right() - 35), int(plot.bottom() + 14), str(end + 1))

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#d1d5db"))
        p.drawRoundedRect(track, 6, 6)
        full_span = max(1, self.controller.count - 1)
        if start > 0 or end < self.controller.count - 1:
            p.setBrush(QColor("#94a3b8"))
            p.drawRoundedRect(self._viewport_rect(), 3, 3)
        knob_x = self._x_for_frame(self.controller.frame, track)
        p.setBrush(QColor("#2563eb"))
        p.drawEllipse(QPointF(knob_x, track.center().y()), 7, 7)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if event.button() == Qt.MouseButton.LeftButton:
            viewport = self._viewport_rect()
            knob_x = self._x_for_frame(self.controller.frame, self._track_rect())
            if (
                (self.controller.view_start > 0 or self.controller.view_end < self.controller.count - 1)
                and viewport.contains(event.position())
                and abs(event.position().x() - knob_x) > 9
            ):
                self.panning_track = True
                self.pan_origin_x = event.position().x()
                self.pan_origin_start = self.controller.view_start
            elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and self._plot_rect().contains(event.position()):
                self.panning_view = True
                self.pan_origin_x = event.position().x()
                self.pan_origin_start = self.controller.view_start
            else:
                self.dragging = True
                self.controller.set_frame(self._frame_for_x(event.position().x()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.dragging:
            x = event.position().x()
            rect = self._plot_rect()
            if x < rect.left() and self.controller.view_start > 0:
                self.controller.set_view(self.controller.view_start - 1, self.controller.view_end - 1)
            elif x > rect.right() and self.controller.view_end < self.controller.count - 1:
                self.controller.set_view(self.controller.view_start + 1, self.controller.view_end + 1)
            self.controller.set_frame(self._frame_for_x(x))
        elif self.panning_view:
            span = self.controller.view_end - self.controller.view_start
            delta = round((self.pan_origin_x - event.position().x()) / max(1, self._plot_rect().width()) * max(1, span))
            start = max(0, min(self.controller.count - span - 1, self.pan_origin_start + delta))
            self.controller.set_view(start, start + span)
        elif self.panning_track:
            span = self.controller.view_end - self.controller.view_start
            delta = round((event.position().x() - self.pan_origin_x) / max(1, self._track_rect().width()) * max(1, self.controller.count - 1))
            start = max(0, min(self.controller.count - span - 1, self.pan_origin_start + delta))
            self.controller.set_view(start, start + span)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.dragging = False
        self.panning_view = False
        self.panning_track = False

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.controller.set_frame(self._frame_for_x(event.position().x()))

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and not self.compact:
            anchor = self._frame_for_x(event.position().x())
            self.controller.zoom(anchor, 0.75 if event.angleDelta().y() > 0 else 1.34)
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            delta = -1 if event.key() == Qt.Key.Key_Left else 1
            self.controller.set_frame(self.controller.frame + delta, ensure_visible=True)
            event.accept()
        else:
            super().keyPressEvent(event)


class ImageCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image = QImage()
        self.boxes: list[tuple[np.ndarray, QColor]] = []
        self.base_source: QRectF | None = None
        self.zoom = 1.0
        self.pan = QPointF()
        self.dragging = False
        self.last_mouse = QPointF()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(240, 180)

    def set_content(self, image_path: Path, boxes: list[tuple[np.ndarray, QColor]], source: QRectF | None = None) -> None:
        self.image = load_image(image_path)
        self.boxes = boxes
        self.base_source = source
        self.reset_view()

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan = QPointF()
        self.update()

    def enterEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().enterEvent(event)

    def _source_rect(self) -> QRectF:
        base = self.base_source or QRectF(0, 0, self.image.width(), self.image.height())
        width, height = base.width() / self.zoom, base.height() / self.zoom
        center = base.center() + self.pan
        return QRectF(center.x() - width / 2, center.y() - height / 2, width, height)

    def _target_rect(self, source: QRectF) -> QRectF:
        if source.width() <= 0 or source.height() <= 0:
            return QRectF()
        scale = min(self.width() / source.width(), self.height() / source.height())
        width, height = source.width() * scale, source.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111827"))
        if self.image.isNull():
            p.setPen(Qt.GlobalColor.white)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "无法读取图像")
            return
        source = self._source_rect()
        target = self._target_rect(source)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(target, self.image, source)
        sx, sy = target.width() / source.width(), target.height() / source.height()
        for box, color in self.boxes:
            x, y, w, h = (float(v) for v in box)
            mapped = QRectF(target.left() + (x - source.left()) * sx, target.top() + (y - source.top()) * sy, w * sx, h * sy)
            p.setPen(QPen(color, 2.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(mapped)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.dragging = True
            self.last_mouse = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.dragging and not self.image.isNull():
            source = self._source_rect()
            target = self._target_rect(source)
            delta = event.position() - self.last_mouse
            self.pan -= QPointF(delta.x() / max(1, target.width()) * source.width(), delta.y() / max(1, target.height()) * source.height())
            self.last_mouse = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.dragging = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier) or self.image.isNull():
            return super().wheelEvent(event)
        old_source = self._source_rect()
        target = self._target_rect(old_source)
        ratio_x = (event.position().x() - target.left()) / max(1, target.width())
        ratio_y = (event.position().y() - target.top()) / max(1, target.height())
        anchor = QPointF(old_source.left() + ratio_x * old_source.width(), old_source.top() + ratio_y * old_source.height())
        self.zoom = max(0.25, min(32.0, self.zoom * (1.2 if event.angleDelta().y() > 0 else 1 / 1.2)))
        new_source = self._source_rect()
        desired_center = QPointF(anchor.x() - (ratio_x - 0.5) * new_source.width(), anchor.y() - (ratio_y - 0.5) * new_source.height())
        base_center = (self.base_source or QRectF(0, 0, self.image.width(), self.image.height())).center()
        self.pan = desired_center - base_center
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.reset_view()
            event.accept()
        else:
            super().keyPressEvent(event)


class PreviewImage(QWidget):
    def __init__(self, path: Path, gt: np.ndarray, prediction: np.ndarray, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image = load_image(path)
        self.gt, self.prediction = gt, prediction
        self.setFixedSize(210, 130)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111827"))
        if self.image.isNull():
            return
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        target = QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)
        p.drawImage(target, self.image)
        for box, color in ((self.gt, RED), (self.prediction, GREEN)):
            x, y, w, h = [float(value) * scale for value in box]
            p.setPen(QPen(color, max(1.0, min(self.width(), self.height()) / 150)))
            p.drawRect(QRectF(target.left() + x, target.top() + y, w, h))
