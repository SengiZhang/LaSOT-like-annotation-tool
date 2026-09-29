from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QProcess, QProcessEnvironment, QRect, QRectF, QSettings, QSize, Qt, Signal, QThread, QTimer
from PySide6.QtGui import QAction, QColor, QCursor, QFont, QIcon, QImage, QIntValidator, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


APP_NAME = "MyLabel"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_FILTER = "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.mpeg *.mpg *.webm);;所有文件 (*.*)"
PROJECT_VERSION = 1


def bundled_path(filename: str) -> Path:
    """Return a resource path that works both from source and a PyInstaller EXE."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root / filename


def application_dir() -> Path:
    """Directory containing source files, or the installed EXE when frozen."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


# MCITrack is external and user-selected. Only its MyLabel adapter is bundled;
# the official repository, L384 checkpoint and Conda interpreter are not.
DEFAULT_TRACKING_CODE = ""
DEFAULT_TRACKING_MODEL = ""
DEFAULT_TRACKING_PYTHON = ""


def natural_key(path: Path) -> list[Any]:
    import re

    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread replacement that supports non-ASCII Windows paths."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()


def frame_id(index: int) -> str:
    return f"{index + 1:08d}"


def default_frame(index: int) -> dict[str, Any]:
    return {"id": frame_id(index), "box": None, "attributes": {}}


@dataclass
class SourceInfo:
    source_type: str
    path: Path
    total_frames: int
    width: int
    height: int
    fps: float = 0.0
    image_files: list[Path] | None = None


@dataclass
class UndoEntry:
    description: str
    focus_index: int
    box_changes: dict[int, list[int] | None]
    attribute_changes: dict[int, dict[str, tuple[bool, str]]]


class FrameSource:
    def __init__(self) -> None:
        self.info: SourceInfo | None = None
        self._capture: cv2.VideoCapture | None = None
        self._cache_index = -1
        self._cache_frame: np.ndarray | None = None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self.info = None
        self._cache_index = -1
        self._cache_frame = None

    def open_images(self, folder: Path) -> SourceInfo:
        files = sorted(
            (p for p in folder.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS),
            key=natural_key,
        )
        if not files:
            raise ValueError("所选文件夹中没有受支持的图片。")
        first = read_image(files[0])
        if first is None:
            raise ValueError(f"无法读取图片：{files[0].name}")
        self.close()
        height, width = first.shape[:2]
        self.info = SourceInfo("images", folder.resolve(), len(files), width, height, image_files=files)
        self._cache_index, self._cache_frame = 0, first
        return self.info

    def open_video(self, path: Path) -> SourceInfo:
        self.close()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("无法打开视频。请确认格式受支持且文件未损坏。")
        declared_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("无法获取视频帧信息。")
        # Container metadata can over-report frames when a recording was not
        # finalized cleanly. Count frames that the decoder can actually read so
        # navigation, project data, exports and tracking all share one boundary.
        total = 0
        while capture.grab():
            total += 1
        if total <= 0:
            capture.release()
            raise ValueError("视频中没有可解码的画面。")
        capture.release()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("统计帧数后无法重新打开视频。")
        self._capture = capture
        self.info = SourceInfo("video", path.resolve(), total, width, height, fps)
        return self.info

    def read(self, index: int) -> np.ndarray | None:
        if self.info is None or index < 0 or index >= self.info.total_frames:
            return None
        if index == self._cache_index and self._cache_frame is not None:
            return self._cache_frame.copy()
        if self.info.source_type == "images":
            assert self.info.image_files is not None
            frame = read_image(self.info.image_files[index])
        else:
            assert self._capture is not None
            if index != self._cache_index + 1:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self._capture.read()
            if not ok:
                frame = None
        if frame is not None:
            self._cache_index, self._cache_frame = index, frame.copy()
        return frame


class RemoteTrackingThread(QThread):
    """Upload frames sequentially while the server retains tracker state."""

    messageReceived = Signal(object)
    trackingCompleted = Signal(bool, str)

    def __init__(self, tracking_request: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tracking_request = tracking_request
        self.session_id = ""

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        token = str(self.tracking_request.get("server_token", "")).strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _api(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        base = str(self.tracking_request["server_url"]).rstrip("/")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(base + path, data=body, headers=self._headers(), method=method)
        timeout = float(self.tracking_request.get("server_timeout", 120))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                details = json.loads(exc.read().decode("utf-8")).get("error", "")
            except Exception:
                details = ""
            raise RuntimeError(details or f"服务端返回HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接自动标注服务器：{exc.reason}") from exc
        if not isinstance(result, dict) or not result.get("ok"):
            raise RuntimeError(str(result.get("error", "服务端返回格式无效。")) if isinstance(result, dict) else "服务端返回格式无效。")
        return result

    def _encode_frame(self, bgr: np.ndarray) -> str:
        import base64

        quality = int(self.tracking_request.get("jpeg_quality", 90))
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("无法压缩待上传的视频帧。")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _read_frames(self) -> Any:
        source_type = str(self.tracking_request["source_type"])
        if source_type == "images":
            paths = [Path(value) for value in self.tracking_request["frame_paths"]]

            def read(index: int) -> np.ndarray:
                frame = read_image(paths[index])
                if frame is None:
                    raise RuntimeError(f"无法读取图片：{paths[index]}")
                return frame

            return read, lambda: None
        capture = cv2.VideoCapture(str(self.tracking_request["source_path"]))
        if not capture.isOpened():
            raise RuntimeError("远程跟踪线程无法打开本地视频。")
        last_index = -1

        def read(index: int) -> np.ndarray:
            nonlocal last_index
            if index != last_index + 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"无法读取第 {index + 1} 帧。")
            last_index = index
            return frame

        return read, capture.release

    def run(self) -> None:
        stopped = False
        error = ""
        close_frames = lambda: None
        try:
            read_frame, close_frames = self._read_frames()
            start = int(self.tracking_request["start_index"])
            total = int(self.tracking_request["total_frames"])
            initial = read_frame(start)
            response = self._api("POST", "/api/v1/sessions", {
                "algorithm": self.tracking_request["algorithm"],
                "box": self.tracking_request["box"],
                "image_base64": self._encode_frame(initial),
            })
            self.session_id = str(response["session_id"])
            self.messageReceived.emit({
                "type": "ready",
                "message": f"服务端会话已建立，正在使用 {response.get('device', '服务器')} 跟踪…",
            })
            for index in range(start + 1, total):
                if self.isInterruptionRequested():
                    stopped = True
                    break
                frame = read_frame(index)
                response = self._api("POST", f"/api/v1/sessions/{self.session_id}/track", {
                    "frame_index": index,
                    "image_base64": self._encode_frame(frame),
                })
                self.messageReceived.emit({"type": "box", "index": index, "box": response["box"]})
            stopped = stopped or self.isInterruptionRequested()
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
        finally:
            try:
                close_frames()
            except Exception:
                pass
            if self.session_id:
                try:
                    self._api("DELETE", f"/api/v1/sessions/{self.session_id}")
                except Exception:
                    pass
            self.trackingCompleted.emit(stopped, error)


class ProjectStore:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.data: dict[str, Any] = {}
        self.dirty = False

    @staticmethod
    def suggested_path(info: SourceInfo) -> Path:
        if info.source_type == "video":
            return info.path.with_suffix(".json")
        return info.path / f"{info.path.name}.json"

    def create(self, info: SourceInfo, attribute_schema: list[dict[str, Any]] | None = None) -> None:
        schema = json.loads(json.dumps(attribute_schema or [], ensure_ascii=False))
        self.path = self.suggested_path(info)
        self.data = {
            "version": PROJECT_VERSION,
            "project": {
                "source_type": info.source_type,
                "source_path": str(info.path),
                "total_frames": info.total_frames,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
            },
            "attribute_schema": schema,
            "frames": [default_frame(i) for i in range(info.total_frames)],
        }
        self.dirty = True
        self.save()

    def load(self, path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取项目文件：{exc}") from exc
        project = data.get("project")
        frames = data.get("frames")
        if not isinstance(project, dict) or not isinstance(frames, list):
            raise ValueError("项目文件缺少 project 或 frames 数据。")
        required = {"source_type", "source_path", "total_frames"}
        if not required.issubset(project):
            raise ValueError("项目文件的源信息不完整。")
        data.setdefault("version", PROJECT_VERSION)
        data.setdefault("attribute_schema", [])
        total = int(project["total_frames"])
        if len(frames) < total:
            frames.extend(default_frame(i) for i in range(len(frames), total))
        for i, item in enumerate(frames[:total]):
            item.setdefault("id", frame_id(i))
            item.setdefault("box", None)
            item.setdefault("attributes", {})
        self.path, self.data, self.dirty = path.resolve(), data, False
        return data

    def save(self) -> None:
        if self.path is None or not self.data:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.data, ensure_ascii=False, indent=2)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.stem}_", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
            self.dirty = False
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


class AttributeTableWidget(QTableWidget):
    """Attribute table whose blank area clears the current cell selection."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)

    def mousePressEvent(self, event: Any) -> None:
        if not self.indexAt(event.position().toPoint()).isValid():
            self.clearSelection()
        super().mousePressEvent(event)


class ImageCanvas(QWidget):
    boxChanged = Signal(object)
    mouseImagePosition = Signal(object)
    copyRequested = Signal()
    pasteRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(480, 320)
        self._image = QImage()
        self._box: list[int] | None = None
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._drag_mode: str | None = None
        self._drag_original_box: list[int] | None = None
        self._target_rect = QRectF()
        self._zoom_factor = 1.0
        self._pan_offset = QPointF()
        self._pan_drag_start: QPoint | None = None
        self._pan_start_offset = QPointF()

    def sizeHint(self) -> QSize:
        return QSize(900, 600)

    def set_frame(self, image: QImage, box: list[int] | None) -> None:
        self._image = image
        self._box = list(box) if box else None
        self._drag_start = self._drag_current = None
        self._drag_mode = None
        self._drag_original_box = None
        self.update()

    def set_box(self, box: list[int] | None) -> None:
        self._box = list(box) if box else None
        self.update()

    def reset_view(self) -> None:
        self._zoom_factor = 1.0
        self._pan_offset = QPointF()
        self._pan_drag_start = None
        self._pan_start_offset = QPointF()
        if self._drag_mode == "pan":
            self._drag_mode = None
        self.unsetCursor()
        self.update()

    def _layout_rect(self) -> QRectF:
        if self._image.isNull():
            return QRectF()
        scaled = QSize(self._image.width(), self._image.height())
        scaled.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        width = scaled.width() * self._zoom_factor
        height = scaled.height() * self._zoom_factor
        left = (self.width() - width) / 2 + self._pan_offset.x()
        top = (self.height() - height) / 2 + self._pan_offset.y()
        return QRectF(left, top, width, height)

    def _zoom_at(self, widget_point: QPointF, factor: float) -> None:
        old_rect = self._layout_rect()
        if old_rect.isEmpty() or not old_rect.contains(widget_point):
            return
        new_zoom = min(max(self._zoom_factor * factor, 0.1), 20.0)
        if abs(new_zoom - self._zoom_factor) < 1e-9:
            return
        relative_x = (widget_point.x() - old_rect.left()) / old_rect.width()
        relative_y = (widget_point.y() - old_rect.top()) / old_rect.height()
        self._zoom_factor = new_zoom
        centered_rect = self._layout_rect()
        desired_left = widget_point.x() - relative_x * centered_rect.width()
        desired_top = widget_point.y() - relative_y * centered_rect.height()
        self._pan_offset += QPointF(desired_left - centered_rect.left(), desired_top - centered_rect.top())
        self.update()

    def _to_image(self, point: QPoint) -> QPoint | None:
        rect = self._layout_rect()
        if rect.isEmpty() or not rect.contains(point):
            return None
        x = int((point.x() - rect.left()) * self._image.width() / rect.width())
        y = int((point.y() - rect.top()) * self._image.height() / rect.height())
        return QPoint(min(max(x, 0), self._image.width() - 1), min(max(y, 0), self._image.height() - 1))

    def _resize_handle_at(self, point: QPoint) -> str | None:
        if not self._box or self._image.isNull():
            return None
        x, y, w, h = self._box
        right, bottom = x + w, y + h
        rect = self._layout_rect()
        tolerance_x = max(2, int(9 * self._image.width() / max(1.0, rect.width())))
        tolerance_y = max(2, int(9 * self._image.height() / max(1.0, rect.height())))
        near_left = abs(point.x() - x) <= tolerance_x
        near_right = abs(point.x() - right) <= tolerance_x
        near_top = abs(point.y() - y) <= tolerance_y
        near_bottom = abs(point.y() - bottom) <= tolerance_y
        within_x = x - tolerance_x <= point.x() <= right + tolerance_x
        within_y = y - tolerance_y <= point.y() <= bottom + tolerance_y
        horizontal = "left" if near_left else "right" if near_right else ""
        vertical = "top" if near_top else "bottom" if near_bottom else ""
        if horizontal and vertical:
            return f"resize_{horizontal}_{vertical}"
        if horizontal and within_y:
            return f"resize_{horizontal}"
        if vertical and within_x:
            return f"resize_{vertical}"
        return None

    def _point_in_box(self, point: QPoint) -> bool:
        if not self._box:
            return False
        x, y, w, h = self._box
        return x <= point.x() <= x + w and y <= point.y() <= y + h

    def _resize_box_to(self, point: QPoint) -> None:
        if not self._drag_original_box or not self._drag_mode:
            return
        x, y, w, h = self._drag_original_box
        left, top, right, bottom = x, y, x + w, y + h
        if "left" in self._drag_mode:
            left = min(point.x(), right - 1)
        if "right" in self._drag_mode:
            right = max(point.x(), left + 1)
        if "top" in self._drag_mode:
            top = min(point.y(), bottom - 1)
        if "bottom" in self._drag_mode:
            bottom = max(point.y(), top + 1)
        self._box = [left, top, right - left, bottom - top]

    def _move_box_to(self, point: QPoint) -> None:
        if not self._drag_original_box or self._drag_start is None or self._image.isNull():
            return
        x, y, w, h = self._drag_original_box
        new_x = x + point.x() - self._drag_start.x()
        new_y = y + point.y() - self._drag_start.y()
        new_x = min(max(new_x, 0), max(0, self._image.width() - w))
        new_y = min(max(new_y, 0), max(0, self._image.height() - h))
        self._box = [new_x, new_y, w, h]

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111827"))
        rect = self._layout_rect()
        self._target_rect = rect
        if self._image.isNull():
            painter.setPen(QColor("#94a3b8"))
            painter.setFont(QFont("Microsoft YaHei UI", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "打开视频或图片文件夹开始标注")
            return
        painter.drawImage(rect, self._image)
        box = self._box
        if self._drag_mode == "new" and self._drag_start is not None and self._drag_current is not None:
            x1, x2 = sorted((self._drag_start.x(), self._drag_current.x()))
            y1, y2 = sorted((self._drag_start.y(), self._drag_current.y()))
            box = [x1, y1, x2 - x1, y2 - y1]
        if box:
            sx, sy = rect.width() / self._image.width(), rect.height() / self._image.height()
            draw_rect = QRectF(rect.left() + box[0] * sx, rect.top() + box[1] * sy, box[2] * sx, box[3] * sy)
            painter.setPen(QPen(QColor("#22d3ee"), 2))
            painter.drawRect(draw_rect)
            painter.setBrush(QColor("#0f172a"))
            painter.setPen(QPen(QColor("#67e8f9"), 1))
            handle_size = 7.0
            handle_points = (
                draw_rect.topLeft(),
                QPoint(int(draw_rect.center().x()), int(draw_rect.top())),
                draw_rect.topRight(),
                QPoint(int(draw_rect.left()), int(draw_rect.center().y())),
                QPoint(int(draw_rect.right()), int(draw_rect.center().y())),
                draw_rect.bottomLeft(),
                QPoint(int(draw_rect.center().x()), int(draw_rect.bottom())),
                draw_rect.bottomRight(),
            )
            for handle_point in handle_points:
                painter.drawRect(
                    QRectF(
                        handle_point.x() - handle_size / 2,
                        handle_point.y() - handle_size / 2,
                        handle_size,
                        handle_size,
                    )
                )
            label = ", ".join(map(str, box))
            metrics = painter.fontMetrics()
            label_rect = metrics.boundingRect(label).adjusted(-5, -3, 5, 3)
            label_rect.moveBottomLeft(draw_rect.topLeft().toPoint())
            if label_rect.top() < rect.top():
                label_rect.moveTopLeft(draw_rect.topLeft().toPoint())
            painter.fillRect(label_rect, QColor(15, 23, 42, 220))
            painter.setPen(QColor("#67e8f9"))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

    def mousePressEvent(self, event: Any) -> None:
        if self._image.isNull():
            return
        widget_point = event.position().toPoint()
        image_point = self._to_image(widget_point)
        self.setFocus()
        control = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if control and event.button() == Qt.MouseButton.LeftButton:
            handle = self._resize_handle_at(image_point) if image_point is not None else None
            if handle:
                self._drag_mode = handle
                self._drag_original_box = list(self._box) if self._box else None
                self._drag_start = self._drag_current = image_point
                self.update()
            elif image_point is None or not self._point_in_box(image_point):
                self._drag_mode = "pan"
                self._pan_drag_start = widget_point
                self._pan_start_offset = QPointF(self._pan_offset)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
        elif control and event.button() == Qt.MouseButton.RightButton:
            if image_point is not None and self._point_in_box(image_point):
                self._drag_mode = "move"
                self._drag_original_box = list(self._box) if self._box else None
                self._drag_start = self._drag_current = image_point
                self.update()
        elif image_point is None:
            return
        elif event.button() == Qt.MouseButton.RightButton:
            self._box = None
            self._drag_start = self._drag_current = None
            self._drag_mode = None
            self._drag_original_box = None
            self.boxChanged.emit(None)
            self.update()
        elif event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = "new"
            self._drag_original_box = None
            self._drag_start = self._drag_current = image_point
            self.update()

    def mouseMoveEvent(self, event: Any) -> None:
        widget_point = event.position().toPoint()
        if self._drag_mode == "pan" and self._pan_drag_start is not None:
            delta = widget_point - self._pan_drag_start
            self._pan_offset = self._pan_start_offset + QPointF(delta)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.update()
            image_point = self._to_image(widget_point)
            self.mouseImagePosition.emit((image_point.x(), image_point.y()) if image_point else None)
            return
        image_point = self._to_image(widget_point)
        self.mouseImagePosition.emit((image_point.x(), image_point.y()) if image_point else None)
        if image_point is None:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier and not self._image.isNull():
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.unsetCursor()
            return
        if self._drag_start is not None:
            if self._drag_mode == "new" and event.buttons() & Qt.MouseButton.LeftButton:
                self._drag_current = image_point
                self.update()
            elif self._drag_mode and self._drag_mode.startswith("resize_") and event.buttons() & Qt.MouseButton.LeftButton:
                self._drag_current = image_point
                self._resize_box_to(image_point)
                self.update()
            elif self._drag_mode == "move" and event.buttons() & Qt.MouseButton.RightButton:
                self._drag_current = image_point
                self._move_box_to(image_point)
                self.update()
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            handle = self._resize_handle_at(image_point)
            if handle in {"resize_left_top", "resize_right_bottom"}:
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif handle in {"resize_right_top", "resize_left_bottom"}:
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif handle in {"resize_left", "resize_right"}:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif handle in {"resize_top", "resize_bottom"}:
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif self._point_in_box(image_point):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    def leaveEvent(self, event: Any) -> None:
        self.mouseImagePosition.emit(None)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self._drag_mode == "pan":
            if event.button() == Qt.MouseButton.LeftButton:
                self._pan_drag_start = None
                self._pan_start_offset = QPointF()
                self._drag_mode = None
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                event.accept()
            return
        expected_button = Qt.MouseButton.RightButton if self._drag_mode == "move" else Qt.MouseButton.LeftButton
        if event.button() != expected_button or self._drag_start is None:
            return
        image_point = self._to_image(event.position().toPoint()) or self._drag_current
        if self._drag_mode == "new" and image_point is not None:
            x1, x2 = sorted((self._drag_start.x(), image_point.x()))
            y1, y2 = sorted((self._drag_start.y(), image_point.y()))
            if x2 > x1 and y2 > y1:
                self._box = [x1, y1, x2 - x1, y2 - y1]
                self.boxChanged.emit(self._box)
        elif self._drag_mode and (self._drag_mode.startswith("resize_") or self._drag_mode == "move"):
            if self._box and self._box != self._drag_original_box:
                self.boxChanged.emit(self._box)
        self._drag_start = self._drag_current = None
        self._drag_mode = None
        self._drag_original_box = None
        self.update()

    def wheelEvent(self, event: Any) -> None:
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier) or self._image.isNull():
            super().wheelEvent(event)
            return
        widget_point = event.position()
        if not self._layout_rect().contains(widget_point):
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.accept()
            return
        self.setFocus()
        self._zoom_at(widget_point, 1.2 ** (delta / 120.0))
        event.accept()

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key.Key_Space and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.reset_view()
            event.accept()
        elif event.matches(QKeySequence.StandardKey.Copy):
            self.copyRequested.emit()
            event.accept()
        elif event.matches(QKeySequence.StandardKey.Paste):
            self.pasteRequested.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


class ChoiceEditorDialog(QDialog):
    def __init__(self, values: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑候选值")
        self.resize(380, 360)
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        for value in values:
            self._add(value)
        layout.addWidget(QLabel("双击可修改；候选值会显示在属性下拉框中。"))
        layout.addWidget(self.list_widget)
        row = QHBoxLayout()
        add_btn, remove_btn = QPushButton("添加"), QPushButton("删除")
        add_btn.clicked.connect(lambda: self._add("新值"))
        remove_btn.clicked.connect(self._remove)
        row.addWidget(add_btn)
        row.addWidget(remove_btn)
        row.addStretch()
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add(self, value: str) -> None:
        item = QTableWidgetItem(value)
        # QListWidget accepts a QListWidgetItem, while this conversion keeps code concise.
        from PySide6.QtWidgets import QListWidgetItem

        list_item = QListWidgetItem(item.text())
        list_item.setFlags(list_item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.list_widget.addItem(list_item)
        self.list_widget.setCurrentItem(list_item)

    def _remove(self) -> None:
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))

    def values(self) -> list[str]:
        result: list[str] = []
        for i in range(self.list_widget.count()):
            value = self.list_widget.item(i).text().strip()
            if value and value not in result:
                result.append(value)
        return result


class SchemaDialog(QDialog):
    def __init__(self, schema: list[dict[str, Any]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("属性设置")
        self.resize(650, 430)
        self._choices: list[list[str]] = []
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("设置属性名、默认值和可选值。默认值允许为空。"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["属性名", "默认值", "候选值"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        add_btn, remove_btn = QPushButton("添加属性"), QPushButton("删除属性")
        add_btn.clicked.connect(lambda: self.add_row({"name": "", "default": "", "choices": []}))
        remove_btn.clicked.connect(self.remove_rows)
        row.addWidget(add_btn)
        row.addWidget(remove_btn)
        row.addStretch()
        layout.addLayout(row)
        for attr in schema:
            self.add_row(attr)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_row(self, attr: dict[str, Any]) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(attr.get("name", ""))))
        self.table.setItem(row, 1, QTableWidgetItem(str(attr.get("default", ""))))
        choices = list(map(str, attr.get("choices", [])))
        self._choices.insert(row, choices)
        button = QPushButton(self._choices_text(choices))
        button.clicked.connect(lambda _checked=False, b=button: self.edit_choices(b))
        self.table.setCellWidget(row, 2, button)

    @staticmethod
    def _choices_text(choices: list[str]) -> str:
        return f"编辑（{len(choices)}）"

    def edit_choices(self, button: QPushButton) -> None:
        row = self.table.indexAt(button.pos()).row()
        if row < 0:
            return
        dialog = ChoiceEditorDialog(self._choices[row], self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._choices[row] = dialog.values()
            button.setText(self._choices_text(self._choices[row]))

    def remove_rows(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            self._choices.pop(row)

    def schema(self) -> list[dict[str, Any]]:
        result = []
        for row in range(self.table.rowCount()):
            name_item, default_item = self.table.item(row, 0), self.table.item(row, 1)
            result.append(
                {
                    "name": name_item.text().strip() if name_item else "",
                    "default": default_item.text() if default_item else "",
                    "choices": self._choices[row],
                }
            )
        return result

    def validate_and_accept(self) -> None:
        schema = self.schema()
        names = [item["name"] for item in schema]
        if any(not name for name in names):
            QMessageBox.warning(self, "属性设置", "属性名不能为空。")
            return
        if len(names) != len(set(names)):
            QMessageBox.warning(self, "属性设置", "属性名不能重复。")
            return
        self.accept()


class FrameAttributesDialog(QDialog):
    def __init__(
        self,
        schema: list[dict[str, Any]],
        current: dict[str, str],
        parent: QWidget | None = None,
        attribute_clipboard: dict[str, str] | None = None,
        frame_index: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("当前帧属性")
        self.resize(450, max(220, min(650, 110 + 55 * len(schema))))
        self._attribute_clipboard = attribute_clipboard if attribute_clipboard is not None else {}
        self.frame_index = frame_index
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.editors: dict[str, QComboBox] = {}
        self._editor_order: list[QComboBox] = []
        for attr in schema:
            name = str(attr["name"])
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(list(map(str, attr.get("choices", []))))
            value = str(current.get(name, attr.get("default", "")))
            combo.setCurrentText(value)
            combo.installEventFilter(self)
            if combo.lineEdit() is not None:
                combo.lineEdit().installEventFilter(self)
            self.editors[name] = combo
            self._editor_order.append(combo)
            form.addRow(name + "：", combo)
        layout.addLayout(form)
        if not schema:
            layout.addWidget(QLabel("尚未定义属性。请先点击主窗口中的“属性设置”。"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("完成并到下一帧")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if self._editor_order:
            QTimer.singleShot(0, lambda: self._focus_editor(0))

    def _focus_editor(self, index: int) -> None:
        if not self._editor_order:
            return
        combo = self._editor_order[index % len(self._editor_order)]
        combo.setFocus(Qt.FocusReason.TabFocusReason)
        editor = combo.lineEdit()
        if editor is not None:
            editor.setFocus(Qt.FocusReason.TabFocusReason)
            editor.selectAll()

    def _cycle_choice(self, combo: QComboBox, step: int) -> None:
        choices = [combo.itemText(index) for index in range(combo.count())]
        if not choices:
            return
        current = combo.currentText()
        try:
            current_index = choices.index(current)
            target_index = (current_index + step) % len(choices)
        except ValueError:
            target_index = 0 if step > 0 else len(choices) - 1
        combo.setCurrentIndex(target_index)
        editor = combo.lineEdit()
        if editor is not None:
            QTimer.singleShot(0, editor.selectAll)

    def eventFilter(self, watched: Any, event: Any) -> bool:
        combo = next(
            (
                candidate for candidate in self._editor_order
                if watched is candidate or watched is candidate.lineEdit()
            ),
            None,
        )
        if combo is not None:
            if event.type() == QEvent.Type.FocusIn:
                editor = combo.lineEdit()
                if editor is not None:
                    QTimer.singleShot(0, editor.selectAll)
            elif event.type() == QEvent.Type.KeyPress:
                key = event.key()
                if self.isActiveWindow() and event.matches(QKeySequence.StandardKey.Copy):
                    name = next((item_name for item_name, editor in self.editors.items() if editor is combo), "")
                    if name:
                        value = combo.currentText()
                        self._attribute_clipboard.clear()
                        self._attribute_clipboard[name] = value
                        QApplication.clipboard().setText(value)
                    event.accept()
                    return True
                if key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
                    current_index = self._editor_order.index(combo)
                    backwards = key == Qt.Key.Key_Backtab or bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                    self._focus_editor(current_index - 1 if backwards else current_index + 1)
                    event.accept()
                    return True
                if key in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
                    current_index = self._editor_order.index(combo)
                    self._focus_editor(current_index - 1 if key == Qt.Key.Key_PageUp else current_index + 1)
                    event.accept()
                    return True
                if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                    self._cycle_choice(combo, -1 if key == Qt.Key.Key_Up else 1)
                    event.accept()
                    return True
        return super().eventFilter(watched, event)

    def values(self) -> dict[str, str]:
        return {name: editor.currentText() for name, editor in self.editors.items()}

    def current_attribute_name(self) -> str | None:
        for name, combo in self.editors.items():
            editor = combo.lineEdit()
            if combo.hasFocus() or (editor is not None and editor.hasFocus()):
                return name
        return next(iter(self.editors), None)

    def paste_current_attribute(self) -> bool:
        name = self.current_attribute_name()
        if name is None or name not in self._attribute_clipboard:
            return False
        combo = self.editors[name]
        combo.setCurrentText(self._attribute_clipboard[name])
        editor = combo.lineEdit()
        if editor is not None:
            QTimer.singleShot(0, editor.selectAll)
        return True

    def apply_attribute_values(self, values: dict[str, str]) -> None:
        for name, value in values.items():
            combo = self.editors.get(name)
            if combo is not None:
                combo.setCurrentText(value)


class AttributeRangePasteDialog(QDialog):
    def __init__(self, current_frame: int, total_frames: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("区间粘贴属性")
        self.setModal(True)
        self.resize(420, 180)
        self.total_frames = total_frames
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("填写一个帧序号时仅粘贴该帧；填写两个时粘贴两者之间的闭区间。"))
        form = QFormLayout()
        self.first_edit = QLineEdit(str(current_frame))
        self.second_edit = QLineEdit()
        validator = QIntValidator(1, max(1, total_frames), self)
        self.first_edit.setValidator(validator)
        self.second_edit.setValidator(validator)
        self.first_edit.setPlaceholderText(f"1～{total_frames}")
        self.second_edit.setPlaceholderText(f"1～{total_frames}（可留空）")
        form.addRow("帧序号一", self.first_edit)
        form.addRow("帧序号二", self.second_edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.setText("确认粘贴")
        ok_button.setDefault(True)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setTabOrder(self.second_edit, self.first_edit)
        self.setTabOrder(self.first_edit, ok_button)
        QTimer.singleShot(0, self._focus_second)

    def _focus_second(self) -> None:
        self.second_edit.setFocus(Qt.FocusReason.OtherFocusReason)
        self.second_edit.selectAll()

    def _parsed_value(self, editor: QLineEdit) -> int | None:
        text = editor.text().strip()
        return int(text) if text else None

    def validate_and_accept(self) -> None:
        first, second = self._parsed_value(self.first_edit), self._parsed_value(self.second_edit)
        invalid = [value for value in (first, second) if value is not None and not 1 <= value <= self.total_frames]
        if invalid:
            QMessageBox.warning(self, "帧序号无效", f"帧序号必须在 1～{self.total_frames} 之间。")
            return
        self.accept()

    def frame_range(self) -> tuple[int, int] | None:
        first, second = self._parsed_value(self.first_edit), self._parsed_value(self.second_edit)
        if first is None and second is None:
            return None
        if first is None:
            first = second
        if second is None:
            second = first
        assert first is not None and second is not None
        return min(first, second) - 1, max(first, second) - 1


class TrackingSettingsDialog(QDialog):
    def __init__(self, settings: QSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("跟踪设置")
        self.resize(720, 430)
        layout = QFormLayout(self)
        local_title = QLabel("本地运行设置")
        local_title.setFont(QFont("Microsoft YaHei UI", 10, QFont.Weight.Bold))
        layout.addRow(local_title)
        self.code_path = QLineEdit(str(settings.value("tracking/mcitrack_code", DEFAULT_TRACKING_CODE)))
        self.model_path = QLineEdit(str(settings.value("tracking/mcitrack_model", DEFAULT_TRACKING_MODEL)))
        self.python_path = QLineEdit(str(settings.value("tracking/python", DEFAULT_TRACKING_PYTHON)))
        layout.addRow("MCITrack 官方代码目录", self._path_row(self.code_path, True))
        layout.addRow("MCITrack-L384 权重", self._path_row(self.model_path, False, "MCITrack权重 (*.pth.tar *.pth);;所有文件 (*.*)"))
        layout.addRow("算法环境 Python", self._path_row(self.python_path, False, "Python (*.exe);;所有文件 (*.*)"))
        note = QLabel(
            "算法始终由上面选择的 Conda Python 独立运行，不会与 MyLabel 的 Python/DLL 混用。"
            "程序会优先使用 CUDA，失败时自动回退 CPU。"
        )
        note.setWordWrap(True)
        layout.addRow(note)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addRow(separator)
        remote_title = QLabel("服务端运行设置")
        remote_title.setFont(QFont("Microsoft YaHei UI", 10, QFont.Weight.Bold))
        layout.addRow(remote_title)
        self.use_server = QCheckBox("使用服务端自动标注（关闭时使用上面的本地设置）")
        self.use_server.setChecked(str(settings.value("tracking/use_server", "false")).casefold() in {"1", "true", "yes"})
        self.server_url = QLineEdit(str(settings.value("tracking/server_url", "http://127.0.0.1:8000")))
        self.server_url.setPlaceholderText("例如：http://192.168.1.100:8000")
        self.server_token = QLineEdit(str(settings.value("tracking/server_token", "")))
        self.server_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.server_token.setPlaceholderText("model.json未配置令牌时可留空")
        self.server_timeout = QSpinBox()
        self.server_timeout.setRange(10, 3600)
        self.server_timeout.setSuffix(" 秒")
        self.server_timeout.setValue(int(settings.value("tracking/server_timeout", 120)))
        layout.addRow(self.use_server)
        layout.addRow("服务器地址", self.server_url)
        layout.addRow("访问令牌", self.server_token)
        layout.addRow("单次请求超时", self.server_timeout)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _path_row(self, editor: QLineEdit, folder: bool, file_filter: str = "所有文件 (*.*)") -> QWidget:
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("浏览…")
        def browse() -> None:
            value = QFileDialog.getExistingDirectory(self, "选择目录", editor.text()) if folder else \
                QFileDialog.getOpenFileName(self, "选择文件", editor.text(), file_filter)[0]
            if value:
                editor.setText(value)
        button.clicked.connect(browse)
        line.addWidget(editor, 1)
        line.addWidget(button)
        return row


class AttributeExportDialog(QDialog):
    def __init__(self, schema: list[dict[str, Any]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择要导出的TXT文件")
        self.resize(420, max(250, min(650, 190 + 38 * len(schema))))
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("groundtruth.txt 与每个勾选的属性将分别导出为TXT文件。"))
        self.groundtruth_checkbox = QCheckBox("groundtruth.txt（目标框 x,y,w,h）")
        self.groundtruth_checkbox.setChecked(True)
        layout.addWidget(self.groundtruth_checkbox)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        self.checkboxes: list[QCheckBox] = []
        for spec in schema:
            checkbox = QCheckBox(str(spec.get("name", "")))
            checkbox.setChecked(True)
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)
        layout.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("一键导出")
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_names(self) -> list[str]:
        return [item.text() for item in self.checkboxes if item.isChecked()]

    def groundtruth_selected(self) -> bool:
        return self.groundtruth_checkbox.isChecked()

    def validate_and_accept(self) -> None:
        if not self.groundtruth_selected() and not self.selected_names():
            QMessageBox.warning(self, "导出TXT", "请至少勾选一个要导出的文件。")
            return
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(str(bundled_path("icon_app.png"))))
        self.resize(1280, 820)
        self.source = FrameSource()
        self.store = ProjectStore()
        self.current_index = 0
        self.copied_box: list[int] | None = None
        self.copied_attributes: dict[str, str] = {}
        self.undo_stack: list[UndoEntry] = []
        self._paste_hold_timer = QTimer(self)
        self._paste_hold_timer.setSingleShot(True)
        self._paste_hold_timer.setInterval(2000)
        self._paste_hold_timer.timeout.connect(self._trigger_long_attribute_paste)
        self._paste_hold_active = False
        self._paste_hold_long_triggered = False
        self._paste_hold_context: tuple[str, FrameAttributesDialog | None] | None = None
        self._undo_disabled_dialog: QMessageBox | None = None
        self._loading_frame = False
        self.settings = QSettings("MyLabel", "MyLabel")
        self._migrate_tracking_settings()
        self.tracking_process: QProcess | None = None
        self.remote_tracking_thread: RemoteTrackingThread | None = None
        self.tracking_request_path: Path | None = None
        self.tracking_temp_dir: Path | None = None
        self.tracking_worker_dir: Path | None = None
        self._tracking_stdout = ""
        self._tracking_stop_requested = False
        self._tracking_error_seen = False
        self._build_ui()
        self._build_actions()
        self._set_enabled(False)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter()
        self.canvas = ImageCanvas()
        self.canvas.boxChanged.connect(self.on_box_changed)
        self.canvas.mouseImagePosition.connect(self.on_mouse_position)
        self.canvas.copyRequested.connect(self.copy_context_value)
        self.canvas.pasteRequested.connect(self.paste_context_value)
        splitter.addWidget(self.canvas)

        side = QFrame()
        side.setFrameShape(QFrame.Shape.StyledPanel)
        side.setMinimumWidth(300)
        side.setMaximumWidth(430)
        side_layout = QVBoxLayout(side)
        title = QLabel("标注信息")
        title.setFont(QFont("Microsoft YaHei UI", 13, QFont.Weight.Bold))
        side_layout.addWidget(title)
        info_grid = QGridLayout()
        info_grid.addWidget(QLabel("当前帧"), 0, 0)
        self.frame_label = QLabel("—")
        info_grid.addWidget(self.frame_label, 0, 1)
        info_grid.addWidget(QLabel("目标框"), 1, 0)
        self.box_label = QLabel("—")
        self.box_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info_grid.addWidget(self.box_label, 1, 1)
        info_grid.addWidget(QLabel("鼠标坐标"), 2, 0)
        self.mouse_label = QLabel("—")
        info_grid.addWidget(self.mouse_label, 2, 1)
        side_layout.addLayout(info_grid)
        side_layout.addSpacing(8)
        attr_head = QHBoxLayout()
        attr_title = QLabel("当前帧属性")
        attr_title.setFont(QFont("Microsoft YaHei UI", 11, QFont.Weight.Bold))
        self.schema_btn = QPushButton("属性设置")
        self.schema_btn.clicked.connect(self.edit_schema)
        attr_head.addWidget(attr_title)
        attr_head.addStretch()
        attr_head.addWidget(self.schema_btn)
        side_layout.addLayout(attr_head)
        self.attr_table = AttributeTableWidget()
        self.attr_table.setColumnCount(2)
        self.attr_table.setHorizontalHeaderLabels(["属性名", "属性值"])
        self.attr_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.attr_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.attr_table.verticalHeader().setVisible(False)
        side_layout.addWidget(self.attr_table, 1)
        self.edit_frame_attrs_btn = QPushButton("编辑当前帧属性（Enter）")
        self.edit_frame_attrs_btn.clicked.connect(self.edit_frame_attributes)
        side_layout.addWidget(self.edit_frame_attrs_btn)
        side_layout.addSpacing(10)
        tracking_title = QLabel("自动跟踪")
        tracking_title.setFont(QFont("Microsoft YaHei UI", 11, QFont.Weight.Bold))
        side_layout.addWidget(tracking_title)
        tracking_row = QHBoxLayout()
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem("MCITrack-384（L384）", "mcitrack_l384")
        self.tracking_settings_btn = QPushButton("设置")
        self.tracking_settings_btn.clicked.connect(self.edit_tracking_settings)
        tracking_row.addWidget(self.algorithm_combo, 1)
        tracking_row.addWidget(self.tracking_settings_btn)
        side_layout.addLayout(tracking_row)
        tracking_buttons = QHBoxLayout()
        self.start_tracking_btn = QPushButton("开始自动标注")
        self.continue_tracking_btn = QPushButton("继续跟踪")
        self.stop_tracking_btn = QPushButton("停止")
        self.start_tracking_btn.clicked.connect(self.start_tracking)
        self.continue_tracking_btn.clicked.connect(self.continue_tracking)
        self.stop_tracking_btn.clicked.connect(self.stop_tracking)
        self.stop_tracking_btn.setEnabled(False)
        tracking_buttons.addWidget(self.start_tracking_btn, 1)
        tracking_buttons.addWidget(self.continue_tracking_btn)
        tracking_buttons.addWidget(self.stop_tracking_btn)
        side_layout.addLayout(tracking_buttons)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        root.addWidget(splitter, 1)

        nav = QHBoxLayout()
        self.prev_btn, self.next_btn = QPushButton("◀"), QPushButton("▶")
        self.prev_btn.setToolTip("上一帧（←）")
        self.next_btn.setToolTip("下一帧（→）")
        self.prev_btn.clicked.connect(lambda: self.move_frame(-1))
        self.next_btn.clicked.connect(lambda: self.move_frame(1))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.valueChanged.connect(self.slider_changed)
        self.jump_spin = QSpinBox()
        self.jump_spin.setPrefix("跳到 ")
        self.jump_spin.valueChanged.connect(self.jump_changed)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.slider, 1)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.jump_spin)
        root.addLayout(nav)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("文件(&F)")
        open_video = QAction("打开视频…", self)
        open_video.setShortcut(QKeySequence("Ctrl+O"))
        open_video.triggered.connect(self.open_video)
        open_images = QAction("打开图片文件夹…", self)
        open_images.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_images.triggered.connect(self.open_image_folder)
        open_project = QAction("导入项目…", self)
        open_project.setShortcut(QKeySequence("Ctrl+I"))
        open_project.triggered.connect(self.open_project)
        save = QAction("保存", self)
        save.setShortcut(QKeySequence.StandardKey.Save)
        save.triggered.connect(self.save_project)
        file_menu.addActions([open_video, open_images, open_project])
        file_menu.addSeparator()
        file_menu.addAction(save)
        export_menu = file_menu.addMenu("导出")
        self.export_video_images_action = QAction("视频保存为图片文件夹…", self)
        self.export_video_images_action.triggered.connect(self.export_video_images)
        self.export_groundtruth_action = QAction("导出 groundtruth.txt…", self)
        self.export_groundtruth_action.triggered.connect(self.export_groundtruth)
        self.export_attributes_action = QAction("导出属性 TXT…", self)
        self.export_attributes_action.triggered.connect(self.export_attributes)
        export_menu.addActions([
            self.export_video_images_action,
            self.export_groundtruth_action,
            self.export_attributes_action,
        ])
        self.export_video_images_action.setEnabled(False)
        self.export_groundtruth_action.setEnabled(False)
        self.export_attributes_action.setEnabled(False)
        edit_menu = self.menuBar().addMenu("标注(&E)")
        copy_action = QAction("复制目标框/属性值", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        copy_action.triggered.connect(self.copy_context_value)
        paste_action = QAction("粘贴目标框/属性值", self)
        paste_action.triggered.connect(self.paste_context_value)
        attr_action = QAction("编辑当前帧属性", self)
        attr_action.setShortcut(QKeySequence(Qt.Key.Key_Return))
        attr_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        attr_action.triggered.connect(self.edit_frame_attributes)
        reset_view_action = QAction("恢复图像视图", self)
        reset_view_action.setShortcut(QKeySequence("Ctrl+Space"))
        reset_view_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        reset_view_action.triggered.connect(self._reset_canvas_view_from_shortcut)
        undo_action = QAction("撤销（Ctrl+Z）", self)
        undo_action.triggered.connect(self.request_undo)
        edit_menu.addActions([copy_action, paste_action, undo_action, attr_action, reset_view_action])

        prev_action = QAction(self)
        prev_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        prev_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        prev_action.triggered.connect(lambda: self.move_frame(-1))
        next_action = QAction(self)
        next_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        next_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        next_action.triggered.connect(lambda: self.move_frame(1))
        self.addActions([prev_action, next_action])

    def _reset_canvas_view_from_shortcut(self) -> None:
        if self.canvas.underMouse() or self.canvas.hasFocus():
            self.canvas.reset_view()

    def eventFilter(self, watched: Any, event: Any) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            control = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if control and event.key() == Qt.Key.Key_V:
                if self._paste_hold_active and self._paste_hold_long_triggered:
                    event.accept()
                    return True
                active_window = QApplication.activeWindow()
                attribute_context = self._attribute_paste_context()
                if attribute_context is not None:
                    if not event.isAutoRepeat() and not self._paste_hold_active:
                        if self.copied_attributes:
                            self._paste_hold_active = True
                            self._paste_hold_long_triggered = False
                            self._paste_hold_context = attribute_context
                            self._paste_hold_timer.start()
                        else:
                            self.statusBar().showMessage("尚未复制属性值", 2000)
                    event.accept()
                    return True
                if active_window is self:
                    if not event.isAutoRepeat() and self._mouse_over_widget(self.canvas):
                        self.paste_box()
                    event.accept()
                    return True
            if control and event.key() == Qt.Key.Key_Z:
                active_window = QApplication.activeWindow()
                warning_active = self._undo_disabled_dialog is not None and active_window is self._undo_disabled_dialog
                if active_window is self or warning_active:
                    if not event.isAutoRepeat():
                        self.request_undo()
                    event.accept()
                    return True
        elif event.type() == QEvent.Type.KeyRelease and self._paste_hold_active:
            if not event.isAutoRepeat() and event.key() in (Qt.Key.Key_V, Qt.Key.Key_Control):
                self._finish_attribute_paste_hold()
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _attribute_paste_context(self) -> tuple[str, FrameAttributesDialog | None] | None:
        active_window = QApplication.activeWindow()
        if active_window is self and self._mouse_over_widget(self.attr_table):
            return "main", None
        if isinstance(active_window, FrameAttributesDialog) and active_window.parent() is self:
            return "dialog", active_window
        return None

    def _attribute_paste_context_valid(self, context: tuple[str, FrameAttributesDialog | None] | None) -> bool:
        if context is None:
            return False
        kind, dialog = context
        if kind == "main":
            return QApplication.activeWindow() is self and self._mouse_over_widget(self.attr_table)
        return dialog is not None and dialog.isVisible() and QApplication.activeWindow() is dialog

    def _finish_attribute_paste_hold(self) -> None:
        context = self._paste_hold_context
        long_triggered = self._paste_hold_long_triggered
        self._paste_hold_timer.stop()
        self._paste_hold_active = False
        self._paste_hold_long_triggered = False
        self._paste_hold_context = None
        if long_triggered or not self._attribute_paste_context_valid(context):
            return
        kind, dialog = context
        if kind == "main":
            self.paste_attribute_values()
        elif dialog is not None:
            dialog.paste_current_attribute()

    def _trigger_long_attribute_paste(self) -> None:
        context = self._paste_hold_context
        if not self._paste_hold_active or not self._attribute_paste_context_valid(context):
            self._paste_hold_active = False
            self._paste_hold_context = None
            return
        self._paste_hold_long_triggered = True
        assert context is not None
        self._show_attribute_range_paste(context[1])

    def _tracking_is_running(self) -> bool:
        return self.tracking_process is not None or self.remote_tracking_thread is not None

    def request_undo(self) -> None:
        if self._tracking_is_running():
            self._show_undo_disabled_warning()
            return
        self.undo_last_action()

    def _show_undo_disabled_warning(self) -> None:
        if self._undo_disabled_dialog is not None and self._undo_disabled_dialog.isVisible():
            self._undo_disabled_dialog.raise_()
            self._undo_disabled_dialog.activateWindow()
            return
        message = QMessageBox(QMessageBox.Icon.Warning, "自动跟踪进行中", "自动跟踪运行期间无法撤销。\n跟踪不会停止，请等待跟踪完成或手动停止后再撤销。", QMessageBox.StandardButton.Ok, self)
        message.setWindowModality(Qt.WindowModality.NonModal)
        message.setModal(False)
        self._undo_disabled_dialog = message
        message.finished.connect(lambda _result, current=message: self._clear_undo_warning_reference(current))
        message.open()

    def _clear_undo_warning_reference(self, dialog: QMessageBox) -> None:
        if self._undo_disabled_dialog is dialog:
            self._undo_disabled_dialog = None

    def _close_undo_disabled_warning(self) -> None:
        dialog = self._undo_disabled_dialog
        self._undo_disabled_dialog = None
        if dialog is not None:
            dialog.close()

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (
            self.prev_btn,
            self.next_btn,
            self.slider,
            self.jump_spin,
            self.schema_btn,
            self.edit_frame_attrs_btn,
            self.algorithm_combo,
            self.start_tracking_btn,
            self.continue_tracking_btn,
        ):
            widget.setEnabled(enabled)
        self.export_groundtruth_action.setEnabled(enabled)
        self.export_attributes_action.setEnabled(enabled)
        self.export_video_images_action.setEnabled(
            enabled and self.source.info is not None and self.source.info.source_type == "video"
        )

    def _show_error(self, title: str, exc: Exception | str) -> None:
        QMessageBox.critical(self, title, str(exc))

    def open_video(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "打开视频", "", VIDEO_FILTER)
        if not filename:
            return
        try:
            info = self.source.open_video(Path(filename))
            project_path = ProjectStore.suggested_path(info)
            if project_path.exists():
                answer = QMessageBox.question(
                    self,
                    "已有项目",
                    "检测到同名项目文件，是否继续已有标注？\n选择“否”将取消打开，不会覆盖原项目。",
                )
                if answer == QMessageBox.StandardButton.Yes:
                    self._load_project_data(project_path, source_already_open=True)
                return
            self.store.create(info, self._load_recent_attribute_schema())
            self._activate_project()
        except Exception as exc:
            self._show_error("打开视频失败", exc)

    def open_image_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "打开图片文件夹")
        if not folder:
            return
        try:
            info = self.source.open_images(Path(folder))
            project_path = ProjectStore.suggested_path(info)
            if project_path.exists():
                answer = QMessageBox.question(
                    self,
                    "已有项目",
                    "检测到项目文件，是否继续已有标注？\n选择“否”将取消打开，不会覆盖原项目。",
                )
                if answer == QMessageBox.StandardButton.Yes:
                    self._load_project_data(project_path, source_already_open=True)
                return
            self.store.create(info, self._load_recent_attribute_schema())
            self._activate_project()
        except Exception as exc:
            self._show_error("打开图片文件夹失败", exc)

    def open_project(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "导入项目", "", "MyLabel 项目 (*.json);;所有文件 (*.*)")
        if filename:
            self._load_project_data(Path(filename))

    def _load_project_data(self, path: Path, source_already_open: bool = False) -> None:
        try:
            data = self.store.load(path)
            # Existing projects keep their own schema. Only an explicitly empty
            # schema inherits the most recently used video/project schema.
            if not data.get("attribute_schema"):
                recent_schema = self._load_recent_attribute_schema()
                if recent_schema:
                    data["attribute_schema"] = recent_schema
                    self.store.dirty = True
            meta = data["project"]
            source_path = Path(meta["source_path"])
            if not source_path.exists():
                replacement = self._locate_missing_source(str(meta["source_type"]))
                if replacement is None:
                    raise ValueError(f"找不到原始数据：{source_path}")
                source_path = replacement
                meta["source_path"] = str(source_path.resolve())
                self.store.dirty = True
            if not source_already_open:
                if meta["source_type"] == "video":
                    info = self.source.open_video(source_path)
                elif meta["source_type"] == "images":
                    info = self.source.open_images(source_path)
                else:
                    raise ValueError("未知的项目源类型。")
            else:
                info = self.source.info
            assert info is not None
            project_total = int(meta["total_frames"])
            if info.total_frames != project_total:
                frames = self.store.data["frames"]
                if len(frames) > info.total_frames:
                    del frames[info.total_frames:]
                elif len(frames) < info.total_frames:
                    frames.extend(default_frame(i) for i in range(len(frames), info.total_frames))
                meta["total_frames"] = info.total_frames
                meta["width"] = info.width
                meta["height"] = info.height
                meta["fps"] = info.fps
                self.store.dirty = True
            self._activate_project()
            if self.store.dirty:
                self.store.save()
        except Exception as exc:
            self._show_error("导入项目失败", exc)

    def _locate_missing_source(self, source_type: str) -> Path | None:
        if source_type == "video":
            filename, _ = QFileDialog.getOpenFileName(self, "重新定位原视频", "", VIDEO_FILTER)
            return Path(filename) if filename else None
        folder = QFileDialog.getExistingDirectory(self, "重新定位图片文件夹")
        return Path(folder) if folder else None

    def _activate_project(self) -> None:
        assert self.source.info is not None
        self.undo_stack.clear()
        self._paste_hold_timer.stop()
        self._paste_hold_active = False
        self._paste_hold_long_triggered = False
        self._paste_hold_context = None
        schema = self.store.data.get("attribute_schema", [])
        if schema:
            self._save_recent_attribute_schema(schema)
        self.current_index = 0
        total = self.source.info.total_frames
        self._loading_frame = True
        self.slider.setRange(1, total)
        self.jump_spin.setRange(1, total)
        self.slider.setValue(1)
        self.jump_spin.setValue(1)
        self._loading_frame = False
        self._set_enabled(True)
        self.setWindowTitle(f"{APP_NAME} — {self.source.info.path.name}")
        self.canvas.reset_view()
        self.show_frame(0)
        self.statusBar().showMessage(f"项目：{self.store.path}")

    def current_frame_data(self) -> dict[str, Any] | None:
        frames = self.store.data.get("frames", [])
        if not frames or not (0 <= self.current_index < len(frames)):
            return None
        return frames[self.current_index]

    def show_frame(self, index: int) -> None:
        info = self.source.info
        if info is None:
            return
        index = min(max(index, 0), info.total_frames - 1)
        frame = self.source.read(index)
        if frame is None:
            self._show_error("读取失败", f"无法读取第 {index + 1} 帧。")
            return
        self.current_index = index
        data = self.current_frame_data()
        box = data.get("box") if data else None
        self.canvas.set_frame(bgr_to_qimage(frame), box)
        self._loading_frame = True
        self.slider.setValue(index + 1)
        self.jump_spin.setValue(index + 1)
        self._loading_frame = False
        self.frame_label.setText(f"{index + 1} / {info.total_frames}  ({frame_id(index)})")
        self.update_box_label()
        self.update_attributes_table()
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < info.total_frames - 1)

    def slider_changed(self, value: int) -> None:
        if not self._loading_frame:
            self.navigate_to(value - 1)

    def jump_changed(self, value: int) -> None:
        if not self._loading_frame:
            self.navigate_to(value - 1)

    def move_frame(self, delta: int) -> None:
        if self.source.info is not None:
            self.navigate_to(self.current_index + delta)

    def navigate_to(self, index: int) -> None:
        info = self.source.info
        if info is None or not (0 <= index < info.total_frames) or index == self.current_index:
            return
        self.apply_defaults_if_empty()
        self.store.save()
        self.show_frame(index)

    def apply_defaults_if_empty(self) -> None:
        frame = self.current_frame_data()
        if frame is None:
            return
        attrs = frame.setdefault("attributes", {})
        changed = False
        for spec in self.store.data.get("attribute_schema", []):
            name = str(spec.get("name", ""))
            if name and name not in attrs:
                attrs[name] = str(spec.get("default", ""))
                changed = True
        if changed:
            self.store.dirty = True

    def _push_undo_entry(self, entry: UndoEntry) -> None:
        if entry.box_changes or entry.attribute_changes:
            self.undo_stack.append(entry)

    def _apply_attribute_values_to_frames(
        self,
        indices: list[int],
        values: dict[str, str],
        description: str,
        focus_index: int,
    ) -> tuple[int, dict[str, str]]:
        schema_names = {str(item.get("name", "")) for item in self.store.data.get("attribute_schema", [])}
        matched = {name: str(value) for name, value in values.items() if name and name in schema_names}
        if not matched:
            return 0, {}
        frames = self.store.data.get("frames", [])
        changes: dict[int, dict[str, tuple[bool, str]]] = {}
        for index in sorted(set(indices)):
            if not 0 <= index < len(frames):
                continue
            attrs = frames[index].setdefault("attributes", {})
            frame_changes: dict[str, tuple[bool, str]] = {}
            for name, value in matched.items():
                existed = name in attrs
                old_value = str(attrs.get(name, ""))
                if existed and old_value == value:
                    continue
                frame_changes[name] = (existed, old_value)
                attrs[name] = value
            if frame_changes:
                changes[index] = frame_changes
        if not changes:
            return 0, matched
        self._push_undo_entry(UndoEntry(description, focus_index, {}, changes))
        self.store.dirty = True
        return len(changes), matched

    def undo_last_action(self) -> None:
        if not self.undo_stack:
            self.statusBar().showMessage("没有可撤销的操作", 2000)
            return
        entry = self.undo_stack.pop()
        frames = self.store.data.get("frames", [])
        for index, old_box in entry.box_changes.items():
            if 0 <= index < len(frames):
                frames[index]["box"] = list(old_box) if old_box is not None else None
        for index, names in entry.attribute_changes.items():
            if not 0 <= index < len(frames):
                continue
            attrs = frames[index].setdefault("attributes", {})
            for name, (existed, old_value) in names.items():
                if existed:
                    attrs[name] = old_value
                else:
                    attrs.pop(name, None)
        self.store.dirty = True
        if self.source.info is not None and frames:
            focus_index = min(max(entry.focus_index, 0), len(frames) - 1)
            self.show_frame(focus_index)
        else:
            self.update_box_label()
            self.update_attributes_table()
        self.statusBar().showMessage(f"已撤销：{entry.description}", 2500)

    def on_box_changed(self, box: list[int] | None) -> None:
        frame = self.current_frame_data()
        if frame is None:
            return
        old_box = list(map(int, frame["box"])) if frame.get("box") else None
        new_box = list(map(int, box)) if box else None
        if old_box == new_box:
            return
        if old_box is None and new_box is not None:
            description = "画框"
        elif old_box is not None and new_box is None:
            description = "删除目标框"
        else:
            description = "调整目标框"
        self._push_undo_entry(UndoEntry(description, self.current_index, {self.current_index: old_box}, {}))
        frame["box"] = new_box
        self.store.dirty = True
        self.store.save()
        self.update_box_label()

    def update_box_label(self) -> None:
        info, frame = self.source.info, self.current_frame_data()
        if info is None or frame is None:
            self.box_label.setText("—")
            return
        box = frame.get("box")
        values = box if box else [info.width, info.height, 1, 1]
        self.box_label.setText(", ".join(map(str, values)))

    def on_mouse_position(self, pos: tuple[int, int] | None) -> None:
        self.mouse_label.setText(f"{pos[0]}, {pos[1]}" if pos else "—")

    def copy_box(self) -> None:
        frame = self.current_frame_data()
        box = frame.get("box") if frame else None
        if box:
            self.copied_box = list(map(int, box))
            self.statusBar().showMessage(f"已复制目标框：{self.copied_box}", 2500)

    def _mouse_over_widget(self, widget: QWidget) -> bool:
        if QApplication.activeWindow() is not self or not widget.isVisible():
            return False
        local_pos = widget.mapFromGlobal(QCursor.pos())
        return widget.rect().contains(local_pos)

    def _selected_attribute_rows(self) -> list[int]:
        selected = sorted({index.row() for index in self.attr_table.selectedIndexes()})
        if selected:
            return selected
        return list(range(self.attr_table.rowCount()))

    def copy_context_value(self) -> None:
        if QApplication.activeWindow() is not self:
            return
        if self._mouse_over_widget(self.attr_table):
            self.copy_attribute_values()
        elif self._mouse_over_widget(self.canvas):
            self.copy_box()

    def paste_context_value(self) -> None:
        if QApplication.activeWindow() is not self:
            return
        if self._mouse_over_widget(self.attr_table):
            self.paste_attribute_values()
        elif self._mouse_over_widget(self.canvas):
            self.paste_box()

    def copy_attribute_values(self) -> None:
        copied: dict[str, str] = {}
        for row in self._selected_attribute_rows():
            name_item = self.attr_table.item(row, 0)
            value_item = self.attr_table.item(row, 1)
            name = name_item.text() if name_item is not None else ""
            if name:
                copied[name] = value_item.text() if value_item is not None else ""
        if not copied:
            return
        self.copied_attributes.clear()
        self.copied_attributes.update(copied)
        QApplication.clipboard().setText("\n".join(f"{name}\t{value}" for name, value in copied.items()))
        self.statusBar().showMessage(f"已复制 {len(copied)} 个属性值", 2000)

    def paste_attribute_values(self) -> None:
        if not self.copied_attributes:
            return
        changed_count, matched = self._apply_attribute_values_to_frames(
            [self.current_index], self.copied_attributes, "粘贴属性值", self.current_index
        )
        if not matched:
            self.statusBar().showMessage("复制的属性与当前属性表没有匹配项", 2500)
            return
        if changed_count:
            self.store.save()
        self.update_attributes_table()
        self.statusBar().showMessage(f"已按属性名粘贴 {len(matched)} 个属性值", 2000)

    def _show_attribute_range_paste(self, source_dialog: FrameAttributesDialog | None) -> None:
        info = self.source.info
        if info is None or not self.copied_attributes:
            return
        parent: QWidget = source_dialog if source_dialog is not None else self
        dialog = AttributeRangePasteDialog(self.current_index + 1, info.total_frames, parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected_range = dialog.frame_range()
        if selected_range is None:
            self.statusBar().showMessage("未填写帧序号，没有粘贴属性", 2000)
            return
        start_index, end_index = selected_range
        indices = list(range(start_index, end_index + 1))
        changed_count, matched = self._apply_attribute_values_to_frames(
            indices, self.copied_attributes, "区间粘贴属性值", end_index
        )
        if not matched:
            self.statusBar().showMessage("复制的属性与当前属性表没有匹配项", 2500)
            return
        if changed_count:
            self.store.save()
        if source_dialog is not None and start_index <= source_dialog.frame_index <= end_index:
            source_dialog.apply_attribute_values(matched)
        self.show_frame(end_index)
        self.statusBar().showMessage(
            f"已按属性名粘贴到第 {start_index + 1}～{end_index + 1} 帧", 3000
        )

    def paste_box(self) -> None:
        frame = self.current_frame_data()
        info = self.source.info
        if frame is None or info is None or self.copied_box is None:
            return
        x, y, w, h = self.copied_box
        x, y = min(max(x, 0), info.width - 1), min(max(y, 0), info.height - 1)
        box = [x, y, min(w, info.width - x), min(h, info.height - y)]
        if box[2] <= 0 or box[3] <= 0:
            return
        old_box = list(map(int, frame["box"])) if frame.get("box") else None
        if old_box == box:
            return
        self._push_undo_entry(UndoEntry("粘贴目标框", self.current_index, {self.current_index: old_box}, {}))
        frame["box"] = box
        self.canvas.set_box(box)
        self.store.dirty = True
        self.store.save()
        self.update_box_label()
        self.statusBar().showMessage("已粘贴目标框", 2000)

    def edit_schema(self) -> None:
        if not self.store.data:
            return
        old_schema = self.store.data.get("attribute_schema", [])
        dialog = SchemaDialog(old_schema, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_schema = dialog.schema()
        new_names = {item["name"] for item in new_schema}
        for frame in self.store.data["frames"]:
            attrs = frame.setdefault("attributes", {})
            for obsolete in list(attrs):
                if obsolete not in new_names:
                    del attrs[obsolete]
        self.store.data["attribute_schema"] = new_schema
        self._save_recent_attribute_schema(new_schema)
        self.store.dirty = True
        self.store.save()
        self.update_attributes_table()

    def edit_frame_attributes(self) -> None:
        frame = self.current_frame_data()
        if frame is None:
            return
        frame_index = self.current_index
        dialog = FrameAttributesDialog(
            self.store.data.get("attribute_schema", []),
            frame.setdefault("attributes", {}),
            self,
            self.copied_attributes,
            frame_index,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            changed_count, _matched = self._apply_attribute_values_to_frames(
                [frame_index], dialog.values(), "修改属性", frame_index
            )
            if changed_count:
                self.store.save()
            if self.current_index != frame_index:
                self.show_frame(self.current_index)
            elif self.source.info and frame_index < self.source.info.total_frames - 1:
                self.show_frame(frame_index + 1)
            else:
                self.update_attributes_table()

    def update_attributes_table(self) -> None:
        frame = self.current_frame_data()
        schema = self.store.data.get("attribute_schema", [])
        attrs = frame.get("attributes", {}) if frame else {}
        self.attr_table.setRowCount(len(schema))
        for row, spec in enumerate(schema):
            name = str(spec.get("name", ""))
            value = str(attrs.get(name, spec.get("default", "")))
            self.attr_table.setItem(row, 0, QTableWidgetItem(name))
            self.attr_table.setItem(row, 1, QTableWidgetItem(value))

    def save_project(self) -> None:
        if self.store.path:
            try:
                self.store.save()
                self.statusBar().showMessage("项目已保存", 2000)
            except Exception as exc:
                self._show_error("保存失败", exc)

    def _require_all_frames_annotated(self, title: str) -> bool:
        frames = self.store.data.get("frames", [])
        missing = [index + 1 for index, frame in enumerate(frames) if not frame.get("box")]
        if not missing:
            return True
        preview = "、".join(map(str, missing[:10]))
        suffix = "……" if len(missing) > 10 else ""
        QMessageBox.warning(
            self,
            title,
            f"必须先标注所有帧才能导出。\n尚未标注 {len(missing)} 帧：{preview}{suffix}",
        )
        return False

    def _default_export_directory(self) -> Path:
        info = self.source.info
        if info is None:
            return Path.cwd()
        return info.path.parent if info.source_type == "video" else info.path

    def _confirm_export_overwrite(self, paths: list[Path], title: str) -> bool:
        existing = [path for path in paths if path.exists()]
        if not existing:
            return True
        preview = "\n".join(f"• {path.name}" for path in existing[:10])
        if len(existing) > 10:
            preview += f"\n……另有 {len(existing) - 10} 个文件"
        answer = QMessageBox.question(
            self,
            title,
            f"检测到 {len(existing)} 个同名文件已经存在：\n\n{preview}\n\n继续导出将覆盖这些文件，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def export_video_images(self) -> None:
        info = self.source.info
        if info is None or info.source_type != "video":
            QMessageBox.information(self, "导出图片", "只有视频项目可以保存为图片文件夹。")
            return
        suggested = self._default_export_directory() / f"{info.path.stem}_frames"
        folder = QFileDialog.getExistingDirectory(self, "选择图片保存文件夹", str(suggested))
        if not folder:
            return
        output = Path(folder)
        target_paths = [output / f"{frame_id(index)}.jpg" for index in range(info.total_frames)]
        if not self._confirm_export_overwrite(target_paths, "确认覆盖图片"):
            return
        progress = QProgressDialog("正在导出视频帧…", "取消", 0, info.total_frames, self)
        progress.setWindowTitle("导出图片")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        exported = 0
        try:
            for index in range(info.total_frames):
                if progress.wasCanceled():
                    break
                frame = self.source.read(index)
                if frame is None:
                    raise RuntimeError(f"无法读取第 {index + 1} 帧。")
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError(f"无法编码第 {index + 1} 帧。")
                encoded.tofile(str(output / f"{frame_id(index)}.jpg"))
                exported += 1
                progress.setValue(exported)
                QApplication.processEvents()
            progress.close()
            if exported == info.total_frames:
                QMessageBox.information(self, "导出完成", f"已导出 {exported} 张图片到：\n{output}")
            else:
                QMessageBox.information(self, "已取消", f"导出已取消，文件夹中保留了 {exported} 张已导出的图片。")
        except Exception as exc:
            progress.close()
            self._show_error("导出图片失败", exc)

    def export_groundtruth(self) -> None:
        if self.source.info is None or not self._require_all_frames_annotated("导出 groundtruth.txt"):
            return
        suggested = self._default_export_directory() / "groundtruth.txt"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "保存 groundtruth.txt",
            str(suggested),
            "文本文件 (*.txt)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not filename:
            return
        target = Path(filename)
        if not self._confirm_export_overwrite([target], "确认覆盖 groundtruth.txt"):
            return
        try:
            lines = [",".join(map(str, frame["box"])) for frame in self.store.data["frames"]]
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
            QMessageBox.information(self, "导出完成", f"groundtruth.txt 已保存：\n{filename}")
        except Exception as exc:
            self._show_error("导出 groundtruth.txt 失败", exc)

    @staticmethod
    def _safe_attribute_filename(name: str) -> str:
        invalid = '<>:"/\\|?*'
        cleaned = "".join("_" if char in invalid or ord(char) < 32 else char for char in name).rstrip(" .")
        return cleaned or "未命名属性"

    def export_attributes(self) -> None:
        if self.source.info is None or not self._require_all_frames_annotated("导出属性 TXT"):
            return
        schema = self.store.data.get("attribute_schema", [])
        dialog = AttributeExportDialog(schema, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        folder = QFileDialog.getExistingDirectory(self, "选择TXT文件保存文件夹", str(self._default_export_directory()))
        if not folder:
            return
        specs = {str(spec.get("name", "")): spec for spec in schema}
        used_names: set[str] = set()
        targets: list[tuple[str, Path]] = []
        if dialog.groundtruth_selected():
            targets.append(("groundtruth", Path(folder) / "groundtruth.txt"))
            used_names.add("groundtruth")
        for name in dialog.selected_names():
            stem = self._safe_attribute_filename(name)
            candidate = stem
            number = 2
            while candidate.casefold() in used_names:
                candidate = f"{stem}_{number}"
                number += 1
            used_names.add(candidate.casefold())
            targets.append((name, Path(folder) / f"{candidate}.txt"))
        if not self._confirm_export_overwrite([path for _name, path in targets], "确认覆盖TXT文件"):
            return
        try:
            exported = 0
            for name, target in targets:
                if name == "groundtruth" and dialog.groundtruth_selected() and target.name.casefold() == "groundtruth.txt":
                    lines = [",".join(map(str, frame["box"])) for frame in self.store.data["frames"]]
                else:
                    default = str(specs[name].get("default", ""))
                    lines = [str(frame.get("attributes", {}).get(name, default)) for frame in self.store.data["frames"]]
                target.write_text("\n".join(lines) + "\n", encoding="utf-8")
                exported += 1
            QMessageBox.information(self, "导出完成", f"已导出 {exported} 个TXT文件到：\n{folder}")
        except Exception as exc:
            self._show_error("导出TXT失败", exc)

    def edit_tracking_settings(self) -> None:
        dialog = TrackingSettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings.setValue("tracking/mcitrack_code", dialog.code_path.text().strip())
            self.settings.setValue("tracking/mcitrack_model", dialog.model_path.text().strip())
            self.settings.setValue("tracking/python", dialog.python_path.text().strip())
            self.settings.setValue("tracking/use_server", dialog.use_server.isChecked())
            self.settings.setValue("tracking/server_url", dialog.server_url.text().strip())
            self.settings.setValue("tracking/server_token", dialog.server_token.text())
            self.settings.setValue("tracking/server_timeout", dialog.server_timeout.value())

    def _migrate_tracking_settings(self) -> None:
        """Tracking paths remain user-selected and are never tied to bundled models."""

    def _load_recent_attribute_schema(self) -> list[dict[str, Any]]:
        raw = str(self.settings.value("attributes/recent_schema", "")).strip()
        if not raw:
            return []
        try:
            schema = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(schema, list):
            return []
        result: list[dict[str, Any]] = []
        for item in schema:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            choices = item.get("choices", [])
            if not isinstance(choices, list):
                choices = []
            result.append({
                "name": name,
                "default": str(item.get("default", "")),
                "choices": [str(value) for value in choices],
            })
        return result

    def _save_recent_attribute_schema(self, schema: list[dict[str, Any]]) -> None:
        normalized: list[dict[str, Any]] = []
        for item in schema:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            choices = item.get("choices", [])
            normalized.append({
                "name": name,
                "default": str(item.get("default", "")),
                "choices": [str(value) for value in choices] if isinstance(choices, list) else [],
            })
        self.settings.setValue("attributes/recent_schema", json.dumps(normalized, ensure_ascii=False))
        self.settings.sync()

    def _worker_path(self) -> Path:
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        return base / "tracker_worker.py"

    def _tracking_source_request(self) -> dict[str, Any]:
        info = self.source.info
        if info is None:
            raise RuntimeError("尚未打开视频或图片文件夹。")
        if info.source_type == "images":
            assert info.image_files is not None
            return {
                "source_type": "images",
                "frame_paths": [str(path.resolve()) for path in info.image_files],
                "total_frames": info.total_frames,
            }
        return {
            "source_type": "video",
            "source_path": str(info.path.resolve()),
            "total_frames": info.total_frames,
        }

    def start_tracking(self) -> None:
        self._start_tracking_from(0)

    def continue_tracking(self) -> None:
        self._start_tracking_from(self.current_index)

    def _start_tracking_from(self, start_index: int) -> None:
        if self.tracking_process is not None or self.remote_tracking_thread is not None or self.source.info is None:
            return
        frames = self.store.data.get("frames", [])
        if start_index < 0 or start_index >= len(frames):
            return
        initial_frame = frames[start_index]
        if not initial_frame.get("box"):
            label = "第一帧" if start_index == 0 else f"当前第 {start_index + 1} 帧"
            QMessageBox.warning(self, "无法自动标注", f"{label}尚未标注目标框，请先标注目标。")
            return
        use_server = str(self.settings.value("tracking/use_server", "false")).casefold() in {"1", "true", "yes"}
        if use_server:
            self._start_remote_tracking(start_index, initial_frame)
            return
        code_path = str(self.settings.value("tracking/mcitrack_code", DEFAULT_TRACKING_CODE)).strip()
        model_path = str(self.settings.value("tracking/mcitrack_model", DEFAULT_TRACKING_MODEL)).strip()
        python_path = str(self.settings.value("tracking/python", DEFAULT_TRACKING_PYTHON)).strip()
        if not python_path or python_path.casefold() in {"python", "python.exe", "py", "py.exe"}:
            QMessageBox.warning(
                self,
                "跟踪设置",
                "请在“设置”中选择 MCITrack Conda 环境内具体的 python.exe。",
            )
            return
        if not Path(python_path).is_file():
            QMessageBox.warning(self, "跟踪设置", f"算法环境 Python 不存在：\n{python_path}")
            return
        if not Path(code_path).is_dir() or not Path(model_path).is_file():
            QMessageBox.warning(
                self,
                "跟踪设置",
                f"算法目录或模型文件不存在，请先在“设置”中修改路径。\n\n算法目录：{code_path}\n模型：{model_path}",
            )
            return
        try:
            request = {
                "algorithm": self.algorithm_combo.currentData(),
                "code_path": code_path,
                "model_path": model_path,
                "prefer_gpu": True,
                "start_index": start_index,
                "box": initial_frame["box"],
                **self._tracking_source_request(),
            }
            import shutil
            worker_dir = Path(tempfile.mkdtemp(prefix="mylabel_worker_"))
            self.tracking_worker_dir = worker_dir
            worker_path = worker_dir / "tracker_worker.py"
            shutil.copy2(self._worker_path(), worker_path)
            filename = str(worker_dir / "request.json")
            with open(filename, "w", encoding="utf-8") as handle:
                json.dump(request, handle, ensure_ascii=False)
            self.tracking_request_path = Path(filename)
            process = QProcess(self)
            process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
            process.setProcessEnvironment(self._external_python_environment(Path(python_path)))
            process.readyReadStandardOutput.connect(self._read_tracking_output)
            process.readyReadStandardError.connect(self._read_tracking_error)
            process.finished.connect(self._tracking_finished)
            self.tracking_process = process
            self._tracking_stdout = ""
            self._tracking_stop_requested = False
            self._tracking_error_seen = False
            restore_dll_directory = self._clear_pyinstaller_dll_directory()
            try:
                # -I ignores user/PYTHON* injection. The worker is copied away
                # from PyInstaller's _MEI directory so Python 3.11 cannot see
                # extension modules bundled for the application's Python 3.13.
                process.start(python_path, ["-I", str(worker_path), filename])
                if not process.waitForStarted(5000):
                    raise RuntimeError(f"无法启动算法 Python：{python_path}")
            finally:
                restore_dll_directory()
            self.start_tracking_btn.setEnabled(False)
            self.continue_tracking_btn.setEnabled(False)
            self.stop_tracking_btn.setEnabled(True)
            self.statusBar().showMessage("正在加载 MCITrack-L384 模型…")
        except Exception as exc:
            self._cleanup_tracking()
            self._show_error("启动自动跟踪失败", exc)

    def _start_remote_tracking(self, start_index: int, initial_frame: dict[str, Any]) -> None:
        server_url = str(self.settings.value("tracking/server_url", "")).strip().rstrip("/")
        if not server_url.startswith(("http://", "https://")):
            QMessageBox.warning(self, "服务端设置", "服务器地址必须以 http:// 或 https:// 开头。")
            return
        assert self.source.info is not None
        request = {
            "algorithm": self.algorithm_combo.currentData(),
            "server_url": server_url,
            "server_token": str(self.settings.value("tracking/server_token", "")),
            "server_timeout": int(self.settings.value("tracking/server_timeout", 120)),
            "jpeg_quality": 90,
            "start_index": start_index,
            "box": initial_frame["box"],
            **self._tracking_source_request(),
        }
        thread = RemoteTrackingThread(request, self)
        thread.messageReceived.connect(self._handle_tracking_message)
        thread.trackingCompleted.connect(self._remote_tracking_finished)
        thread.finished.connect(thread.deleteLater)
        self.remote_tracking_thread = thread
        self._tracking_stop_requested = False
        self._tracking_error_seen = False
        self.start_tracking_btn.setEnabled(False)
        self.continue_tracking_btn.setEnabled(False)
        self.stop_tracking_btn.setEnabled(True)
        self.statusBar().showMessage("正在连接自动标注服务器并加载模型…")
        thread.start()

    @staticmethod
    def _external_python_environment(python_path: Path) -> QProcessEnvironment:
        environment = QProcessEnvironment.systemEnvironment()
        for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__"):
            environment.remove(key)
        # PyInstaller uses these private variables for its child processes. An
        # independent Python interpreter must not inherit them.
        for key in list(environment.keys()):
            if key.startswith("_PYI_"):
                environment.remove(key)
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUNBUFFERED", "1")
        # Selecting a Conda interpreter directly does not run `conda activate`.
        # Add the environment's native DLL and executable directories explicitly.
        env_root = python_path.resolve().parent
        conda_entries = [
            env_root,
            env_root / "Library" / "mingw-w64" / "bin",
            env_root / "Library" / "usr" / "bin",
            env_root / "Library" / "bin",
            env_root / "Scripts",
            env_root / "bin",
        ]
        bundle_dir = str(getattr(sys, "_MEIPASS", ""))
        if bundle_dir:
            path_entries = environment.value("PATH").split(os.pathsep)
            bundle_norm = os.path.normcase(os.path.abspath(bundle_dir))
            filtered = []
            for entry in path_entries:
                if not entry:
                    continue
                try:
                    entry_norm = os.path.normcase(os.path.abspath(entry))
                except OSError:
                    entry_norm = entry
                if entry_norm != bundle_norm and not entry_norm.startswith(bundle_norm + os.sep):
                    filtered.append(entry)
        else:
            filtered = [entry for entry in environment.value("PATH").split(os.pathsep) if entry]
        existing = {os.path.normcase(os.path.abspath(entry)) for entry in filtered}
        prepend = [str(entry) for entry in conda_entries if entry.is_dir() and os.path.normcase(str(entry)) not in existing]
        environment.insert("PATH", os.pathsep.join(prepend + filtered))
        if (env_root / "conda-meta").is_dir():
            environment.insert("CONDA_PREFIX", str(env_root))
        return environment

    @staticmethod
    def _clear_pyinstaller_dll_directory() -> Any:
        if os.name != "nt" or not getattr(sys, "frozen", False):
            return lambda: None
        import ctypes
        bundle_dir = str(getattr(sys, "_MEIPASS", ""))
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.SetDllDirectoryW.argtypes = [ctypes.c_wchar_p]
        kernel32.SetDllDirectoryW.restype = ctypes.c_bool
        kernel32.SetDllDirectoryW(None)
        return lambda: kernel32.SetDllDirectoryW(bundle_dir or None)

    def _read_tracking_output(self) -> None:
        if self.tracking_process is None:
            return
        self._tracking_stdout += bytes(self.tracking_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        lines = self._tracking_stdout.split("\n")
        self._tracking_stdout = lines.pop()
        for line in lines:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._handle_tracking_message(message)

    def _handle_tracking_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "box":
            index = int(message["index"])
            box = [int(value) for value in message["box"]]
            if not 0 <= index < len(self.store.data.get("frames", [])):
                return
            frame = self.store.data["frames"][index]
            frame["box"] = box
            frame["attributes"] = {
                str(spec.get("name", "")): str(spec.get("default", ""))
                for spec in self.store.data.get("attribute_schema", []) if spec.get("name")
            }
            self.store.dirty = True
            self.store.save()
            self.show_frame(index)
            total = self.source.info.total_frames if self.source.info else 0
            self.statusBar().showMessage(f"自动跟踪：{index + 1} / {total}")
        elif kind in {"ready", "fallback", "warning"}:
            self.statusBar().showMessage(message.get("message") or f"正在使用 {message.get('device')} 跟踪…")
        elif kind == "environment":
            requested = Path(str(self.settings.value("tracking/python", DEFAULT_TRACKING_PYTHON))).resolve()
            actual = Path(str(message.get("python", ""))).resolve()
            if os.path.normcase(str(requested)) != os.path.normcase(str(actual)):
                self._tracking_error_seen = True
                self._show_error("算法环境错误", f"实际启动的 Python 与设置不一致。\n设置：{requested}\n实际：{actual}")
                self.stop_tracking()
        elif kind == "error":
            self._tracking_error_seen = True
            self._show_error("自动跟踪失败", message.get("message", "未知错误"))

    def _read_tracking_error(self) -> None:
        if self.tracking_process is not None:
            # Keep stderr drained, but do not show third-party warnings in the
            # status bar. Structured UTF-8 errors arrive through stdout JSON.
            self.tracking_process.readAllStandardError()

    def stop_tracking(self) -> None:
        if self.tracking_process is not None:
            self._tracking_stop_requested = True
            self.tracking_process.kill()
            self.statusBar().showMessage("正在停止自动跟踪…")
        elif self.remote_tracking_thread is not None:
            self._tracking_stop_requested = True
            self.remote_tracking_thread.requestInterruption()
            self.statusBar().showMessage("正在停止服务端自动跟踪…")

    def _tracking_finished(self, exit_code: int, _status: Any) -> None:
        stopped = self._tracking_stop_requested
        failed = self._tracking_error_seen or (exit_code != 0 and not stopped)
        self._cleanup_tracking()
        self.start_tracking_btn.setEnabled(self.source.info is not None)
        self.continue_tracking_btn.setEnabled(self.source.info is not None)
        self.stop_tracking_btn.setEnabled(False)
        self._close_undo_disabled_warning()
        if stopped:
            text = "自动跟踪已停止"
        elif failed:
            text = "自动跟踪失败"
        else:
            text = "自动跟踪完成，项目已保存"
        self.statusBar().showMessage(text, 4000)

    def _remote_tracking_finished(self, stopped: bool, error: str) -> None:
        thread = self.remote_tracking_thread
        self.remote_tracking_thread = None
        self.start_tracking_btn.setEnabled(self.source.info is not None)
        self.continue_tracking_btn.setEnabled(self.source.info is not None)
        self.stop_tracking_btn.setEnabled(False)
        self._close_undo_disabled_warning()
        if error:
            self._tracking_error_seen = True
            self._show_error("服务端自动跟踪失败", error)
            text = "服务端自动跟踪失败"
        elif stopped or self._tracking_stop_requested:
            text = "服务端自动跟踪已停止"
        else:
            text = "服务端自动跟踪完成，项目已保存"
        self._tracking_stop_requested = False
        self._tracking_error_seen = False
        self.statusBar().showMessage(text, 4000)

    def _cleanup_tracking(self) -> None:
        if self.tracking_request_path is not None:
            try:
                self.tracking_request_path.unlink(missing_ok=True)
            except OSError:
                pass
        if self.tracking_temp_dir is not None:
            import shutil
            try:
                shutil.rmtree(self.tracking_temp_dir)
            except OSError:
                pass
        if self.tracking_worker_dir is not None:
            import shutil
            try:
                shutil.rmtree(self.tracking_worker_dir)
            except OSError:
                pass
        self.tracking_worker_dir = None
        self.tracking_temp_dir = None
        self.tracking_request_path = None
        self.tracking_process = None
        self._tracking_stop_requested = False
        self._tracking_error_seen = False

    def closeEvent(self, event: Any) -> None:
        try:
            if self.tracking_process is not None:
                self.tracking_process.kill()
                self.tracking_process.waitForFinished(2000)
            if self.remote_tracking_thread is not None:
                self.remote_tracking_thread.requestInterruption()
                wait_ms = (int(self.settings.value("tracking/server_timeout", 120)) + 5) * 1000
                self.remote_tracking_thread.wait(wait_ms)
            if self.store.path:
                self.store.save()
            self.source.close()
            event.accept()
        except Exception as exc:
            answer = QMessageBox.question(self, "保存失败", f"关闭前保存失败：{exc}\n仍要退出吗？")
            event.accept() if answer == QMessageBox.StandardButton.Yes else event.ignore()


def main() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("MyLabel")
    app.setWindowIcon(QIcon(str(bundled_path("icon_app.png"))))
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).suffix.casefold() == ".json":
        window._load_project_data(Path(sys.argv[1]))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
