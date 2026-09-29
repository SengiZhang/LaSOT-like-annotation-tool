from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REQUIRED_FILES = ("groundtruth.txt", "full_occlusion.txt", "out_of_view.txt")
SEARCH_MODE_INDEPENDENT = "independent"
SEARCH_MODE_SYNC_GROUND_TRUTH = "sync_ground_truth"
SEARCH_MODE_SYNC_PREDICTION = "sync_prediction"


class DataError(ValueError):
    pass


@dataclass(slots=True)
class SequenceSource:
    name: str
    directory: Path
    _images: list[Path] | None = field(default=None, init=False, repr=False)

    @property
    def image_directory(self) -> Path:
        return self.directory / "img"

    def first_image(self) -> Path:
        """Return a preview image without enumerating the whole image directory."""
        for stem in ("00000001", "00000000", "1", "0001", "000001"):
            for extension in IMAGE_EXTENSIONS:
                candidate = self.image_directory / f"{stem}{extension}"
                if candidate.is_file():
                    return candidate
        try:
            with os.scandir(self.image_directory) as entries:
                for entry in entries:
                    if entry.is_file() and Path(entry.name).suffix.casefold() in IMAGE_EXTENSIONS:
                        return Path(entry.path)
        except OSError as exc:
            raise DataError(f"无法读取 img 文件夹：{exc}") from exc
        raise DataError("img 文件夹中没有支持的图像")

    def load_images(self) -> list[Path]:
        """Enumerate and cache frame paths only when a detail view is opened."""
        if self._images is None:
            try:
                self._images = sorted(
                    (
                        Path(entry.path)
                        for entry in os.scandir(self.image_directory)
                        if entry.is_file() and Path(entry.name).suffix.casefold() in IMAGE_EXTENSIONS
                    ),
                    key=natural_key,
                )
            except OSError as exc:
                raise DataError(f"无法读取 img 文件夹：{exc}") from exc
            if not self._images:
                raise DataError("img 文件夹中没有支持的图像")
        return self._images


@dataclass(slots=True)
class SequenceAnalysis:
    source: SequenceSource
    prediction_path: Path
    preview_image: Path
    ground_truth: np.ndarray
    full_occlusion: np.ndarray
    out_of_view: np.ndarray
    prediction: np.ndarray
    frame_count: int
    iou: np.ndarray
    center_distance: np.ndarray

    @property
    def name(self) -> str:
        return self.source.name

def natural_key(value: str | Path) -> list[object]:
    name = value.name if isinstance(value, Path) else value
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", name)]


def _tokens(text: str) -> list[str]:
    return [part for part in re.split(r"[,\s]+", text.strip()) if part]


def read_boxes(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="gb18030").splitlines()
    for line_number, line in enumerate(lines, 1):
        parts = _tokens(line)
        if not parts:
            continue
        if len(parts) != 4:
            raise DataError(f"{path.name} 第 {line_number} 行不是 4 个数值")
        try:
            row = [float(value) for value in parts]
            if not np.all(np.isfinite(row)):
                raise DataError(f"{path.name} 第 {line_number} 行包含非有限数值")
            rows.append(row)
        except ValueError as exc:
            raise DataError(f"{path.name} 第 {line_number} 行包含无效数值") from exc
    if not rows:
        raise DataError(f"{path.name} 没有有效目标框")
    return np.asarray(rows, dtype=np.float64)


def read_flags(path: Path) -> np.ndarray:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="gb18030")
    parts = _tokens(text)
    if not parts:
        raise DataError(f"{path.name} 没有有效标记")
    values: list[int] = []
    for part in parts:
        try:
            value = int(float(part))
        except ValueError as exc:
            raise DataError(f"{path.name} 包含无效标记：{part}") from exc
        if value not in (0, 1):
            raise DataError(f"{path.name} 只允许 0 和 1")
        values.append(value)
    return np.asarray(values, dtype=np.uint8)


def is_sequence_directory(path: Path) -> bool:
    return path.is_dir() and (path / "img").is_dir() and all((path / name).is_file() for name in REQUIRED_FILES)


def _sequence_directories(root: Path) -> list[Path]:
    """Walk only directory metadata and never descend into an image directory."""
    found: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as iterator:
                entries = list(iterator)
        except OSError:
            continue
        by_name = {entry.name.casefold(): entry for entry in entries}
        image_entry = by_name.get("img")
        if (
            image_entry is not None
            and image_entry.is_dir(follow_symlinks=False)
            and all(name.casefold() in by_name and by_name[name.casefold()].is_file() for name in REQUIRED_FILES)
        ):
            found.append(current)
            continue
        for entry in entries:
            if entry.name.casefold() == "img":
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
            except OSError:
                continue
    return found


def scan_dataset(path: Path) -> tuple[dict[str, SequenceSource], list[str]]:
    if not path.is_dir():
        raise DataError("数据集路径不是文件夹")
    directories = _sequence_directories(path)
    directories.sort(key=lambda p: natural_key(str(p.relative_to(path))))
    if not directories:
        raise DataError("未找到包含 img 和三个标注文件的序列目录")

    result: dict[str, SequenceSource] = {}
    warnings: list[str] = []
    for directory in directories:
        name = directory.name
        if name in result:
            warnings.append(f"序列名重复，已忽略：{directory}")
            continue
        result[name] = SequenceSource(name, directory)
    if not result:
        raise DataError("所有序列均解析失败")
    return result, warnings


def scan_predictions(path: Path) -> tuple[dict[str, Path], list[str]]:
    if path.is_file():
        files = [path] if path.suffix.casefold() == ".txt" else []
    elif path.is_dir():
        files = sorted((p for p in path.rglob("*.txt") if p.is_file()), key=lambda p: natural_key(p.name))
    else:
        raise DataError("预测结果路径无效")
    if not files:
        raise DataError("未找到预测结果 TXT 文件")
    result: dict[str, Path] = {}
    warnings: list[str] = []
    for file in files:
        if file.stem in result:
            warnings.append(f"预测文件名重复，已忽略：{file}")
        else:
            result[file.stem] = file
    return result, warnings


def calculate_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt = ground_truth.astype(np.float64, copy=False)
    pred = prediction.astype(np.float64, copy=False)
    gt_x2 = gt[:, 0] + np.maximum(gt[:, 2], 0)
    gt_y2 = gt[:, 1] + np.maximum(gt[:, 3], 0)
    pr_x2 = pred[:, 0] + np.maximum(pred[:, 2], 0)
    pr_y2 = pred[:, 1] + np.maximum(pred[:, 3], 0)
    iw = np.maximum(0, np.minimum(gt_x2, pr_x2) - np.maximum(gt[:, 0], pred[:, 0]))
    ih = np.maximum(0, np.minimum(gt_y2, pr_y2) - np.maximum(gt[:, 1], pred[:, 1]))
    intersection = iw * ih
    union = np.maximum(gt[:, 2], 0) * np.maximum(gt[:, 3], 0) + np.maximum(pred[:, 2], 0) * np.maximum(pred[:, 3], 0) - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    gt_center = gt[:, :2] + gt[:, 2:4] / 2
    pred_center = pred[:, :2] + pred[:, 2:4] / 2
    distance = np.linalg.norm(gt_center - pred_center, axis=1)
    return iou, distance


def search_reference_boxes(
    ground_truth: np.ndarray, prediction: np.ndarray, frame: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference boxes for the truth and prediction search canvases."""
    index = max(0, frame - 1)
    truth_reference = ground_truth[index]
    prediction_reference = prediction[index]
    if mode == SEARCH_MODE_INDEPENDENT:
        return truth_reference, prediction_reference
    if mode == SEARCH_MODE_SYNC_GROUND_TRUTH:
        return truth_reference, truth_reference
    if mode == SEARCH_MODE_SYNC_PREDICTION:
        return prediction_reference, prediction_reference
    raise DataError(f"未知搜索区域模式：{mode}")


def build_analyses(
    dataset: dict[str, SequenceSource], predictions: dict[str, Path]
) -> tuple[list[SequenceAnalysis], list[str], list[str], list[str]]:
    missing = sorted(set(dataset) - set(predictions), key=natural_key)
    extra = sorted(set(predictions) - set(dataset), key=natural_key)
    warnings: list[str] = []
    analyses: list[SequenceAnalysis] = []
    for name in sorted(set(dataset) & set(predictions), key=natural_key):
        source = dataset[name]
        try:
            preview_image = source.first_image()
            ground_truth = read_boxes(source.directory / "groundtruth.txt")
            full_occlusion = read_flags(source.directory / "full_occlusion.txt")
            out_of_view = read_flags(source.directory / "out_of_view.txt")
            prediction = read_boxes(predictions[name])
            count = min(len(ground_truth), len(full_occlusion), len(out_of_view), len(prediction))
            if count == 0:
                raise DataError("没有可共同分析的帧")
            lengths = (len(ground_truth), len(full_occlusion), len(out_of_view), len(prediction))
            if len(set(lengths)) != 1:
                warnings.append(f"{name}：真值/遮挡/出视野/预测帧数不同 {lengths}，使用前 {count} 帧")
            ground_truth = ground_truth[:count]
            full_occlusion = full_occlusion[:count]
            out_of_view = out_of_view[:count]
            prediction = prediction[:count]
            iou, distance = calculate_metrics(ground_truth, prediction)
            analyses.append(
                SequenceAnalysis(
                    source,
                    predictions[name],
                    preview_image,
                    ground_truth,
                    full_occlusion,
                    out_of_view,
                    prediction,
                    count,
                    iou,
                    distance,
                )
            )
        except (OSError, DataError) as exc:
            warnings.append(f"{name}：预测结果解析失败：{exc}")
    return analyses, missing, extra, warnings
