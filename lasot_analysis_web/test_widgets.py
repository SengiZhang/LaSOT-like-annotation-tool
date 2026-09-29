import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from widgets import PlotController


def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_keyboard_style_movement_scrolls_zoomed_view() -> None:
    app()
    controller = PlotController(100)
    controller.set_view(20, 39)
    controller.set_frame(39)
    controller.set_frame(40, ensure_visible=True)
    assert controller.frame == 40
    assert (controller.view_start, controller.view_end) == (21, 40)


def test_zoom_never_exceeds_complete_range() -> None:
    app()
    controller = PlotController(10)
    controller.zoom(5, 0.5)
    assert controller.view_end - controller.view_start + 1 == 5
    controller.zoom(5, 100)
    assert (controller.view_start, controller.view_end) == (0, 9)
