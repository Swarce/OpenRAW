"""The OpenRAW main window: queue on the left, before/after compare below it,
settings on the right. Conversions run in a worker process through the same
code as the command line (see worker.py), so both always behave alike."""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

from PySide6.QtCore import QObject, QPoint, QPointF, QRect, QRunnable, QSettings, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QDesktopServices, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog,
                               QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu,
                               QMessageBox, QProgressBar, QPushButton, QSlider, QSplitter, QStatusBar, QTableWidget,
                               QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from . import preview, style
from .worker import Runner, find_pretrained

REPO_URL = "https://github.com/Swarce/OpenRAW"
JPEG_EXTS = {".jpg", ".jpeg"}

# ------------------------------------------------------------------ building blocks


def panel(title: str, hint: str = "", actions: list[QWidget] | None = None) -> tuple[QFrame, QVBoxLayout]:
    """A bordered panel with a header strip; returns (frame, body layout)."""
    frame = QFrame(objectName="panel")
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    head = QFrame(objectName="panelHeader")
    head.setMinimumHeight(44)
    h = QHBoxLayout(head)
    h.setContentsMargins(12, 6, 8, 6)
    h.addWidget(QLabel(title, objectName="panelTitle"))
    if hint:
        h.addSpacing(8)
        h.addWidget(QLabel(hint, objectName="panelHint"))
    h.addStretch(1)
    for a in actions or []:
        h.addWidget(a)
    outer.addWidget(head)
    body = QVBoxLayout()
    body.setContentsMargins(0, 0, 0, 0)
    body.setSpacing(0)
    outer.addLayout(body, 1)
    return frame, body


def button(text: str, kind: str = "", tip: str = "") -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if tip:
        b.setToolTip(tip)
    b.setCursor(Qt.PointingHandCursor)
    return b


class Segmented(QWidget):
    """Mutually exclusive choices side by side."""
    changed = Signal(int)

    def __init__(self, labels: list[str]):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        for i, text in enumerate(labels):
            b = QPushButton(text, objectName="segment")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            self.group.addButton(b, i)
            lay.addWidget(b, 1)
        self.group.button(0).setChecked(True)
        self.group.idClicked.connect(self.changed)

    def index(self) -> int:
        return self.group.checkedId()

    def set_index(self, i: int) -> None:
        if 0 <= i < len(self.group.buttons()):
            self.group.button(i).setChecked(True)
            self.changed.emit(i)


def _qimage(a) -> QImage:
    import numpy as np
    a = np.ascontiguousarray(a)
    h, w, _ = a.shape
    return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def _fmt_secs(s: float) -> str:
    s = int(round(s))
    return f"{s // 60} min {s % 60:02d} s" if s >= 60 else f"{s} s"


def _fmt_size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.0f} KB"


# ------------------------------------------------------------------ before / after view


class CompareView(QWidget):
    """Camera JPEG left of a draggable divider, the DNG right of it.
    Fit mode scales both to the view; 100% shows pixels 1:1 -- drag to pan,
    drag the divider handle to move it."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)
        self.setMouseTracking(True)
        self.left: QImage | None = None
        self.right: QImage | None = None
        self.split = 0.5
        self.labels = ("Camera JPEG", "OpenRAW DNG")
        self.message = "Select a converted photo in the queue to compare it with its DNG"
        self.canvas = QColor("#2A3440")
        self.zoom_fit = True
        self.center = QPointF(0.5, 0.5)  # image point (fraction) at the view centre, 100% mode
        self._drag: str | None = None
        self._last = QPoint()

    def set_images(self, left: QImage | None, right: QImage | None, labels=None, message: str = "") -> None:
        if left is not None and (self.left is None or left.size() != self.left.size()):
            self.center = QPointF(0.5, 0.5)
        self.left, self.right = left, right
        if labels:
            self.labels = labels
        if message:
            self.message = message
        self.update()

    def set_fit(self, fit: bool) -> None:
        self.zoom_fit = fit
        self.update()

    def _target(self) -> QRect:
        img = self.left or self.right
        r = self.rect().adjusted(10, 10, -10, -10)
        if img is None or img.isNull():
            return r
        if self.zoom_fit:
            s = min(r.width() / img.width(), r.height() / img.height(), 1.0)
        else:
            s = 1.0 / max(self.devicePixelRatioF(), 1.0)
        w, h = int(img.width() * s), int(img.height() * s)
        if self.zoom_fit or (w <= r.width() and h <= r.height()):
            return QRect(r.x() + (r.width() - w) // 2, r.y() + (r.height() - h) // 2, w, h)
        cx = r.x() + r.width() / 2 - self.center.x() * w
        cy = r.y() + r.height() / 2 - self.center.y() * h
        return QRect(int(cx), int(cy), w, h)

    def _visible(self) -> QRect:
        return self._target().intersected(self.rect())

    def _divider_x(self) -> int:
        vis = self._visible()
        return int(vis.x() + vis.width() * self.split) if vis.width() > 0 else vis.x()

    def mousePressEvent(self, e):
        p = e.position().toPoint()
        self._last = p
        if abs(p.x() - self._divider_x()) < 10 or self.zoom_fit:
            self._drag = "split"
            self._move_split(p)
        else:
            self._drag = "pan"
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, _):
        self._drag = None
        self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, e):
        p = e.position().toPoint()
        if self._drag == "split":
            self._move_split(p)
        elif self._drag == "pan":
            t = self._target()
            d = p - self._last
            if t.width() and t.height():
                self.center = QPointF(min(1, max(0, self.center.x() - d.x() / t.width())),
                                      min(1, max(0, self.center.y() - d.y() / t.height())))
            self.update()
        else:
            near = abs(p.x() - self._divider_x()) < 10
            self.setCursor(Qt.SplitHCursor if near else (Qt.ArrowCursor if self.zoom_fit else Qt.OpenHandCursor))
        self._last = p

    def _move_split(self, p: QPoint):
        vis = self._visible()
        if vis.width() > 0:
            self.split = min(1.0, max(0.0, (p.x() - vis.x()) / vis.width()))
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom_fit)
        p.fillRect(self.rect(), self.canvas)
        if self.left is None and self.right is None:
            p.setPen(QColor("#B8C2CC"))
            p.drawText(self.rect().adjusted(20, 0, -20, 0), Qt.AlignCenter | Qt.TextWordWrap, self.message)
            return
        t = self._target()
        x = self._divider_x()
        if self.left is not None:
            p.save(); p.setClipRect(QRect(t.x(), t.y(), x - t.x(), t.height())); p.drawImage(t, self.left); p.restore()
        if self.right is not None:
            p.save(); p.setClipRect(QRect(x, t.y(), t.right() - x + 1, t.height())); p.drawImage(t, self.right)
            p.restore()
        vis = self._visible()
        p.setPen(QPen(QColor("white"), 2))
        p.drawLine(x, vis.y(), x, vis.bottom())
        p.setBrush(QColor("white"))
        p.drawRect(x - 5, vis.center().y() - 14, 10, 28)
        f = QFont(self.font()); f.setPointSizeF(8.5); f.setBold(True); p.setFont(f)
        for text, left in ((self.labels[0], True), (self.labels[1], False)):
            w = p.fontMetrics().horizontalAdvance(text) + 14
            r = QRect(max(vis.x() + 6, x - 8 - w) if left else min(vis.right() - 6 - w, x + 8), vis.bottom() - 28, w, 20)
            p.fillRect(r, QColor(20, 26, 33, 210))
            p.setPen(QColor("white"))
            p.drawText(r, Qt.AlignCenter, text)


# ------------------------------------------------------------------ queue table


class QueueTable(QTableWidget):
    dropped = Signal(list)
    remove_requested = Signal()

    def __init__(self, columns):
        super().__init__(0, len(columns))
        self.setHorizontalHeaderLabels(columns)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.empty_text = "Drop JPEG photos or folders here, or use Add photos."

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.dropped.emit(paths)
            e.acceptProposedAction()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.remove_requested.emit()
        else:
            super().keyPressEvent(e)

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.rowCount() == 0:
            p = QPainter(self.viewport())
            p.setPen(self.palette().color(self.palette().ColorRole.PlaceholderText))
            p.drawText(self.viewport().rect(), Qt.AlignCenter, self.empty_text)


# ------------------------------------------------------------------ preview loading (thread pool)


class _PreviewSignals(QObject):
    ready = Signal(int, object, object, str)  # generation, jpeg u8, dng display u8, error


class _PreviewTask(QRunnable):
    cache: dict = {}  # dng path -> (mtime, jpeg u8, dng linear)

    def __init__(self, gen: int, src: str, dst: str, mode: int, signals: _PreviewSignals):
        super().__init__()
        self.gen, self.src, self.dst, self.mode, self.signals = gen, src, dst, mode, signals

    def run(self):
        try:
            key, mtime = self.dst, os.path.getmtime(self.dst)
            hit = self.cache.get(key)
            if hit and hit[0] == mtime:
                jpeg, lin = hit[1], hit[2]
            else:
                jpeg, lin = preview.load_jpeg(self.src), preview.render_dng(self.dst)
                if len(self.cache) >= 3:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[key] = (mtime, jpeg, lin)
            h, w = min(jpeg.shape[0], lin.shape[0]), min(jpeg.shape[1], lin.shape[1])
            jpeg = jpeg[:h, :w]
            out = (self.gen, jpeg, preview.dng_display(lin[:h, :w], self.mode, jpeg), "")
        except Exception as e:  # noqa: BLE001
            out = (self.gen, None, None, f"{type(e).__name__}: {e}")
        try:
            self.signals.ready.emit(*out)
        except RuntimeError:  # the window closed meanwhile
            pass


class _GpuSignals(QObject):
    done = Signal(list, str)


class _GpuProbe(QRunnable):
    """Imports torch off the UI thread (it takes seconds) to list GPUs."""

    def __init__(self, signals: _GpuSignals):
        super().__init__()
        self.signals = signals

    def run(self):
        try:
            import torch
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] \
                if torch.cuda.is_available() else []
            ver = torch.__version__.split("+")[0]
        except Exception:  # noqa: BLE001
            names, ver = [], ""
        try:
            self.signals.done.emit(names, ver)
        except RuntimeError:  # the window closed meanwhile
            pass


# ------------------------------------------------------------------ main window

RECOMMENDED = "r2e19"  # the method new installs start on (and old defaults move to)

METHODS = [  # (key, menu label, checkpoint file or None, note)
    ("r2e19", "InvISP (recommended)", "openraw-fivek-raise-r2-e19.pth",
     "Neural network trained on real raws (FiveK + RAISE). Best results on real camera JPEGs; slow on a CPU."),
    ("e14", "InvISP · previous (e14)", "openraw-fivek-raise-e14-best.pth",
     "The earlier FiveK + RAISE model, kept for comparison."),
    ("nikon", "InvISP · upstream Nikon", "nikon.pth",
     "The original InvISP authors' Nikon D700 model."),
    ("classical", "Classical", None,
     "No neural network: deblocking, tone curve and colour steps. Fast."),
    ("custom", "Custom checkpoint…", "custom",
     "A .pth from your own training run."),
]
BIT_DEPTHS = [("12-bit (recommended)", 12), ("14-bit", 14), ("16-bit (bit-exact)", 16), ("10-bit (smallest)", 10)]
COMPRESSIONS = [("Lossless JPEG", "ljpeg"), ("None (largest)", "none")]
QUEUE_COLUMNS = ("File", "Size", "Status", "Progress", "Output")
WAITING, CONVERTING, DONE, FAILED, SKIPPED, STOPPED = "Waiting", "Converting", "Done", "Failed", "Skipped", "Stopped"
# (a job handed to the worker but not started yet is "Sent"; shown as Waiting)


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        from .. import __version__
        self.version = __version__
        self.settings = settings or QSettings("OpenRAW", "OpenRAW")
        self.theme = style.resolve(self.settings.value("theme", style.SYSTEM))
        self.jobs: list[dict] = []
        self.runner = Runner()
        self.running = False
        self.batch = dict(start=time.time(), done=0, total=0, secs=[])
        self.batch_config = None
        self.torch_version = ""
        self.preview_gen = 0
        self.preview_signals = _PreviewSignals()
        self.preview_signals.ready.connect(self._preview_ready)
        self.torch_ok = importlib.util.find_spec("torch") is not None
        self.custom_checkpoint = self.settings.value("custom_checkpoint", "") or ""
        self._wanted_device = "cpu"
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._poll)

        self.setWindowTitle("OpenRAW")
        self.setWindowIcon(QPixmap(str(style.HERE / "mark-light.png")))
        self.setAcceptDrops(True)
        self.resize(1360, 860)
        root = QWidget(objectName="work")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._shell())

        work = QSplitter(Qt.Horizontal)
        work.setHandleWidth(10)
        left = QSplitter(Qt.Vertical)
        left.setHandleWidth(10)
        left.addWidget(self._queue_panel())
        left.addWidget(self._preview_panel())
        left.setSizes([360, 440])
        work.addWidget(left)
        work.addWidget(self._settings_panel())
        work.setStretchFactor(0, 1)
        work.setSizes([1000, 350])
        wrap = QVBoxLayout()
        wrap.setContentsMargins(10, 10, 10, 10)
        wrap.addWidget(work)
        v.addLayout(wrap, 1)

        sb = QStatusBar()
        sb.setSizeGripEnabled(False)
        self.status_left = QLabel("Ready")
        self.status_right = QLabel("")
        sb.addWidget(self.status_left, 1)
        sb.addPermanentWidget(self.status_right)
        self.setStatusBar(sb)

        self._load_settings()
        self.apply_theme(self.theme)
        self._update_buttons()
        self._update_status()
        if QThreadPool.globalInstance().maxThreadCount() < 2:
            QThreadPool.globalInstance().setMaxThreadCount(2)
        if self.torch_ok:  # find out about GPUs without holding up the window
            self._gpu_signals = _GpuSignals()
            self._gpu_signals.done.connect(self._gpu_found)
            QTimer.singleShot(300, lambda: QThreadPool.globalInstance().start(_GpuProbe(self._gpu_signals)))

    # ============================================================== layout
    def _shell(self) -> QFrame:
        bar = QFrame(objectName="shell")
        bar.setFixedHeight(48)
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 0, 10, 0)
        h.setSpacing(10)
        self.logo = QLabel()
        h.addWidget(self.logo)
        h.addWidget(QLabel("OpenRAW", objectName="shellTitle"))
        sep = QFrame(objectName="shellSep")
        sep.setFixedSize(1, 20)
        h.addWidget(sep)
        h.addWidget(QLabel("JPEG to raw DNG converter", objectName="shellSub"))
        h.addStretch(1)
        h.addWidget(QLabel(f"v{self.version}", objectName="shellSub"))

        self.theme_button = QToolButton(objectName="shellButton")
        self.theme_button.setText("Theme")
        self.theme_button.setPopupMode(QToolButton.InstantPopup)
        self.theme_button.setCursor(Qt.PointingHandCursor)
        menu = QMenu(self.theme_button)
        group = QActionGroup(menu)
        current = self.settings.value("theme", style.SYSTEM)
        self.theme_actions = {}
        for name in style.THEME_CHOICES:
            a = QAction(name, menu, checkable=True)
            a.setChecked(name == current)
            a.triggered.connect(lambda _=False, n=name: self.set_theme(n))
            group.addAction(a)
            menu.addAction(a)
            self.theme_actions[name] = a
            if name == style.SYSTEM:
                menu.addSeparator()
        self.theme_button.setMenu(menu)
        h.addWidget(self.theme_button)

        for text, slot in (("Documentation", lambda: QDesktopServices.openUrl(QUrl(REPO_URL + "#readme"))),
                           ("About", self.show_about)):
            b = QToolButton(objectName="shellButton")
            b.setText(text)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(slot)
            h.addWidget(b)
        return bar

    def _queue_panel(self) -> QFrame:
        self.add_files = button("Add photos…")
        self.add_folder = button("Add folder…")
        self.clear_done = button("Clear finished", "ghost")
        frame, body = panel("Queue", "Drop JPEG files or folders here", [self.add_files, self.add_folder, self.clear_done])
        t = QueueTable(QUEUE_COLUMNS)
        t.verticalHeader().setVisible(False)
        t.setShowGrid(False)
        t.setAlternatingRowColors(True)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setWordWrap(False)
        t.setTextElideMode(Qt.ElideMiddle)
        hh = t.horizontalHeader()
        hh.setHighlightSections(False)
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for col, w in ((1, 80), (2, 190), (3, 140), (4, 240)):
            hh.setSectionResizeMode(col, QHeaderView.Interactive)
            t.setColumnWidth(col, w)
        t.verticalHeader().setDefaultSectionSize(30)
        t.setContextMenuPolicy(Qt.CustomContextMenu)
        t.customContextMenuRequested.connect(self._queue_menu)
        t.itemSelectionChanged.connect(self._selection_changed)
        t.dropped.connect(self.add_paths)
        t.remove_requested.connect(self.remove_selected)
        t.doubleClicked.connect(lambda _: self._open_selected_folder())
        self.queue = t
        body.addWidget(t)

        self.add_files.clicked.connect(self._pick_files)
        self.add_folder.clicked.connect(self._pick_folder)
        self.clear_done.clicked.connect(self.clear_finished)
        return frame

    def _preview_panel(self) -> QFrame:
        self.fit = button("Fit", "ghost")
        self.zoom100 = button("100%", "ghost")
        for b in (self.fit, self.zoom100):
            b.setCheckable(True)
        self.fit.setChecked(True)
        self.render_mode = QComboBox()
        self.render_mode.addItems(["As a raw editor opens it (flat)", "Matched to the JPEG's exposure"])
        self.render_mode.setToolTip("Flat: the DNG with no tone curve, as a raw editor shows it before its own "
                                    "default look.\nMatched: brightness and white balance fitted to the JPEG, to "
                                    "compare colour and detail.")
        frame, body = panel("Compare", "Drag the divider", [self.render_mode, self.fit, self.zoom100])
        self.compare = CompareView()
        body.addWidget(self.compare, 1)
        self.preview_info = QLabel("", objectName="infoLine")
        self.preview_info.setContentsMargins(12, 6, 12, 6)
        body.addWidget(self.preview_info)
        self.fit.clicked.connect(lambda: self._set_zoom(True))
        self.zoom100.clicked.connect(lambda: self._set_zoom(False))
        self.render_mode.currentIndexChanged.connect(lambda _: self._selection_changed())
        return frame

    def _settings_panel(self) -> QFrame:
        frame, body = panel("Settings")
        frame.setMinimumWidth(330)
        frame.setMaximumWidth(420)
        g = QGridLayout()
        g.setContentsMargins(14, 6, 14, 14)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(8)
        g.setColumnStretch(1, 1)
        row = 0

        def section(title: str):
            nonlocal row
            g.addWidget(QLabel(title.upper(), objectName="groupTitle"), row, 0, 1, 2)
            row += 1

        def field(label: str, w: QWidget) -> QLabel:
            nonlocal row
            lab = QLabel(label, objectName="fieldLabel")
            g.addWidget(lab, row, 0, Qt.AlignVCenter)
            g.addWidget(w, row, 1)
            row += 1
            return lab

        def note(text: str) -> QLabel:
            nonlocal row
            n = QLabel(text, objectName="note")
            n.setWordWrap(True)
            g.addWidget(n, row, 1)
            row += 1
            return n

        section("Reconstruction")
        self.method = QComboBox()
        for key, label, ckpt, _ in METHODS:
            self.method.addItem(label, key)
            if ckpt not in (None, "custom") and not (self.torch_ok and find_pretrained(ckpt)):
                i = self.method.count() - 1
                self.method.model().item(i).setEnabled(False)
                why = "needs PyTorch: pip install 'openraw[invisp]'" if not self.torch_ok else f"{ckpt} not found"
                self.method.setItemData(i, why, Qt.ToolTipRole)
        if not self.torch_ok:
            self.method.model().item(len(METHODS) - 1).setEnabled(False)
        field("Method", self.method)
        self.method_note = note("")
        self.device = QComboBox()
        self.device.addItem("CPU", "cpu")
        field("Run on", self.device)

        section("Output DNG")
        self.layout_sel = Segmented(["Linear", "Bayer (CFA)"])
        field("Layout", self.layout_sel)
        note("Bayer: a mosaic your editor demosaics, like a camera raw; ~3x smaller. Keep Linear for "
             "graphics and screenshots.")
        aa = QWidget()
        ah = QHBoxLayout(aa)
        ah.setContentsMargins(0, 0, 0, 0)
        ah.setSpacing(8)
        self.aa_slider = QSlider(Qt.Horizontal)
        self.aa_slider.setRange(0, 100)
        self.aa_slider.setSingleStep(5)
        self.aa_slider.setPageStep(25)
        self.aa_value = QLabel("Off")
        self.aa_value.setFixedWidth(34)
        ah.addWidget(self.aa_slider, 1)
        ah.addWidget(self.aa_value)
        self.aa_label = field("Anti-aliasing", aa)
        self.aa_note = note("Emulates a camera's anti-aliasing filter: less moiré on fine detail, slightly "
                            "softer. Try 0.50.")
        self.aa_widget = aa
        self.bit_depth = QComboBox()
        for label, v in BIT_DEPTHS:
            self.bit_depth.addItem(label, v)
        field("Bit depth", self.bit_depth)
        self.compression = QComboBox()
        for label, v in COMPRESSIONS:
            self.compression.addItem(label, v)
        field("Compression", self.compression)
        self.keep_exif = QCheckBox("Copy camera and lens metadata")
        self.keep_exif.setChecked(True)
        g.addWidget(self.keep_exif, row, 1)
        row += 1
        note("GPS location is never copied.")
        self.preview_chk = QCheckBox("Embed a preview image")
        self.preview_chk.setChecked(True)
        g.addWidget(self.preview_chk, row, 1)
        row += 1

        section("Destination")
        self.dest_mode = Segmented(["Next to originals", "Folder"])
        field("Save to", self.dest_mode)
        dest = QWidget()
        dh = QHBoxLayout(dest)
        dh.setContentsMargins(0, 0, 0, 0)
        dh.setSpacing(6)
        self.dest_path = QLineEdit()
        self.dest_path.setPlaceholderText("Choose a folder…")
        self.dest_browse = button("Browse")
        dh.addWidget(self.dest_path, 1)
        dh.addWidget(self.dest_browse)
        self.dest_label = field("Folder", dest)
        self.dest_widget = dest
        self.overwrite = QCheckBox("Replace existing DNGs")
        g.addWidget(self.overwrite, row, 1)
        row += 1

        holder = QWidget()
        holder.setLayout(g)
        body.addWidget(holder)
        body.addStretch(1)

        foot = QFrame(objectName="panelFoot")
        fh = QHBoxLayout(foot)
        fh.setContentsMargins(14, 10, 14, 10)
        self.stop_button = button("Stop")
        self.convert_button = button("Convert", "primary")
        for b in (self.stop_button, self.convert_button):
            b.setMinimumHeight(30)
        fh.addWidget(self.stop_button)
        fh.addWidget(self.convert_button, 1)
        body.addWidget(foot)

        self.method.currentIndexChanged.connect(self._method_changed)
        self.layout_sel.changed.connect(self._layout_changed)
        self.dest_mode.changed.connect(self._dest_changed)
        self.aa_slider.valueChanged.connect(lambda v: self.aa_value.setText("Off" if v == 0 else f"{v / 100:.2f}"))
        self.dest_browse.clicked.connect(self._pick_dest)
        self.convert_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.convert_button.setShortcut(QKeySequence("Ctrl+Return"))
        return frame

    # ============================================================== settings
    def _load_settings(self) -> None:
        s = self.settings

        def get(key, default, kind=str):
            v = s.value(key, default)
            if kind is bool:
                return v in (True, "true", "1", 1)
            try:
                return kind(v)
            except (TypeError, ValueError):
                return default

        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        keys = [self.method.itemData(i) for i in range(self.method.count())]
        want = get("method", RECOMMENDED)
        # Settings saved while an older model was the recommended one still name it.
        # Someone who was on the recommended model moves to the new one; anyone who
        # picked another method keeps it. (Releases before 0.1.0a2 recommended "e14".)
        if want == (s.value("recommended") or "e14") and want != RECOMMENDED:
            want = RECOMMENDED
        i = keys.index(want) if want in keys else 0
        if not self.method.model().item(i).isEnabled() or (want == "custom" and not Path(self.custom_checkpoint).is_file()):
            i = next((j for j in range(self.method.count()) if self.method.model().item(j).isEnabled()
                      and keys[j] != "custom"), 0)
        self.method.blockSignals(True)
        self.method.setCurrentIndex(i)
        self.method.blockSignals(False)
        self._method_changed(i)
        self.layout_sel.set_index(get("layout", 0, int))
        self.aa_slider.setValue(get("cfa_aa", 0, int))
        bd = [self.bit_depth.itemData(i) for i in range(self.bit_depth.count())]
        want_bd = get("bit_depth", 12, int)
        self.bit_depth.setCurrentIndex(bd.index(want_bd) if want_bd in bd else 0)
        self.compression.setCurrentIndex(1 if get("compression", "ljpeg") == "none" else 0)
        self.keep_exif.setChecked(get("exif", True, bool))
        self.preview_chk.setChecked(get("embed_preview", True, bool))
        self.dest_mode.set_index(get("dest_mode", 0, int))
        self.dest_path.setText(get("dest_path", ""))
        self.overwrite.setChecked(get("overwrite", False, bool))
        self.render_mode.setCurrentIndex(get("render_mode", 0, int))
        self._wanted_device = get("device", "cpu")

    def save_settings(self) -> None:
        s = self.settings
        s.setValue("geometry", self.saveGeometry())
        s.setValue("method", self.method.currentData())
        s.setValue("device", self.device.currentData())
        s.setValue("layout", self.layout_sel.index())
        s.setValue("cfa_aa", self.aa_slider.value())
        s.setValue("bit_depth", self.bit_depth.currentData())
        s.setValue("compression", self.compression.currentData())
        s.setValue("exif", self.keep_exif.isChecked())
        s.setValue("embed_preview", self.preview_chk.isChecked())
        s.setValue("dest_mode", self.dest_mode.index())
        s.setValue("dest_path", self.dest_path.text())
        s.setValue("overwrite", self.overwrite.isChecked())
        s.setValue("render_mode", self.render_mode.currentIndex())
        s.setValue("custom_checkpoint", self.custom_checkpoint)
        s.setValue("recommended", RECOMMENDED)
        s.sync()

    # ============================================================== themes
    def set_theme(self, name: str) -> None:
        self.settings.setValue("theme", name)
        if name in self.theme_actions:
            self.theme_actions[name].setChecked(True)
        self.apply_theme(style.resolve(name))

    def apply_theme(self, theme: style.Theme) -> None:
        self.theme = theme
        app = QApplication.instance()
        if app is not None:
            style.apply(app, theme)
        self.logo.setPixmap(QPixmap(str(style.HERE / theme.shell_logo)).scaled(
            28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.compare.canvas = QColor(theme.canvas)
        self.compare.update()
        for r in range(len(self.jobs)):
            self._paint_row(r)

    # ============================================================== queue
    def add_paths(self, paths) -> int:
        """Add JPEGs, and every JPEG inside folders (recursively; the folder's
        structure is mirrored when saving to a destination folder)."""
        known = {j["src"].resolve() for j in self.jobs}
        added = 0
        for raw in paths:
            p = Path(raw)
            items = []
            if p.is_dir():
                items = [(f, p) for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in JPEG_EXTS]
            elif p.is_file() and p.suffix.lower() in JPEG_EXTS:
                items = [(p, None)]
            for f, base in items:
                key = f.resolve()
                if key in known:
                    continue
                known.add(key)
                try:
                    size = f.stat().st_size
                except OSError:
                    size = 0
                self.jobs.append(dict(src=f, base=base, size=size, status=WAITING, detail="", progress=None,
                                      dst=None, result=None, stage=""))
                self._append_row(len(self.jobs) - 1)
                added += 1
        self._update_buttons()
        self._update_status()
        if added == 0 and paths:
            self.status_left.setText("No new JPEG photos found there")
        return added

    def _append_row(self, i: int) -> None:
        t = self.queue
        t.insertRow(i)
        for c in range(len(QUEUE_COLUMNS)):
            t.setItem(i, c, QTableWidgetItem(""))
        self._paint_row(i)

    def _paint_row(self, r: int) -> None:
        job = self.jobs[r]
        th = self.theme
        t = self.queue
        t.item(r, 0).setText(job["src"].name)
        t.item(r, 0).setToolTip(str(job["src"]))
        t.item(r, 1).setText(_fmt_size(job["size"]) if job["size"] else "")
        shown = WAITING if job["status"] == "Sent" else job["status"]
        status = shown + (f" · {job['stage']}" if shown == CONVERTING and job["stage"] else "")
        colour = {DONE: th.ok, FAILED: th.err, CONVERTING: th.accent}.get(job["status"], th.text_dim)
        it = t.item(r, 2)
        it.setText(("○  " if shown in (WAITING, SKIPPED, STOPPED) else "●  ") + status)
        it.setForeground(QColor(colour))
        it.setToolTip(job["detail"])
        prog = job["progress"]
        if isinstance(prog, float):
            cell = t.cellWidget(r, 3)
            if cell is None:
                cell = QWidget()
                lay = QHBoxLayout(cell)
                lay.setContentsMargins(8, 0, 12, 0)
                b = QProgressBar()
                b.setRange(0, 1000)
                b.setTextVisible(False)
                lay.addWidget(b)
                t.setCellWidget(r, 3, cell)
            cell.findChild(QProgressBar).setValue(int(prog * 1000))
            t.item(r, 3).setText("")
        else:
            if t.cellWidget(r, 3) is not None:
                t.removeCellWidget(r, 3)
            t.item(r, 3).setText(prog or "")
            t.item(r, 3).setForeground(QColor(th.text_dim))
        out = t.item(r, 4)
        if job["status"] == FAILED:
            out.setText(job["detail"])
            out.setForeground(QColor(th.err))
        else:
            out.setText(str(job["dst"]) if job["dst"] and job["status"] in (DONE, SKIPPED) else "")
            out.setForeground(QColor(th.text_dim))
        out.setToolTip(out.text())

    def selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.queue.selectedIndexes()})

    def remove_selected(self) -> None:
        for r in reversed(self.selected_rows()):
            job = self.jobs[r]
            if job["status"] == CONVERTING:
                continue
            if job["status"] == "Sent":
                continue  # the worker is picking it up right now
            if self.running and job["status"] == WAITING and job.get("id"):
                self.batch["total"] -= 1
            del self.jobs[r]
            self.queue.removeRow(r)
        self._update_buttons()
        self._update_status()

    def clear_finished(self) -> None:
        for r in reversed(range(len(self.jobs))):
            if self.jobs[r]["status"] in (DONE, SKIPPED):
                del self.jobs[r]
                self.queue.removeRow(r)
        self._update_buttons()
        self._update_status()

    def _queue_menu(self, pos) -> None:
        rows = self.selected_rows()
        if not rows:
            return
        m = QMenu(self)
        a_open = m.addAction("Show the DNG in its folder")
        a_open.setEnabled(any(self.jobs[r]["status"] in (DONE, SKIPPED) for r in rows))
        a_src = m.addAction("Show the JPEG in its folder")
        m.addSeparator()
        a_rm = m.addAction("Remove from queue")
        a_rm.setEnabled(any(self.jobs[r]["status"] != CONVERTING for r in rows))
        chosen = m.exec(self.queue.viewport().mapToGlobal(pos))
        if chosen == a_open:
            self._open_selected_folder()
        elif chosen == a_src:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.jobs[rows[0]]["src"].parent)))
        elif chosen == a_rm:
            self.remove_selected()

    def _open_selected_folder(self) -> None:
        rows = self.selected_rows()
        if rows and self.jobs[rows[0]]["dst"] and self.jobs[rows[0]]["status"] in (DONE, SKIPPED):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.jobs[rows[0]]["dst"]).parent)))

    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add photos", self.settings.value("last_dir", "") or "",
                                                "JPEG photos (*.jpg *.jpeg *.JPG *.JPEG)")
        if files:
            self.settings.setValue("last_dir", str(Path(files[0]).parent))
            self.add_paths(files)

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Add a folder of photos", self.settings.value("last_dir", "") or "")
        if d:
            self.settings.setValue("last_dir", d)
            self.add_paths([d])

    def _pick_dest(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Save DNGs to", self.dest_path.text() or "")
        if d:
            self.dest_path.setText(d)
            self.dest_mode.set_index(1)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    # ============================================================== settings logic
    def _method_changed(self, i: int) -> None:
        key = self.method.itemData(i)
        note = METHODS[i][3]
        if key == "custom":
            if not self.custom_checkpoint or not Path(self.custom_checkpoint).is_file():
                f, _ = QFileDialog.getOpenFileName(self, "Choose an InvISP checkpoint", "", "Checkpoints (*.pth)")
                if not f:
                    self.method.setCurrentIndex(0)
                    return
                self.custom_checkpoint = f
            note = f"{Path(self.custom_checkpoint).name}"
        self.method_note.setText(note)
        self.device.setEnabled(key != "classical" and not self.running)
        self._update_status()

    def _layout_changed(self, i: int) -> None:
        for w in (self.aa_widget, self.aa_label, self.aa_note):
            w.setEnabled(i == 1 and not self.running)

    def _dest_changed(self, i: int) -> None:
        for w in (self.dest_widget, self.dest_label):
            w.setEnabled(i == 1 and not self.running)

    def _gpu_found(self, names: list, torch_version: str) -> None:
        self.torch_version = torch_version
        for i, n in enumerate(names):
            self.device.addItem(f"GPU · {n}", f"cuda:{i}")
        idx = self.device.findData(self._wanted_device)
        if idx >= 0:
            self.device.setCurrentIndex(idx)
        self._update_status()

    def config(self):
        from ..pipeline import PipelineConfig
        key = self.method.currentData()
        kw = {}
        if key != "classical":
            ckpt = self.custom_checkpoint if key == "custom" else find_pretrained(METHODS[self.method.currentIndex()][2])
            kw = dict(use_invisp=True, invisp_checkpoint=str(ckpt or ""), invisp_device=self.device.currentData() or "cpu")
        cfa = self.layout_sel.index() == 1
        return PipelineConfig(dng_layout="cfa" if cfa else "linear",
                              dng_cfa_olpf=self.aa_slider.value() / 100 if cfa else 0.0,
                              dng_bit_depth=self.bit_depth.currentData(),
                              dng_compression=self.compression.currentData(),
                              preserve_exif=self.keep_exif.isChecked(),
                              write_preview=self.preview_chk.isChecked(), **kw)

    def destination(self, job: dict) -> Path:
        src, base = job["src"], job["base"]
        if self.dest_mode.index() == 1 and self.dest_path.text().strip():
            out = Path(self.dest_path.text().strip()).expanduser()
            return out / src.relative_to(base).with_suffix(".dng") if base is not None else out / (src.stem + ".dng")
        return src.with_suffix(".dng")

    # ============================================================== running
    def start(self) -> None:
        if self.running:
            return
        if self.dest_mode.index() == 1 and not self.dest_path.text().strip():
            QMessageBox.information(self, "OpenRAW", "Choose a destination folder first, or save next to the originals.")
            return
        todo = [j for j in self.jobs if j["status"] in (WAITING, STOPPED, FAILED)]
        if not todo:
            return
        config = self.config()
        if config.use_invisp and not Path(config.invisp_checkpoint or "").is_file():
            QMessageBox.warning(self, "OpenRAW", "The selected model's checkpoint file wasn't found.")
            return
        self.save_settings()
        taken: dict[Path, Path] = {}
        self.batch = dict(start=time.time(), done=0, total=0, secs=[])
        self.batch_config = config
        overwrite = self.overwrite.isChecked()
        for j in todo:
            dst = self.destination(j)
            j.update(dst=dst, result=None, stage="", detail="", progress=None, id=None)
            k = dst.resolve()
            if k in taken:
                j.update(status=FAILED, detail=f"Same output file as {taken[k].name}")
            elif dst.exists() and not overwrite:
                j.update(status=SKIPPED, detail="A DNG with this name already exists", progress="already there")
            else:
                taken[k] = j["src"]
                j.update(status=WAITING, id=id(j))
                self.batch["total"] += 1
            self._paint_row(self.jobs.index(j))
        if self.batch["total"]:
            self.running = True
            self._submit_next()
            self.timer.start()
        else:
            self._update_status(final=True)
        self._update_buttons()
        if self.running:
            self._update_status()

    def stop(self) -> None:
        if not self.running:
            return
        self.runner.stop([j["dst"] for j in self.jobs if j["status"] == CONVERTING])
        for r in reversed(range(len(self.jobs))):
            j = self.jobs[r]
            if j["status"] == CONVERTING:
                j.update(status=STOPPED, stage="", progress=None, detail="Stopped before it finished")
            elif j["status"] == "Sent":
                j["status"] = WAITING
        self._finish_batch(stopped=True)

    def _submit_next(self) -> bool:
        """Hand the worker the next queued job. One at a time, so removing a
        waiting photo from the queue simply means it is never sent."""
        nxt = next((j for j in self.jobs if j["status"] == WAITING and j.get("id")), None)
        if nxt is None:
            return False
        nxt["status"] = "Sent"
        self.runner.submit(nxt["id"], nxt["src"], nxt["dst"], self.batch_config)
        return True

    def _job(self, jid):
        return next((j for j in self.jobs if j.get("id") == jid), None)

    def _poll(self) -> None:
        for ev in self.runner.poll():
            kind = ev[0]
            if kind == "crashed":
                for j in self.jobs:
                    if j["status"] == CONVERTING:
                        self.batch["done"] += 1
                        j.update(status=FAILED, progress=None, stage="",
                                 detail=f"The converter stopped unexpectedly (exit code {ev[1]}); out of memory?")
                        self._paint_row(self.jobs.index(j))
                for j in self.jobs:
                    if j["status"] == "Sent":
                        j["status"] = WAITING
                if not self._submit_next():
                    self._finish_batch()
                continue
            j = self._job(ev[1])
            if j is None:
                continue
            if kind == "started":
                j.update(status=CONVERTING, progress=0.0, stage="Starting", started=time.time())
            elif kind == "progress":
                j.update(stage=ev[2], progress=float(ev[3]))
            elif kind == "finished":
                r = ev[2]
                self.batch["done"] += 1
                if r.get("ok"):
                    j.update(status=DONE, stage="", progress=_fmt_secs(r["secs"]), result=r, detail="")
                    self.batch["secs"].append(r["secs"])
                else:
                    j.update(status=FAILED, stage="", progress=None, detail=r.get("error", "failed"))
            self._paint_row(self.jobs.index(j))
            if kind == "finished":
                rows = self.selected_rows()
                if rows and self.jobs[rows[0]] is j:
                    self._selection_changed()
                if not self._submit_next():
                    self._finish_batch()
        if self.running:
            self._update_status()
            self._update_buttons()

    def _finish_batch(self, stopped: bool = False) -> None:
        self.timer.stop()
        self.running = False
        for r in range(len(self.jobs)):
            self._paint_row(r)
        self._update_buttons()
        self._update_status(final=True, stopped=stopped)
        QApplication.alert(self)

    def _update_buttons(self) -> None:
        waiting = sum(j["status"] in (WAITING, STOPPED, FAILED) for j in self.jobs)
        self.convert_button.setEnabled(not self.running and waiting > 0)
        self.stop_button.setEnabled(self.running)
        if self.running:
            self.convert_button.setText(f"Converting… {self.batch['done']} / {self.batch['total']}")
        else:
            self.convert_button.setText(f"Convert {waiting} photo{'s' if waiting != 1 else ''}" if waiting else "Convert")
        for w in (self.method, self.layout_sel, self.aa_slider, self.bit_depth, self.compression,
                  self.keep_exif, self.preview_chk, self.dest_mode, self.overwrite):
            w.setEnabled(not self.running)
        self._layout_changed(self.layout_sel.index())
        self._dest_changed(self.dest_mode.index())
        self.device.setEnabled(self.method.currentData() != "classical" and not self.running)

    def _update_status(self, final: bool = False, stopped: bool = False) -> None:
        key = self.method.currentData()
        right = self.method.currentText().replace(" (recommended)", "")
        if key != "classical":
            right += f" on {self.device.currentText()}"
            if self.torch_version:
                right += f"  ·  PyTorch {self.torch_version}"
        self.status_right.setText(right)
        visible = self.jobs
        n = len(visible)
        done = sum(j["status"] == DONE for j in visible)
        failed = sum(j["status"] == FAILED for j in visible)
        skipped = sum(j["status"] == SKIPPED for j in visible)
        if self.running:
            b = self.batch
            parts = [f"{b['done']} of {b['total']} converted"]
            if failed:
                parts.append(f"{failed} failed")
            remaining = b["total"] - b["done"]
            if b["secs"] and remaining:
                cur = next((j for j in self.jobs if j["status"] == CONVERTING), None)
                avg = sum(b["secs"]) / len(b["secs"])
                eta = avg * remaining - (time.time() - cur["started"] if cur and cur.get("started") else 0)
                parts.append(f"about {_fmt_secs(max(eta, 1))} left")
            self.status_left.setText("  ·  ".join(parts))
        elif final:
            took = _fmt_secs(time.time() - self.batch["start"])
            parts = [("Stopped" if stopped else "Finished") + f" after {took}", f"{done} done"]
            if failed:
                parts.append(f"{failed} failed")
            if skipped:
                parts.append(f"{skipped} skipped (DNG already there)")
            self.status_left.setText("  ·  ".join(parts))
        else:
            self.status_left.setText(f"{n} photo{'s' if n != 1 else ''} in the queue" if n else "Ready")

    # ============================================================== compare
    def _set_zoom(self, fit: bool) -> None:
        self.fit.setChecked(fit)
        self.zoom100.setChecked(not fit)
        self.compare.set_fit(fit)

    def _selection_changed(self) -> None:
        rows = self.selected_rows()
        self.preview_gen += 1
        if not rows:
            return
        job = self.jobs[rows[0]]
        dst = job.get("dst")
        if job["status"] not in (DONE, SKIPPED) or not dst or not Path(dst).is_file():
            msg = {CONVERTING: "Converting… the comparison appears when it's done.",
                   FAILED: f"Conversion failed: {job['detail']}"}.get(
                       job["status"], "Convert this photo to compare it with its DNG.")
            self.compare.set_images(None, None, message=msg)
            self.preview_info.setText(str(job["src"]))
            return
        if not preview.available():
            self.compare.set_images(None, None, message="Comparing needs rawpy: pip install rawpy")
            return
        self.compare.set_images(None, None, message="Rendering the DNG…")
        r = job.get("result") or {}
        info = [job["src"].name]
        if r.get("w"):
            info.append(f"{r['w']} × {r['h']}")
        if r.get("q"):
            info.append(f"JPEG quality ≈ {r['q']:.0f}")
        try:
            info.append(f"DNG {_fmt_size(Path(dst).stat().st_size)}")
        except OSError:
            pass
        if r.get("secs"):
            info.append(f"converted in {_fmt_secs(r['secs'])}")
        self.preview_info.setText("  ·  ".join(info))
        QThreadPool.globalInstance().start(_PreviewTask(self.preview_gen, str(job["src"]), str(dst),
                                                        self.render_mode.currentIndex(), self.preview_signals))

    def _preview_ready(self, gen: int, jpeg, dng, error: str) -> None:
        if gen != self.preview_gen:
            return  # the selection changed since
        if error:
            self.compare.set_images(None, None, message=f"Couldn't render the DNG: {error}")
            return
        mode = "as your editor opens it" if self.render_mode.currentIndex() == 0 else "matched to the JPEG"
        self.compare.set_images(_qimage(jpeg), _qimage(dng), ("Camera JPEG", f"OpenRAW DNG · {mode}"))

    # ============================================================== misc
    def show_about(self) -> None:
        QMessageBox.about(
            self, "About OpenRAW",
            f"<h3>OpenRAW {self.version}</h3>"
            "<p>Turns JPEGs into editable raw DNGs, linear or Bayer mosaic.</p>"
            f"<p><a href='{REPO_URL}'>{REPO_URL}</a></p>"
            "<p>Licence: CC BY-NC-SA 4.0 (non-commercial use, with credit, under the same terms). The "
            "included model weights were trained on MIT-Adobe FiveK and RAISE and are for non-commercial "
            "research use.</p>"
            "<p>Built on <b>InvISP</b> (Xing, Qian &amp; Chen, CVPR 2021). Training ideas from <b>Edit-aware "
            "RAW Reconstruction</b> (Punnappurath et al., CVPR 2026), <b>RawGen</b> (Kim et al., ECCV 2026), "
            "<b>SpiralDiff</b> (Yue et al., CVPR 2026) and <b>ParamISP</b> (Kim et al., CVPR 2024). "
            "Full credits in NOTICE.md.</p><p>Saira typeface: SIL Open Font License.</p>")

    def closeEvent(self, e):
        if self.running:
            ans = QMessageBox.question(self, "OpenRAW", "Conversions are still running. Stop them and quit?")
            if ans != QMessageBox.Yes:
                e.ignore()
                return
            self.stop()
        self.save_settings()
        self.runner.shutdown()
        e.accept()
