"""Look and feel: a quiet, dense, squared-off workbench in the spirit of
enterprise suites -- a shell bar, a calm work area, one accent colour, 1 px
borders, no rounded corners. Saira (bundled, OFL) for headings; the
platform's UI font for everything else, so it reads natively everywhere.

Every colour comes from a Theme; the stylesheet is one template filled in
per theme, so adding a theme is adding one entry to THEMES."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Theme:
    name: str
    dark: bool
    shell: str          # top bar
    shell_text: str
    shell_dim: str
    shell_hover: str
    bg: str             # work area behind the panels
    panel: str
    border: str
    header: str         # panel / table headers
    text: str
    text_dim: str
    input_border: str
    accent: str
    accent_hover: str
    accent_text: str    # text on an accent background
    accent_soft: str    # selection / hover tint
    ok: str
    err: str
    canvas: str         # behind the compare images
    alt_row: str
    disabled: str
    border_width: int = 1
    shell_logo: str = "mark-dark.png"  # light strokes, for a dark bar


THEMES: dict[str, Theme] = {t.name: t for t in (
    Theme("Light", False,
          shell="#1D2D3E", shell_text="#FFFFFF", shell_dim="#A9B4BE", shell_hover="#263A4F",
          bg="#EEF0F2", panel="#FFFFFF", border="#D5DADD", header="#F5F6F7", text="#1D2D3E",
          text_dim="#5B6B79", input_border="#89919A", accent="#0A6ED1", accent_hover="#085CAF",
          accent_text="#FFFFFF", accent_soft="#E5F0FA", ok="#256F3A", err="#BB0000", canvas="#2A3440",
          alt_row="#FAFBFB", disabled="#9AA5AF"),
    Theme("Dark", True,
          shell="#0B1118", shell_text="#F2F4F6", shell_dim="#8A99A8", shell_hover="#1A2632",
          bg="#11171E", panel="#1A222B", border="#2C3844", header="#202A34", text="#E3E8ED",
          text_dim="#93A2B0", input_border="#4A5866", accent="#4DB1FF", accent_hover="#7AC4FF",
          accent_text="#0B1118", accent_soft="#1E3A55", ok="#5DD47A", err="#FF6B6B", canvas="#0B1015",
          alt_row="#1D262F", disabled="#56636F"),
    Theme("Darkroom", True,  # a safelight: deep neutral greys with a red accent
          shell="#0A0A0A", shell_text="#EDEDED", shell_dim="#8C8C8C", shell_hover="#1C1C1C",
          bg="#121212", panel="#1A1A1A", border="#2E2E2E", header="#202020", text="#E6E6E6",
          text_dim="#9A9A9A", input_border="#4D4D4D", accent="#E5484D", accent_hover="#F2666A",
          accent_text="#FFFFFF", accent_soft="#3A1E20", ok="#62C97A", err="#FF8A8A", canvas="#080808",
          alt_row="#1E1E1E", disabled="#555555"),
    Theme("High contrast dark", True,  # black, white borders, orange accent
          shell="#000000", shell_text="#FFFFFF", shell_dim="#FFFFFF", shell_hover="#262626",
          bg="#000000", panel="#000000", border="#FFFFFF", header="#000000", text="#FFFFFF",
          text_dim="#F0F0F0", input_border="#FFFFFF", accent="#FF9E1B", accent_hover="#FFB54D",
          accent_text="#000000", accent_soft="#4D2E00", ok="#7CFF8A", err="#FF7A7A", canvas="#000000",
          alt_row="#000000", disabled="#9A9A9A", border_width=2),
    Theme("High contrast light", False,  # white, black borders, dark orange accent
          shell="#FFFFFF", shell_text="#000000", shell_dim="#000000", shell_hover="#FFE3C2",
          bg="#FFFFFF", panel="#FFFFFF", border="#000000", header="#FFFFFF", text="#000000",
          text_dim="#1A1A1A", input_border="#000000", accent="#A84300", accent_hover="#7F3300",
          accent_text="#FFFFFF", accent_soft="#FFE3C2", ok="#0B6623", err="#B00000", canvas="#000000",
          alt_row="#FFFFFF", disabled="#6B6B6B", border_width=2, shell_logo="mark-light.png"),
    Theme("Solarized", False,  # warm paper tones for long sessions
          shell="#073642", shell_text="#FDF6E3", shell_dim="#93A1A1", shell_hover="#0D4553",
          bg="#EEE8D5", panel="#FDF6E3", border="#D9D2BE", header="#F5EFDC", text="#073642",
          text_dim="#586E75", input_border="#93A1A1", accent="#268BD2", accent_hover="#1F72AD",
          accent_text="#FFFFFF", accent_soft="#E3EEF2", ok="#5F7A00", err="#C62B28", canvas="#073642",
          alt_row="#F8F2E0", disabled="#A3AFAF"),
)}

DEFAULT_THEME = "Light"
SYSTEM = "Match system"  # Light or Dark, following the operating system
THEME_CHOICES = [SYSTEM] + list(THEMES)


def _icons(t: Theme) -> dict[str, str]:
    """Small SVGs in the theme's colours (check mark, combo arrow), written
    to a temp folder: a stylesheet can only take icons by file path."""
    d = Path(tempfile.gettempdir()) / "openraw-gui-icons"
    d.mkdir(exist_ok=True)
    icons = {
        "check": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14"><path d="M3 7.2l2.6 2.6L11 4.4" '
                 f'fill="none" stroke="{t.accent_text}" stroke-width="2"/></svg>',
        "arrow": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6"><path d="M1 1l4 4 4-4" '
                 f'fill="none" stroke="{t.text_dim}" stroke-width="1.5"/></svg>',
    }
    out = {}
    for name, svg in icons.items():
        p = d / f"{name}-{hashlib.sha1(svg.encode()).hexdigest()[:10]}.svg"
        if not p.exists():
            p.write_text(svg)
        out[name] = p.as_posix()
    return out


def stylesheet(t: Theme) -> str:
    i = _icons(t)
    bw = t.border_width
    return f"""
* {{ color: {t.text}; }}
QMainWindow, QWidget#work, QDialog {{ background: {t.bg}; }}
QToolTip {{ background: {t.panel}; color: {t.text}; border: 1px solid {t.border}; padding: 4px; }}

QFrame#shell {{ background: {t.shell}; border: none; border-bottom: {bw - 1}px solid {t.border}; }}
QLabel#shellTitle {{ color: {t.shell_text}; font-family: "Saira SemiBold"; font-size: 18px; }}
QLabel#shellSub {{ color: {t.shell_dim}; font-size: 12px; }}
QFrame#shellSep {{ background: {t.shell_dim}; }}
QToolButton#shellButton {{ color: {t.shell_text}; background: transparent; border: 1px solid transparent;
    padding: 4px 10px; font-size: 12px; }}
QToolButton#shellButton:hover {{ border-color: {t.shell_dim}; background: {t.shell_hover}; }}
QToolButton#shellButton::menu-indicator {{ image: none; width: 0; }}

QFrame#panel {{ background: {t.panel}; border: {bw}px solid {t.border}; }}
QFrame#panelHeader {{ background: {t.header}; border: none; border-bottom: {bw}px solid {t.border}; }}
QFrame#panelFoot {{ background: {t.header}; border: none; border-top: {bw}px solid {t.border}; }}
QLabel#panelTitle {{ font-family: "Saira SemiBold"; font-size: 15px; }}
QLabel#panelHint {{ color: {t.text_dim}; font-size: 11px; }}
QLabel#groupTitle {{ font-family: "Saira SemiBold"; font-size: 12px; color: {t.text_dim};
    letter-spacing: 1px; padding-top: 6px; }}
QLabel#fieldLabel {{ color: {t.text_dim}; font-size: 12px; }}
QLabel#fieldLabel:disabled, QLabel#note:disabled {{ color: {t.disabled}; }}
QLabel#note {{ color: {t.text_dim}; font-size: 11px; }}
QLabel#infoLine {{ color: {t.text_dim}; font-size: 11px; }}

QPushButton {{ background: {t.panel}; border: {bw}px solid {t.accent}; color: {t.accent}; padding: 5px 14px;
    font-size: 12px; min-height: 18px; }}
QPushButton:hover {{ background: {t.accent_soft}; }}
QPushButton:disabled {{ border-color: {t.border}; color: {t.disabled}; }}
QPushButton#primary {{ background: {t.accent}; color: {t.accent_text}; font-weight: 600; }}
QPushButton#primary:hover {{ background: {t.accent_hover}; border-color: {t.accent_hover}; }}
QPushButton#primary:disabled {{ background: {t.header}; border-color: {t.disabled}; color: {t.disabled}; }}
QPushButton#ghost {{ border-color: transparent; }}
QPushButton#ghost:hover {{ border-color: {t.border}; background: {t.header}; }}
QPushButton#ghost:checked {{ border-color: {t.accent}; color: {t.accent}; }}
QPushButton#segment {{ border: {bw}px solid {t.input_border}; color: {t.text}; background: {t.panel};
    padding: 4px 10px; }}
QPushButton#segment:checked {{ background: {t.accent}; color: {t.accent_text}; border-color: {t.accent};
    font-weight: 600; }}
QPushButton#segment:disabled {{ color: {t.disabled}; border-color: {t.border}; }}
QPushButton#segment:checked:disabled {{ background: {t.accent_soft}; color: {t.text}; border-color: {t.border}; }}

QComboBox, QLineEdit {{ background: {t.panel}; border: {bw}px solid {t.input_border};
    padding: 3px 6px; min-height: 18px; font-size: 12px; selection-background-color: {t.accent};
    selection-color: {t.accent_text}; }}
QComboBox:hover, QLineEdit:hover {{ border-color: {t.accent}; }}
QComboBox:focus, QLineEdit:focus {{ border: {bw}px solid {t.accent}; }}
QComboBox:disabled, QLineEdit:disabled {{ color: {t.disabled}; border-color: {t.border}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url({i['arrow']}); width: 10px; height: 6px; }}
QComboBox QAbstractItemView {{ background: {t.panel}; border: {bw}px solid {t.border};
    selection-background-color: {t.accent_soft}; selection-color: {t.text}; outline: none; }}
QCheckBox {{ font-size: 12px; spacing: 6px; }}
QCheckBox:disabled {{ color: {t.disabled}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: {bw}px solid {t.input_border}; background: {t.panel}; }}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; image: url({i['check']}); }}
QSlider::groove:horizontal {{ height: 4px; background: {t.border}; }}
QSlider::sub-page:horizontal {{ background: {t.accent}; }}
QSlider::sub-page:horizontal:disabled {{ background: {t.disabled}; }}
QSlider::handle:horizontal {{ width: 12px; margin: -5px 0; background: {t.panel}; border: {bw}px solid {t.accent}; }}
QSlider::handle:horizontal:disabled {{ border-color: {t.disabled}; }}

QTableWidget {{ background: {t.panel}; border: none; gridline-color: {t.header}; font-size: 12px;
    selection-background-color: {t.accent_soft}; alternate-background-color: {t.alt_row};
    outline: none; }}
QHeaderView::section {{ background: {t.header}; border: none; border-bottom: {bw}px solid {t.border};
    border-right: 1px solid {t.border}; padding: 5px 8px; font-size: 11px; font-weight: 600; color: {t.text_dim}; }}
QProgressBar {{ border: none; background: {t.border}; max-height: 6px; text-align: center; font-size: 1px; }}
QProgressBar::chunk {{ background: {t.accent}; }}

QMenu {{ background: {t.panel}; border: {bw}px solid {t.border}; padding: 4px 0; }}
QMenu::item {{ padding: 5px 22px 5px 14px; font-size: 12px; color: {t.text}; }}
QMenu::item:selected {{ background: {t.accent_soft}; }}
QMenu::item:checked {{ color: {t.accent}; font-weight: 600; }}
QMenu::item:disabled {{ color: {t.disabled}; }}
QMenu::separator {{ height: 1px; background: {t.border}; margin: 4px 0; }}
QStatusBar {{ background: {t.panel}; border-top: {bw}px solid {t.border}; }}
QStatusBar QLabel {{ font-size: 11px; color: {t.text_dim}; padding: 0 6px; }}
QSplitter::handle {{ background: {t.bg}; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: {t.bg}; width: 10px; height: 10px; border: none; }}
QScrollBar::handle {{ background: {t.input_border}; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QTextBrowser {{ background: {t.panel}; border: {bw}px solid {t.border}; font-size: 12px; }}
"""


def resolve(name: str | None) -> Theme:
    """A theme by name; 'Match system' (or unknown) follows the OS light/dark setting."""
    if name in THEMES:
        return THEMES[name]
    dark = False
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication
        dark = QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except Exception:  # noqa: BLE001 -- older Qt without scheme info: light
        pass
    return THEMES["Dark" if dark else DEFAULT_THEME]


def load_fonts() -> None:
    from PySide6.QtGui import QFontDatabase
    for f in HERE.glob("Saira*.ttf"):
        QFontDatabase.addApplicationFont(str(f))


def apply(app, theme: Theme) -> None:
    from PySide6.QtGui import QColor, QPalette
    app.setStyle("Fusion")
    pal = QPalette()
    for role, colour in ((QPalette.Window, theme.bg), (QPalette.Base, theme.panel), (QPalette.Text, theme.text),
                         (QPalette.WindowText, theme.text), (QPalette.Button, theme.panel),
                         (QPalette.ButtonText, theme.text), (QPalette.Highlight, theme.accent),
                         (QPalette.HighlightedText, theme.text), (QPalette.ToolTipBase, theme.panel),
                         (QPalette.ToolTipText, theme.text), (QPalette.PlaceholderText, theme.disabled),
                         (QPalette.AlternateBase, theme.alt_row)):
        pal.setColor(role, QColor(colour))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet(theme))
