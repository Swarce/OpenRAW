# PyInstaller spec for the OpenRAW desktop app.
#
#   pip install ".[gui]" pyinstaller            # + torch (CPU) for the InvISP models
#   pyinstaller packaging/openraw.spec
#
# Produces dist/OpenRAW/ (Windows, Linux) or dist/OpenRAW.app (macOS), with
# the recommended InvISP model and upstream's Nikon model bundled. Smoke test
# a build with:  dist/OpenRAW/OpenRAW --selftest <scratch folder>
# (QT_QPA_PLATFORM=offscreen on a machine without a display).

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 -- SPECPATH is set by PyInstaller
sys.path.insert(0, str(ROOT))
from openraw import __version__  # noqa: E402

datas = collect_data_files("openraw")  # fonts, logo, licences
datas += [(str(ROOT / "pretrained" / f), "pretrained")
          for f in ("openraw-fivek-raise-e14-best.pth", "nikon.pth")]
datas += [(str(ROOT / "examples" / "test_source.jpg"), "examples"),  # for --selftest
          (str(ROOT / "LICENSE"), "."), (str(ROOT / "NOTICE.md"), ".")]

hiddenimports = collect_submodules("imagecodecs")  # codecs are imported lazily
hiddenimports += collect_submodules("openraw")

excludes = ["tkinter", "matplotlib", "torchvision", "torchaudio", "IPython", "jupyter", "pytest",
            "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml", "PySide6.QtQuick",
            "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization"]
if os.environ.get("OPENRAW_BUILD_NO_TORCH"):  # classical-only build (small; for testing the packaging)
    excludes.append("torch")

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "openraw_app.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

icon = str(ROOT / "packaging" / ("openraw.icns" if sys.platform == "darwin" else "openraw.ico"))
exe = EXE(  # noqa: F821
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="OpenRAW",
    console=False,
    icon=icon,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="OpenRAW", upx=False)  # noqa: F821

if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        name="OpenRAW.app",
        icon=icon,
        bundle_identifier="io.github.swarce.openraw",
        version=__version__,
        info_plist={
            "CFBundleShortVersionString": __version__,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "CFBundleDocumentTypes": [{"CFBundleTypeName": "JPEG image", "CFBundleTypeRole": "Viewer",
                                       "LSItemContentTypes": ["public.jpeg"]}],
        },
    )
