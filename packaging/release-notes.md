The first pre-release of **OpenRAW**: turn JPEGs into editable raw DNGs (linear or Bayer mosaic) for more grading headroom in any raw editor. Now with a desktop app.

![OpenRAW](https://raw.githubusercontent.com/Swarce/OpenRAW/main/assets/screenshots/app-light.png)

### Downloads

| File | For |
|---|---|
| `OpenRAW-*-windows-x64.zip` | Windows 10/11, 64-bit — unzip, run `OpenRAW.exe` |
| `OpenRAW-*-macos-arm64.zip` | macOS 11+ on Apple silicon — unzip, move `OpenRAW.app` to Applications |
| `OpenRAW-*-linux-x64.tar.gz` | Linux x64, glibc 2.35+ (Ubuntu 22.04 and newer) — extract, run `OpenRAW/OpenRAW` |
| `openraw-*.whl` / `.tar.gz` | Python package (command line; add `[gui]` for the app) |

Everything is included: no Python or other installs needed. The builds aren't code-signed yet, so Windows SmartScreen and macOS Gatekeeper ask once on first launch — see [First launch](https://github.com/Swarce/OpenRAW/blob/main/docs/app.md#first-launch).

### Highlights

- **Desktop app**: drag-and-drop queue, live progress, before/after compare against the DNG as a raw editor renders it, themes including dark and high contrast (orange accent).
- **InvISP reconstruction** with OpenRAW's FiveK + RAISE model `r2-e19` (recommended), the earlier `e14`, or upstream's Nikon model; scored against real Sony NEX-7 raws in [docs/invisp.md](https://github.com/Swarce/OpenRAW/blob/main/docs/invisp.md).
- **Bayer (CFA) DNGs**, ~3x smaller, demosaiced by your editor — now with optional camera-style **anti-aliasing**.

Full list in the [changelog](https://github.com/Swarce/OpenRAW/blob/main/CHANGELOG.md).

**Model weights** are trained on MIT-Adobe FiveK and RAISE and are for **non-commercial research use**; the code is licensed CC BY-NC-SA 4.0 (non-commercial, share-alike). See [NOTICE.md](https://github.com/Swarce/OpenRAW/blob/main/NOTICE.md).

This is an alpha: please report problems with the version info from the app's *About* box or `openraw --version`.
