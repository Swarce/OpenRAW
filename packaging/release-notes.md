A new pre-release of **OpenRAW**: turn JPEGs into editable raw DNGs (linear or Bayer mosaic) for more grading headroom in any raw editor, from the command line or the desktop app.

![OpenRAW](https://raw.githubusercontent.com/Swarce/OpenRAW/main/assets/screenshots/app-light.png)

### Downloads

| File | For |
|---|---|
| `OpenRAW-*-windows-x64.zip` | Windows 10/11, 64-bit — unzip, run `OpenRAW.exe` |
| `OpenRAW-*-macos-arm64.zip` | macOS 11+ on Apple silicon — unzip, move `OpenRAW.app` to Applications |
| `OpenRAW-*-linux-x64.tar.gz` | Linux x64, glibc 2.35+ (Ubuntu 22.04 and newer) — extract, run `OpenRAW/OpenRAW` |
| `openraw-*.whl` / `.tar.gz` | Python package (command line; add `[gui]` for the app) |

Everything is included: no Python or other installs needed. The builds aren't code-signed yet, so Windows SmartScreen and macOS Gatekeeper ask once on first launch — see [First launch](https://github.com/Swarce/OpenRAW/blob/main/docs/app.md#first-launch).

### What's new

- **A better model, now the default:** `r2-e19`, from a second FiveK + RAISE training run. On real Sony NEX-7 RAW+JPEG pairs it gets closer to the camera's actual raw than the previous model (40.45 vs 39.42 dB, gain-matched) and slightly ahead of upstream InvISP's Nikon model (40.32 dB). If you used the previous default, the app switches you over; the old model is still in the list as *InvISP · previous (e14)*. Details and side-by-side crops: [docs/invisp.md](https://github.com/Swarce/OpenRAW/blob/main/docs/invisp.md).
- **New licence: CC BY-NC-SA 4.0** (non-commercial, share-alike), matching the non-commercial terms the model weights already had. Earlier releases stay MIT. This lets OpenRAW include, with credit, code from the research it builds on; see [NOTICE.md](https://github.com/Swarce/OpenRAW/blob/main/NOTICE.md).
- **Credits in the app:** the *About* box names the papers OpenRAW builds on.

### Also in the app

- Drag-and-drop queue, live progress, before/after compare against the DNG as a raw editor renders it, themes including dark and high contrast (orange accent).
- Bayer (CFA) DNGs, ~3x smaller, demosaiced by your editor, with optional camera-style anti-aliasing.

Full list in the [changelog](https://github.com/Swarce/OpenRAW/blob/main/CHANGELOG.md).

**Licence:** code CC BY-NC-SA 4.0; model weights trained on MIT-Adobe FiveK and RAISE, for **non-commercial research use**. See [NOTICE.md](https://github.com/Swarce/OpenRAW/blob/main/NOTICE.md).

This is an alpha: please report problems with the version info from the app's *About* box or `openraw --version`.
