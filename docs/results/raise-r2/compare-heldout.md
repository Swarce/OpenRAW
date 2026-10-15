# InvISP checkpoints on 40 held-out test images

512 px centre crops, the images train.py's evaluation uses. raw PSNR: JPEG -> raw (OpenRAW's direction; picks best.pth). rgb PSNR: raw -> JPEG.

| Model | raw PSNR | rgb PSNR | Canon_EOS_5D (20) | NIKON_D700 (13) | RAISE_Nikon_D7000 (6) | RAISE_Nikon_D90 (1) |
|---|---|---|---|---|---|---|
| canon.pth | 25.77 dB | 18.40 dB | 24.83 | 26.64 | 27.12 | 25.45 |
| nikon.pth | 37.48 dB | 34.36 dB | 38.25 | 37.94 | 33.36 | 41.00 |
| openraw-fivek-e35-best.pth | 38.32 dB | 35.09 dB | 39.02 | 39.12 | 34.01 | 39.78 |
| openraw-fivek-e46-latest.pth | 38.32 dB | 34.79 dB | 39.69 | 37.59 | 35.51 | 37.31 |
| openraw-fivek-e53-best.pth | 38.55 dB | 35.47 dB | 39.45 | 39.13 | 34.12 | 39.54 |
| openraw-fivek-raise-e14-best.pth | 39.01 dB | 35.75 dB | 39.94 | 38.47 | 37.00 | 39.65 |
| **best.pth** | 39.11 dB | 35.69 dB | 40.14 | 38.75 | 36.44 | 39.32 |
| latest.pth | 39.04 dB | 35.80 dB | 40.08 | 38.54 | 36.62 | 39.31 |

raw PSNR per camera; the number of test images in brackets.
