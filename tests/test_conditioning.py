"""The optional additions to training: EXIF reading (openraw/camera_params.py),
per-camera adapters and EXIF conditioning (openraw/conditioning.py), the
edit-aware loss and finishing augmentation (openraw/train_losses.py), the
camera-settings backfill (data/camera_params.py), and their path through
train.py and the converter."""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "data"))

from openraw import camera_params as cp  # noqa: E402
from openraw import conditioning as C  # noqa: E402
from openraw import train_losses as TL  # noqa: E402
from openraw.third_party.invisp.model.model import InvISPNet  # noqa: E402

NAN = float("nan")


# ------------------------------------------------------------------ EXIF
def _jpeg_with_exif(make="NIKON CORPORATION", model="NIKON D700", exposure=(1, 200), fnum=(56, 10), iso=800,
                    focal=(35, 1)) -> bytes:
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational
    ex = Image.Exif()
    ex[271], ex[272] = make, model
    sub = ex.get_ifd(0x8769)
    sub[33434], sub[33437], sub[34855], sub[37386] = (IFDRational(*exposure), IFDRational(*fnum), iso,
                                                      IFDRational(*focal))
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (90, 120, 150)).save(buf, "JPEG", exif=ex.tobytes())
    return buf.getvalue()


def test_reads_settings_from_a_jpeg_and_its_tiff_block(tmp_path):
    data = _jpeg_with_exif()
    want = {"exposure_time": 0.005, "f_number": 5.6, "iso": 800.0, "focal_length": 35.0,
            "make": "NIKON CORPORATION", "model": "NIKON D700"}
    got = cp.from_bytes(data)
    assert {k: (round(v, 6) if isinstance(v, float) else v) for k, v in got.items()} == want
    p = tmp_path / "a.jpg"; p.write_bytes(data)
    assert cp.from_file(p)["iso"] == 800.0
    # the same TIFF structure on its own, as a raw file starts
    tiff = data[data.index(b"Exif\0\0") + 6:]
    assert cp.from_bytes(tiff)["exposure_time"] == pytest.approx(0.005)
    with pytest.raises(cp._Truncated):
        cp.from_bytes(tiff[:40])  # ends inside IFD0: the backfill fetches more
    # decode.load_jpeg's exif_fields give the same
    from openraw.decode import load_jpeg
    assert cp.from_exif_fields(load_jpeg(str(p)).exif_fields)["f_number"] == pytest.approx(5.6)


def test_missing_or_broken_exif_is_unknown_not_an_error(tmp_path):
    from PIL import Image
    p = tmp_path / "plain.jpg"
    Image.new("RGB", (8, 8)).save(p)
    assert cp.from_file(p) == {k: None for k in cp.PARAMS}
    (tmp_path / "junk.bin").write_bytes(b"\x00" * 100)
    assert cp.from_file(tmp_path / "junk.bin") == {k: None for k in cp.PARAMS}
    assert cp.from_file(tmp_path / "nope.jpg") == {k: None for k in cp.PARAMS}
    v = cp.as_vector({"iso": 100.0})
    assert v[2] == 100.0 and all(math.isnan(x) for x in v[:2] + v[3:])
    assert cp.from_bytes(_jpeg_with_exif(exposure=(0, 1)))["exposure_time"] is None  # zero = unknown


# ------------------------------------------------------------------ conditioning
CAMS = ["NIKON_D700", "Canon_EOS_5D", "Leica_M8", "RAISE_NIKON_D90", "NIKON_D70"]


@pytest.mark.parametrize("make,model,want", [
    ("NIKON CORPORATION", "NIKON D700", 0), ("Canon", "Canon EOS 5D", 1), ("Leica Camera AG", "M8", 2),
    ("NIKON CORPORATION", "NIKON D90", 3), ("NIKON", "NIKON D70", 4), ("SONY", "NEX-7", -1),
    ("Canon", "Canon EOS 5D Mark II", -1), (None, None, -1)])
def test_exif_make_model_picks_the_training_camera(make, model, want):
    assert C.match_camera(CAMS, make, model) == want


def _net(blocks=2, seed=0, **kw):
    torch.manual_seed(seed)
    net = InvISPNet(channel_in=3, channel_out=3, block_num=blocks)
    if kw:
        C.attach(net, **kw)
    return net


def _shake(net, seed=1):
    g = torch.Generator().manual_seed(seed)
    for p in C.conditioned_parameters(net):
        p.data.copy_(torch.randn(p.shape, generator=g) * 0.05)


P = torch.tensor([[1 / 200, 4.0, 200.0, 35.0], [NAN, NAN, NAN, NAN]])


def test_added_parts_start_as_no_change_and_stay_invertible():
    x = torch.rand(2, 3, 24, 24)
    base = _net()(x)
    net = _net(cameras=CAMS[:2], lora_rank=4, exif=True)
    C.set_context(net, [0, 1], P)
    assert torch.equal(net(x), base)  # freshly attached: identical outputs
    _shake(net)
    C.set_context(net, [0, 1], P)
    y = net(x)
    assert (y - base).abs().max() > 1e-3  # now they matter...
    assert (net(y, rev=True) - x).abs().max() < 1e-5  # ...and the network still inverts exactly
    C.set_context(net, [1], P[1:])  # a batch gives each image its own camera and settings
    assert torch.allclose(net(x[1:]), y[1:], atol=1e-6)
    C.set_context(net, [-1], None)  # unknown camera, no EXIF
    assert (net(net(x), rev=True) - x).abs().max() < 1e-5


def test_checkpoint_round_trip_and_camera_remap():
    net = _net(cameras=CAMS[:2], lora_rank=4, exif=True)
    _shake(net)
    sd = C.with_meta(net.state_dict(), net)
    assert sd[C.META_KEY]["cameras"] == CAMS[:2] and sd[C.META_KEY]["lora_rank"] == 4
    sd2, meta = C.split_meta(sd)
    other = _net(seed=5)
    C.attach_from_meta(other, meta)
    other.load_state_dict(sd2)
    x = torch.rand(1, 3, 16, 16)
    for n in (net, other):
        C.set_context(n, [1], P[:1])
    assert torch.equal(net(x), other(x))
    # a later run lists the cameras differently: each keeps its own adapter
    re = C.remap_cameras(sd2, CAMS[:2], ["X", CAMS[1]])
    assert torch.equal(re[f"{C.PREFIX}lora.1.0.a.weight"], sd2[f"{C.PREFIX}lora.1.0.a.weight"])
    assert not any(k.startswith(f"{C.PREFIX}lora.0.") for k in re)


def test_gradient_checkpointing_stays_exact_with_conditioning():
    import torch.nn.functional as F
    import train

    def build(ckpt):
        net = _net(blocks=2, cameras=CAMS[:2], lora_rank=2, exif=True)
        _shake(net)
        if ckpt:
            train._enable_checkpointing(net)
        return net
    a, b = build(False), build(True)
    torch.manual_seed(3)
    x, t = torch.rand(2, 3, 20, 20), torch.rand(2, 3, 20, 20)

    def step(net):
        C.set_context(net, [0, 1], P)
        y = net(x)
        loss = F.l1_loss(y, t) + F.l1_loss(net(y, rev=True), x)
        loss.backward()
        return loss.item()
    assert step(a) == step(b)
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is not None or pb.grad is not None:
            assert torch.allclose(pa.grad, pb.grad, atol=1e-7), n


def test_param_dropout_and_unknown_settings_use_the_learned_default():
    e = C.ParamEmbed(16)
    with torch.no_grad():
        for p in e.parameters():
            p.normal_()
        none = e(torch.full((1, 4), NAN))
        dropped = e(P[:1], dropout=1.0)  # every setting hidden
        assert torch.allclose(none, dropped)
        assert not torch.allclose(e(P[:1]), none)


# ------------------------------------------------------------------ losses
def test_edit_aware_loss():
    torch.manual_seed(0)
    raw = torch.rand(2, 3, 16, 16) * 0.8
    assert TL.edit_aware_loss(raw, raw.clone(), raw_gamma=True).item() == 0.0
    pred = (raw + 0.05 * torch.randn_like(raw)).requires_grad_()
    loss = TL.edit_aware_loss(pred, raw, raw_gamma=True)
    loss.backward()
    assert loss.item() > 0 and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    # one edit per image, the same for prediction and truth
    e = TL.sample_edits(2, "cpu")
    assert e["exposure"].shape == (2,) and e["wb"].shape == (2, 2)
    out = TL.render_edit(raw, e)
    assert out.shape == raw.shape and out.min() >= 0 and out.max() <= 1


def test_tone_curves_are_monotone_and_finishing_stays_in_range():
    x = torch.linspace(0, 1, 101).view(1, 1, 1, -1).expand(1, 3, 1, -1)
    for a in (-0.3, 0.0, 0.6):
        y = TL._contrast(x, torch.tensor([a]))
        assert (y[..., 1:] - y[..., :-1] >= -1e-6).all()
    rgb = torch.rand(3, 3, 8, 8)
    f = TL.apply_finish(rgb, TL.sample_finish(3, "cpu"))
    assert f.shape == rgb.shape and f.min() >= 0 and f.max() <= 1 and not torch.equal(f, rgb)


# ------------------------------------------------------------------ backfill + data loader
def test_backfill_writes_the_table_the_loader_reads(tmp_path):
    import camera_params as bf  # data/camera_params.py
    data = tmp_path / "data"
    (data / "Testco_T1" / "RAW").mkdir(parents=True)
    for n in ("a1", "a2"):
        np.savez(data / "Testco_T1" / "RAW" / f"{n}.npz", wb=np.ones(4))
    meta = data / "fivek" / "_metadata"; meta.mkdir(parents=True)
    for f in ("training.json", "validation.json", "testing.json"):
        (meta / f).write_text(json.dumps({"a1": {"urls": {"dng": "http://x/a1.dng"}, "camera": {}}} if f == "training.json" else {}))
    jpeg = _jpeg_with_exif()
    tiff = jpeg[jpeg.index(b"Exif\0\0") + 6:]
    asked = []

    def fetch(url, n):
        asked.append(n)
        return tiff[:n] if n and n < 40 else tiff
    monkey_steps = bf.STEPS
    try:
        bf.STEPS = (30, 0)  # first piece too short: must fetch more
        out = bf.backfill(data, fetch=fetch, log=lambda *_: None)
    finally:
        bf.STEPS = monkey_steps
    table = json.loads(out.read_text())
    assert table == {"Testco_T1/a1": [pytest.approx(0.005), pytest.approx(5.6), 800.0, 35.0]}
    assert asked == [30, 0]  # a2 has no URL: skipped
    assert bf.backfill(data, fetch=fetch, log=lambda *_: None) == out and asked == [30, 0]  # nothing left

    from dataset.base_dataset import BaseDataset
    ds = BaseDataset.__new__(BaseDataset)
    ds.data_path, ds.camera_params_file = str(data) + "/", None
    with np.load(data / "Testco_T1" / "RAW" / "a1.npz") as z:
        v = ds.camera_params(z, str(data / "Testco_T1" / "RAW" / "a1.npz"))
    assert v.dtype == np.float32 and v[2] == 800.0
    with np.load(data / "Testco_T1" / "RAW" / "a2.npz") as z:
        assert np.isnan(ds.camera_params(z, str(data / "Testco_T1" / "RAW" / "a2.npz"))).all()


# ------------------------------------------------------------------ train.py end to end
def _setup(tmp_path, monkeypatch):
    pytest.importorskip("rawpy")
    pytest.importorskip("colour_demosaicing")
    import fivek_prepare as fp
    from .cfa_dng import make_cfa_dng
    data = tmp_path / "data"
    raw = data / "fivek" / "raw" / "Testco_T1"; raw.mkdir(parents=True)
    meta = data / "fivek" / "_metadata"; meta.mkdir(parents=True)
    split = {"a1": "train", "a2": "train", "a3": "test"}
    for n in split:
        make_cfa_dng(str(raw / f"{n}.dng"), "GRBG", h=320, w=432)
    for sp, f in {"train": "training.json", "val": "validation.json", "test": "testing.json"}.items():
        (meta / f).write_text(json.dumps({n: {"urls": {"dng": "http://unused/x.dng", "tiff16": {}},
                                              "camera": {"make": "Testco", "model": "T1"}}
                                          for n, s in split.items() if s == sp}))
    cams = fp.prepare_cameras(["Testco T1"], str(data) + "/", workers=1, log=lambda *_: None, use_available=True)
    with np.load(data / cams[0] / "RAW" / "a1.npz") as z:
        assert "camera_params" in z.files  # stored when preparing (all unknown for this synthetic DNG)
    monkeypatch.chdir(ROOT)
    import train
    from openraw.third_party.invisp.utils.JPEG import DiffJPEG
    monkeypatch.setattr(train, "InvISPNet", lambda **kw: InvISPNet(channel_in=3, channel_out=3, block_num=1))
    monkeypatch.setattr(train, "DiffJPEG", DiffJPEG(differentiable=True, quality=90), raising=False)
    out = str(tmp_path / "exps") + "/"
    from types import SimpleNamespace

    def run(**over):
        kw = dict(task="t", data_path=str(data) + "/", batch_size=1, debug_mode=False, gamma=True, camera=cams,
                  rgb_weight=1, out_path=out, resume=False, loss="L1", lr=1e-4, aug=True, workers=0,
                  epochs=1, start_epoch=None, device="cpu", eval_every=0, eval_images=40, eval_crop=512,
                  time_limit_hours=0, gpus=1, blocks=1, edit_sigma=0.75, finish_prob=1.0, cam_dropout=0.2,
                  param_dropout=0.2)
        kw.update(over)
        train.launch(SimpleNamespace(**kw))
        return train._load(out + "t/checkpoint/latest_state.pth", "cpu")
    return run, Path(out) / "t" / "checkpoint", cams


def test_training_with_every_addition_saves_a_described_checkpoint(tmp_path, monkeypatch, capsys):
    run, ckpt, cams = _setup(tmp_path, monkeypatch)
    st = run(edit_loss=1.0, finish_aug=0.5, cam_lora=2, exif_cond=True)
    log = capsys.readouterr().out
    assert "edit_loss:" in log and "finish_loss:" in log and "per-camera adapters" in log
    assert st[C.META_KEY]["cameras"] == cams and st[C.META_KEY]["exif"]
    import train
    best = train._load(str(ckpt / "best.pth"), "cpu")
    assert best[C.META_KEY]["lora_rank"] == 2 and any(k.startswith(C.PREFIX) for k in best)
    st = run(epochs=2, resume=True, edit_loss=1.0, finish_aug=0.5, cam_lora=2, exif_cond=True)  # exact resume
    assert st["epoch"] == 1
    with pytest.raises(SystemExit, match="different settings"):  # resuming without them is refused, clearly
        run(epochs=3, resume=True)


def test_fine_tuning_only_the_added_parts_of_an_existing_model(tmp_path, monkeypatch):
    run, ckpt, _ = _setup(tmp_path, monkeypatch)
    run()  # a plain model, like e14
    import train
    plain = train._load(str(ckpt / "latest.pth"), "cpu")
    (ckpt / "latest_state.pth").unlink()  # INIT_FROM leaves only latest.pth
    st = run(resume=True, cam_lora=2, exif_cond=True, train_new_only=True, epochs=2, start_epoch=1)
    for k, v in plain.items():  # InvISP's own weights untouched
        assert torch.equal(st["net"][k], v), k
    assert any(not torch.equal(v, torch.zeros_like(v)) for k, v in st["net"].items()
               if k.startswith(C.PREFIX + "lora.") and ".b." in k)  # adapters learned something


def test_converter_uses_the_photos_camera_and_settings(tmp_path):
    from openraw import invisp_bridge as ib
    net = _net(blocks=8, cameras=CAMS[:2], lora_rank=2, exif=True)
    _shake(net)
    path = tmp_path / "cond.pth"
    torch.save(C.with_meta(net.state_dict(), net), path)
    loaded = ib._load_net("", "", "cpu", checkpoint=str(path))
    assert C.get(loaded).cameras == CAMS[:2]
    assert ib._condition(loaded, {"make": "Canon", "model": "Canon EOS 5D", "iso": 100.0}) == "adapter Canon_EOS_5D"
    assert ib._condition(loaded, {"make": "SONY", "model": "NEX-7"}) == "shared network"
    rgb = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    a = ib.reconstruct_pseudo_raw(rgb, checkpoint=str(path), camera_params={"make": "Canon", "model": "Canon EOS 5D"})
    b = ib.reconstruct_pseudo_raw(rgb, checkpoint=str(path), camera_params=None)
    assert a.shape == b.shape and not np.allclose(a, b)
    # plain checkpoints ignore camera_params
    plain = tmp_path / "plain.pth"
    torch.save(_net(blocks=8).state_dict(), plain)
    assert ib._condition(ib._load_net("", "", "cpu", checkpoint=str(plain)), {"model": "x"}) is None
