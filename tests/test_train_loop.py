"""
train.py's real training loop: parallel data loading, full-state checkpoints,
exact resume, and resume from weights-only checkpoints. Uses a 1-block InvISP
so it fits small machines (the full 8-block net needs ~4 GB for one CPU step);
everything else is train.py's own code. Skipped without torch.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rawpy")
pytest.importorskip("colour_demosaicing")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "data"))
from .cfa_dng import make_cfa_dng  # noqa: E402


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    import fivek_prepare as fp
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

    monkeypatch.chdir(ROOT)
    import train
    from openraw.third_party.invisp.model.model import InvISPNet
    from openraw.third_party.invisp.utils.JPEG import DiffJPEG
    monkeypatch.setattr(train, "InvISPNet", lambda **kw: InvISPNet(channel_in=3, channel_out=3, block_num=1))
    monkeypatch.setattr(train, "DiffJPEG", DiffJPEG(differentiable=True, quality=90), raising=False)
    out = str(tmp_path / "exps") + "/"

    def run(**over):
        kw = dict(task="t", data_path=str(data) + "/", batch_size=1, debug_mode=False, gamma=True, camera=cams,
                  rgb_weight=1, out_path=out, resume=False, loss="L1", lr=1e-4, aug=True, workers=1,
                  epochs=1, start_epoch=None, device="cpu", eval_every=0, eval_images=40, eval_crop=512, time_limit_hours=0,
                  gpus=1, blocks=1)
        kw.update(over)
        train.launch(SimpleNamespace(**kw))
        return train._load(out + "t/checkpoint/latest_state.pth", "cpu")

    return run, Path(out) / "t" / "checkpoint"


def test_train_with_workers_then_resume_exactly(trainer):
    run, ckpt = trainer
    st = run(epochs=1, workers=1)  # 2 train images -> 2 steps/epoch
    assert (st["epoch"], st["step"], st["scheduler"]["last_epoch"]) == (0, 2, 1)
    assert (ckpt / "latest.pth").exists()  # weights-only file stays usable by `openraw --invisp`
    st = run(epochs=2, resume=True)
    assert (st["epoch"], st["step"], st["scheduler"]["last_epoch"]) == (1, 4, 2)
    assert any("exp_avg" in v for v in st["optimizer"]["state"].values())  # Adam moments saved
    assert not list(ckpt.glob("*.tmp"))  # atomic saves leave no temp files


def test_resume_from_weights_only_checkpoint_with_start_epoch(trainer):
    """Runs started before full-state checkpoints only have latest.pth."""
    run, ckpt = trainer
    run(epochs=1)
    (ckpt / "latest_state.pth").unlink()
    st = run(epochs=3, resume=True, start_epoch=2)
    # continued at epoch 2 with the LR schedule fast-forwarded and the step count estimated
    assert (st["epoch"], st["step"], st["scheduler"]["last_epoch"]) == (2, 6, 3)


def test_gradient_checkpointing_is_exact():
    """Same loss and gradients for every parameter with and without
    checkpointing. Networks are built independently from one seed: upstream's
    InvBlock keeps its 1x1 conv behind a lambda capturing `self`, so a
    copy.deepcopy'd model would still route gradients to the ORIGINAL's conv
    parameters -- which once made checkpointing look like it lost 24 of them."""
    import torch.nn.functional as F
    import train
    from openraw.third_party.invisp.model.model import InvISPNet
    from openraw.third_party.invisp.utils.JPEG import DiffJPEG

    def build(ckpt):
        torch.manual_seed(0)
        net = InvISPNet(channel_in=3, channel_out=3, block_num=3)
        if ckpt:
            train._enable_checkpointing(net)
        return net
    a, b = build(False), build(True)
    jpeg = DiffJPEG(differentiable=True, quality=90)
    torch.manual_seed(1)
    x, t = torch.rand(1, 3, 48, 48), torch.rand(1, 3, 48, 48)

    def step(net):
        rgb = torch.clamp(net(x), 0, 1)
        loss = F.l1_loss(rgb, t) + F.l1_loss(net(jpeg(rgb), rev=True), x)
        loss.backward()
        return loss.item()
    assert step(a) == step(b)
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert pa.grad is not None and pb.grad is not None, n
        assert torch.equal(pa.grad, pb.grad), n


def test_training_runs_with_checkpointing(trainer):
    run, _ = trainer
    st = run(epochs=1, checkpointing="on")
    assert st["step"] == 2


def test_lr_schedule_scales_with_epochs(trainer):
    """LR drops land at the same fraction of training as upstream's 50/80 of 300."""
    run, _ = trainer
    st = run(epochs=6)
    assert sorted(st["scheduler"]["milestones"]) == [1, 2]  # round(6*50/300), round(6*80/300)
    assert st["scheduler"]["last_epoch"] == 6



def test_evaluation_writes_csv_and_keeps_best_across_resume(trainer):
    run, ckpt = trainer
    st = run(epochs=2, eval_every=1)
    rows = (ckpt.parent / "eval.csv").read_text().strip().splitlines()
    assert rows[0] == "epoch,step,raw_psnr,rgb_psnr" and len(rows) == 3  # header + 2 evaluations
    psnrs = [float(r.split(",")[2]) for r in rows[1:]]
    assert all(5 < p < 99 for p in psnrs)  # real measurements, not placeholders
    assert (ckpt / "best.pth").exists()
    assert st["best_raw_psnr"] == pytest.approx(max(psnrs), abs=1e-3)  # CSV keeps 4 decimals
    st2 = run(epochs=3, eval_every=1, resume=True)  # best survives the resume
    assert st2["best_raw_psnr"] >= st["best_raw_psnr"]


def test_eval_crop_matches_full_frame_crop(tmp_path):
    """The evaluation crop must be exactly the centre of the full frame."""
    import fivek_prepare as fp
    from dataset.FiveK_dataset import FiveKDatasetTest
    data = tmp_path / "data"
    raw = data / "fivek" / "raw" / "Testco_T1"; raw.mkdir(parents=True)
    meta = data / "fivek" / "_metadata"; meta.mkdir(parents=True)
    make_cfa_dng(str(raw / "a3.dng"), "GRBG", h=320, w=432)
    for sp, f in {"train": "training.json", "val": "validation.json", "test": "testing.json"}.items():
        (meta / f).write_text(json.dumps({"a3": {"urls": {"dng": "x", "tiff16": {}}, "camera": {"make": "Testco", "model": "T1"}}}
                                         if sp == "test" else {}))
    cams = fp.prepare_cameras(["Testco T1"], str(data) + "/", workers=1, log=lambda *_: None, use_available=True)
    ds = FiveKDatasetTest(SimpleNamespace(debug_mode=False, data_path=str(data) + "/", camera=cams, gamma=True))
    full = ds[0]
    ds.eval_crop = 128
    c = ds[0]
    H, W = full["input_raw"].shape[1:]
    y, x = (H - 128) // 4 * 2, (W - 128) // 4 * 2
    assert c["input_raw"].shape[1:] == (128, 128)
    assert torch.allclose(c["input_raw"], full["input_raw"][:, y:y + 128, x:x + 128], atol=1e-6)
    assert torch.equal(c["target_rgb"], full["target_rgb"][:, y:y + 128, x:x + 128])



def test_time_limit_stops_between_epochs_and_resume_continues(trainer):
    """A session time budget (e.g. Kaggle's 12 h) must stop the run cleanly
    after a completed, saved epoch -- and --resume must pick up from there."""
    run, _ = trainer
    st = run(epochs=5, time_limit_hours=1e-6)  # any real epoch exceeds this budget
    assert st["epoch"] == 0 and st["step"] == 2  # stopped after the first epoch, saved
    st = run(epochs=5, resume=True, time_limit_hours=1e-6)
    assert st["epoch"] == 1 and st["step"] == 4  # one more epoch per "session"



# ------------------------------------------------------------------ multi-GPU
def _tiny_setup():
    from openraw.third_party.invisp.model.model import InvISPNet
    from openraw.third_party.invisp.utils.JPEG import DiffJPEG
    torch.manual_seed(0)
    net = InvISPNet(channel_in=3, channel_out=3, block_num=2)
    jpeg = DiffJPEG(differentiable=True, quality=90)
    g = torch.Generator().manual_seed(1)
    data = [(torch.rand(1, 3, 32, 32, generator=g), torch.rand(1, 3, 32, 32, generator=g)) for _ in range(2)]
    return net, jpeg, data


def _loss(net, jpeg, x, t):
    import torch.nn.functional as F
    rgb = torch.clamp(net(x), 0, 1)
    return F.l1_loss(rgb, t) + F.l1_loss(net(jpeg(rgb), rev=True), x)


def _avg_worker(rank, world, port, out):
    """One 'GPU': gradients on its own sample, then train._average_gradients."""
    import torch.distributed as dist
    import train
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        net, jpeg, data = _tiny_setup()
        _loss(net, jpeg, *data[rank]).backward()
        train._average_gradients(list(net.parameters()), world)
        torch.save([p.grad.clone() for p in net.parameters()], f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def test_multi_gpu_gradient_averaging_equals_batch_training(tmp_path):
    """Two processes, one sample each, gradients averaged == one process
    training on both samples' mean loss -- i.e. exactly batch size 2."""
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]
    out = str(tmp_path / "grads")
    mp.spawn(_avg_worker, args=(2, port, out), nprocs=2, join=True)
    g0, g1 = torch.load(out + ".0"), torch.load(out + ".1")
    net, jpeg, data = _tiny_setup()
    ((_loss(net, jpeg, *data[0]) + _loss(net, jpeg, *data[1])) / 2).backward()
    ref = [p.grad for p in net.parameters()]
    for a, b, r in zip(g0, g1, ref):
        assert torch.equal(a, b)                      # every process takes the identical step
        assert torch.allclose(a, r, rtol=1e-5, atol=1e-7)  # ...and it's the batch-2 gradient


def test_two_process_training_stops_together_and_resumes_on_one(trainer):
    """--gpus 2 (CPU processes here; NCCL on GPUs): an epoch is split between the
    processes, the time limit stops BOTH (a lone process would hang the other),
    the checkpoint resumes in single-GPU mode, and vice versa."""
    run, ckpt = trainer
    st = run(epochs=5, gpus=2, eval_every=1, time_limit_hours=1e-6)
    assert (st["epoch"], st["step"]) == (0, 1)  # 2 train images / 2 processes = 1 step per epoch
    assert (ckpt.parent / "eval.csv").read_text().count("\n") == 2  # header + one evaluation, by rank 0 only
    st = run(epochs=2, resume=True)             # continue on one process
    assert (st["epoch"], st["step"]) == (1, 3)
    st = run(epochs=3, resume=True, gpus=2)      # and back to two
    assert (st["epoch"], st["step"]) == (2, 4)


def test_ema_math_and_warmup():
    import train
    net = torch.nn.Linear(3, 2)
    with torch.no_grad():
        net.weight.zero_(); net.bias.zero_()
    ema = train._EMA(net, decay=0.9)
    with torch.no_grad():
        net.weight.fill_(1.0)
    ema.update()  # warm-up: effective decay min(0.9, 2/11)
    d = 2 / 11
    assert torch.allclose(ema.state_dict()["weight"], torch.full((2, 3), 1 - d))
    for _ in range(200):
        ema.update()
    assert torch.allclose(ema.state_dict()["weight"], torch.ones(2, 3), atol=1e-6)  # converges to the weights
    assert ema.state_dict()["weight"] is not net.weight  # a separate copy, not an alias


def test_ema_is_saved_evaluated_and_resumed(trainer):
    run, ckpt = trainer
    st = run(epochs=2, eval_every=1, ema=0.9)
    assert st["ema_updates"] == 4  # 2 steps x 2 epochs
    latest = torch.load(ckpt / "latest.pth", weights_only=True)
    best = torch.load(ckpt / "best.pth", weights_only=True)
    # latest.pth / best.pth are the AVERAGED weights, not the live ones
    k = next(k for k, v in st["net"].items() if v.is_floating_point() and v.numel() > 1)
    assert torch.equal(latest[k], st["ema"][k])
    assert not torch.equal(latest[k], st["net"][k])
    assert set(best) == set(st["net"])  # loadable by `openraw --invisp-checkpoint`
    st2 = run(epochs=3, eval_every=1, ema=0.9, resume=True)
    assert st2["ema_updates"] == 6  # the average continued, not restarted
    rows = (ckpt.parent / "eval.csv").read_text().strip().splitlines()
    assert len(rows) == 4


def test_ema_starts_from_current_weights_when_resuming_an_older_run(trainer):
    """A run saved before --ema existed (no 'ema' in latest_state.pth) resumes fine."""
    run, ckpt = trainer
    st = run(epochs=1)  # EMA off: like an older run
    assert "ema" not in st
    st2 = run(epochs=2, ema=0.9, resume=True)
    assert st2["ema_updates"] == 2


def test_ema_off_saves_live_weights(trainer):
    run, ckpt = trainer
    st = run(epochs=1, ema=0)
    latest = torch.load(ckpt / "latest.pth", weights_only=True)
    assert all(torch.equal(latest[k], v) for k, v in st["net"].items())
    assert "ema" not in st
