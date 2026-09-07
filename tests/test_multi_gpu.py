import argparse
import subprocess
import sys
from pathlib import Path
import pytest
import torch
import yaml

from pod_sim2real.training.train import parse_gpu_ids, run_configs_multi_gpu


def test_parse_gpu_ids():
    assert parse_gpu_ids(None) == []
    assert parse_gpu_ids("") == []
    assert parse_gpu_ids("0") == ["0"]
    assert parse_gpu_ids("0,1,2,3") == ["0", "1", "2", "3"]
    assert parse_gpu_ids("0-3") == ["0", "1", "2", "3"]
    assert parse_gpu_ids("1-3, 5") == ["1", "2", "3", "5"]
    assert parse_gpu_ids("0, 1, 1, 2") == ["0", "1", "2"]

    auto_gpus = parse_gpu_ids("auto")
    assert isinstance(auto_gpus, list)
    assert len(auto_gpus) >= 1

    all_gpus = parse_gpu_ids("all")
    assert isinstance(all_gpus, list)
    assert len(all_gpus) >= 1


def test_cli_accepts_gpus_flag():
    import os
    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    res = subprocess.run(
        [sys.executable, "-m", "pod_sim2real.training.train", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0
    assert "--gpus" in res.stdout


def test_multi_gpu_dispatcher_execution(tmp_path, monkeypatch):
    """Test dispatcher loop with mocked subprocesses to ensure correct queue and status handling."""
    cfg_dir = tmp_path / "yamls"
    cfg_dir.mkdir()
    out_dir = tmp_path / "runs"

    configs = []
    for name in ("cfg_a", "cfg_b", "cfg_c"):
        cfg_file = cfg_dir / f"{name}.yaml"
        cfg_file.write_text(yaml.dump({
            "model": {"name": "fno"},
            "run_name": name,
            "output_dir": str(out_dir / name),
            "data": {"use_tensor_cache": False},
        }), encoding="utf-8")
        configs.append(cfg_file)

    class DummyProc:
        def __init__(self, cmd, stdout, stderr, env):
            self.cmd = cmd
            self.stdout = stdout
            self.env = env
            self.poll_count = 0
            self.stdout.write("Mock child process output\n")
            self.stdout.flush()

        def poll(self):
            self.poll_count += 1
            if self.poll_count >= 2:
                # Write mock results.json
                for i, arg in enumerate(self.cmd):
                    if arg == "--config":
                        cfg_p = Path(self.cmd[i + 1])
                        break
                r_dir = out_dir / cfg_p.stem
                r_dir.mkdir(parents=True, exist_ok=True)
                (r_dir / "results.json").write_text('{"real_test_mse": 0.001, "real_test_metrics": {"rel_l2": 0.02}}')
                return 0
            return None

        def terminate(self):
            pass

        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", DummyProc)

    args = argparse.Namespace(
        config_dir=cfg_dir,
        model=None,
        data_root=None,
        output_dir=out_dir,
        best_checkpoints_dir=tmp_path / "best",
        resolution=None,
        stride=None,
        epochs=None,
        batch_size=None,
        device="cpu",
        seed=None,
        num_workers=None,
        resume=None,
        test_mode=None,
        prefix_frames=None,
    )

    results = run_configs_multi_gpu(configs, ["0", "1"], args)
    assert len(results) == 3
    for name in ("cfg_a", "cfg_b", "cfg_c"):
        assert name in results
        assert results[name]["status"] == "SUCCESS"
        assert results[name]["results"]["real_test_mse"] == 0.001
