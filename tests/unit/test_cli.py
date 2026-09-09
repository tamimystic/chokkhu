from __future__ import annotations

import os
import tempfile
import pandas as pd
import pytest
from chokkhu.cli import main


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_cli_help_and_empty():
    assert main([]) == 0


def test_cli_clean_and_pipeline():
    import shutil
    import matplotlib.pyplot as plt

    tmpdir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(tmpdir, "test.csv")
        out_path = os.path.join(tmpdir, "cleaned.csv")
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, None, 4.0, 5.0, 6.0],
                "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        df.to_csv(csv_path, index=False)

        # Test clean
        ret = main(["clean", "-d", csv_path, "-o", out_path])
        assert ret == 0
        assert os.path.exists(out_path)

        # Test pipeline
        pipe_save = os.path.join(tmpdir, "pipe.pkl")
        ret = main(["pipeline", "-d", csv_path, "-t", "target", "-s", pipe_save])
        assert ret == 0

        # Test train
        ret = main(
            ["train", "-d", out_path, "-t", "target", "-m", "logistic_regression"]
        )
        assert ret == 0

        # Test automl
        ret = main(["automl", "-d", out_path, "-t", "target", "--time-budget", "3"])
        assert ret == 0
    finally:
        plt.close("all")
        shutil.rmtree(tmpdir, ignore_errors=True)
