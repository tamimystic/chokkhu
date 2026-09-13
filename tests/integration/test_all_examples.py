"""Integration tests verifying that all runnable example scripts execute successfully."""

import importlib.util
from pathlib import Path


def _run_example(example_filename: str):
    root_dir = Path(__file__).resolve().parent.parent.parent
    example_path = root_dir / "examples" / example_filename
    assert example_path.exists(), f"Example file {example_filename} does not exist"

    spec = importlib.util.spec_from_file_location("example_module", str(example_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main"), f"{example_filename} missing main() function"
    module.main()


def test_example_01_automl_tabular():
    _run_example("01_automl_tabular_pipeline.py")


def test_example_02_frontier_llms_reasoning():
    _run_example("02_frontier_llms_reasoning_agents.py")


def test_example_03_computer_vision_and_3d():
    _run_example("03_computer_vision_and_3d.py")


def test_example_04_timeseries_and_state():
    _run_example("04_timeseries_forecasting_and_state_filtering.py")


def test_example_05_causal_and_survival():
    _run_example("05_causal_inference_and_survival_analysis.py")


def test_example_06_diffusion_and_audio():
    _run_example("06_generative_diffusion_and_audio.py")


def test_example_07_sciml_and_koopman():
    _run_example("07_sciml_pinns_and_koopman_dynamics.py")


def test_example_08_benchmarks():
    _run_example("08_benchmarks_zero_heavy_dependencies.py")
