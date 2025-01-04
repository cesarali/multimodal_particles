from pathlib import Path

base_dir = Path(__file__).resolve().parent
project_dir = (base_dir / "..").resolve()
data_dir = project_dir / "data"
test_resources_dir = project_dir / "tests" / "resources"
results_dir = project_dir / 'results'
config_dir = project_dir /'configs_files'