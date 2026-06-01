import importlib

from packaging.version import Version

requirements = {
    "torch": "2.0.0",
    "torchvision": "0.15.0",
    "torch_geometric": "2.3.0",
    "torch_scatter": "2.1.0",
    "torch_sparse": "0.6.17",
    "transformers": "4.35.0",
    "tokenizers": "0.14.0",
    "sklearn": "1.3.0",
    "xgboost": "1.7.0",
    "lightgbm": "4.0.0",
    "pandas": "2.0.0",
    "numpy": "1.24.0",
    "pyarrow": "12.0.0",
    "scipy": "1.11.0",
    "shap": "0.43.0",
    "lime": "0.2.0.1",
    "matplotlib": "3.7.0",
    "seaborn": "0.12.0",
    "plotly": "5.15.0",
    "networkx": "3.1.0",
    "torch_cluster": "1.6.0",
    "nltk": "3.8.0",
    "spacy": "3.6.0",
    "tqdm": "4.65.0",
    "yaml": "6.0",          # pyyaml imports as yaml
    "optuna": "3.2.0",
    "wandb": "0.15.0",
    "requests": "2.31.0",
    "bs4": "4.12.0",        # beautifulsoup4 imports as bs4
}

print("=" * 70)
print("MGNN-IPV Dependency Check")
print("=" * 70)

missing = []
outdated = []
ok = []

for module_name, required_version in requirements.items():
    try:
        module = importlib.import_module(module_name)
        installed_version = getattr(module, "__version__", "unknown")

        if installed_version != "unknown":
            if Version(installed_version) >= Version(required_version):
                ok.append((module_name, installed_version))
                status = "OK"
            else:
                outdated.append((module_name, installed_version, required_version))
                status = "OUTDATED"
        else:
            ok.append((module_name, installed_version))
            status = "UNKNOWN VERSION"

        print(f"[{status:<15}] {module_name:<20} installed={installed_version} required>={required_version}")

    except ImportError:
        missing.append((module_name, required_version))
        print(f"[MISSING         ] {module_name:<20} required>={required_version}")

print("\n" + "=" * 70)

print(f"\nInstalled OK: {len(ok)}")
print(f"Outdated:     {len(outdated)}")
print(f"Missing:      {len(missing)}")

if missing:
    print("\nMissing packages:")
    for name, ver in missing:
        print(f"  pip install {name}>={ver}")

if outdated:
    print("\nOutdated packages:")
    for name, installed, required in outdated:
        print(f"  pip install --upgrade {name}>={required}")

print("\nDone.")
