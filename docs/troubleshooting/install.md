# Install Troubleshooting

## ImportError: No module named 'nibabel'

You are trying to use a visualization or MRI feature without installing the required extra.

```bash
pip install "qortex[mri]"
```

The error message always names the missing module and which extra provides it.

## nibabel version conflict

If your environment already has nibabel (e.g., from nilearn), you may have a version conflict.

Check the version:

```bash
python -c "import nibabel; print(nibabel.__version__)"
```

Qortex requires nibabel ≥ 5.0. If you have an older version, upgrade:

```bash
pip install "nibabel>=5.0"
```

## Python version error

Qortex requires Python 3.10 or later. Check your version:

```bash
python --version
```

If you need to maintain a Python 3.9 environment, create a separate conda environment:

```bash
conda create -n qortex_env python=3.11
conda activate qortex_env
pip install qortex
```

## pip install fails with "could not build wheels"

Some extras (dipy, MNE) have C extensions. If the build fails:

```bash
# Install build tools
pip install --upgrade pip setuptools wheel

# Then retry
pip install "qortex[dwi]"
```

On Linux, you may need system headers:

```bash
sudo apt-get install python3-dev build-essential
```

## Extras installed but still getting ImportError

Python may be loading the wrong environment. Verify which Python is being used:

```bash
which python
python -c "import sys; print(sys.executable)"
```

If Qortex was installed in a different environment, activate the correct one or reinstall.
