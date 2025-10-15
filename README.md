# Authorship identification of electronic texts

This project implements the classifiers used in the paper
["Authorship identification of electronic texts"](doi.org/10.1109/ACCESS.2021.3098192)
published in IEEE Access in 2021.

### dataset

A copy of the dataset is committed to the repo but here are the steps to
recreate it.

Download [PAN12 Author Identification](https://doi.org/10.5281/zenodo.3713273)
dataset and extract the 2012 train and test corpus and filter the
"12CtrainXn.txt" files alone in `dataset/train` where X is the author ID and n
is the sample number.

Also, copy the "12Ctestn.txt" files to `dataset/test` folder.

### installation

`pyproject.toml` file lists all the dependencies for this project.

Use `pip install -e .[dev]` and
`pip install -e .[torch-gpu] --index-url https://download.pytorch.org/whl/cu124`
if installing using pip manually in a virtualenv/conda env.

Use `uv sync` if using [uv](https://docs.astral.sh/uv/) for simpler
management of deps.

### citation

1. M. Khonji, Y. Iraqi and L. Mekouar, "Authorship Identification of Electronic Texts," in IEEE Access, vol. 9, pp. 101124-101146, 2021, doi: 10.1109/ACCESS.2021.3098192
