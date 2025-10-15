# Authorship identification of electronic texts

This project implements the classifiers used in the paper
["Authorship identification of electronic texts"](doi.org/10.1109/ACCESS.2021.3098192)
published in IEEE Access in 2021.

### installation

`pyproject.toml` file lists all the dependencies for this project.

Use `pip install -e .[dev]` and
`pip install -e .[torch-gpu] --index-url https://download.pytorch.org/whl/cu124`
if installing using pip manually in a virtualenv/conda env.

Use `uv sync` if using [uv](https://docs.astral.sh/uv/) for simpler
management of deps.

### citation

1. M. Khonji, Y. Iraqi and L. Mekouar, "Authorship Identification of Electronic Texts," in IEEE Access, vol. 9, pp. 101124-101146, 2021, doi: 10.1109/ACCESS.2021.3098192
