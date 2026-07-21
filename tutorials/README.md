# HomOPT Tutorials

Tutorials are small, runnable explanations of the active HomOPT workflow. They
use the public package APIs and save derived figures below
`results/figs/tutorials/`; they do not create a second experiment runner,
reference-solver path, or record format.

Start Jupyter from the repository root:

```bash
jupyter lab tutorials/
```

The notebooks assume an editable installation of HomOPT and the dependencies
listed in `pyproject.toml`. Each notebook states any extra solver requirement
before the first solver call.

## Available Tutorials

- [Hom-PGD tutorials](hom_pgd/README.md)

## Design Rules

- Notebook configuration is explicit in the first executable cell.
- Reference solutions use the same shared reference infrastructure as scripts.
- Figures are rebuilt from the notebook state and saved as PDFs.
- Tutorials illustrate one concept at a time; comparison studies remain in
  `scripts/`.
