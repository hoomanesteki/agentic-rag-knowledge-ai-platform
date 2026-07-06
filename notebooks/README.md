# Notebooks

Runnable walkthroughs of the platform, offline on the synthetic data (no keys, no Docker).

- [01-data-architecture.ipynb](01-data-architecture.ipynb): the data layer step by step. Build the
  medallion, see gold come out clean and typed, watch PII get masked between bronze and gold, and
  pull a governed number from the semantic layer with a chart.
- [02-evaluation.ipynb](02-evaluation.ipynb): the quality stack. Run the offline gate, watch it
  **block a simulated regression**, and see drift (PSI) flag a distribution shift, with charts.
- [03-continuous-training.ipynb](03-continuous-training.ipynb): the CT loop, distinct from CI/CD.
  Run the trigger and promotion **policy** offline (no keys): what fires a cycle, what clears the
  bar to be promoted, and why CT only ever *proposes* a change for a human to approve.

## Running them

```bash
uv sync --extra notebook        # adds matplotlib and pandas (kept out of the base install)
```

Then open the notebook in your editor (VS Code's notebook support, or a Jupyter you already have)
and run the cells top to bottom. Each cell has a short note above it on what it shows and why.
