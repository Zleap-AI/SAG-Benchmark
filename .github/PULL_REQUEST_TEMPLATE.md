## Summary

<!-- What problem does this change solve? What behavior changes? -->

## Scope

- [ ] Focused change; unrelated files and generated artifacts are excluded.
- [ ] Public documentation or CLI help was updated when behavior changed.
- [ ] No credentials, private paths, private infrastructure details, or local outputs are included.

## Validation

Commands run:

```text
# e.g. uv run pytest
```

External services or tests that were unavailable:

<!-- State the limitation and why it does not invalidate the change. -->

## Reproducibility

<!-- For benchmark changes, list dataset, source configuration, model/dimension,
     strategy parameters, and any expected metric impact. -->


## External method changes

<!-- Complete this section when external/ or the Judge artifact flow changes. -->

- [ ] Not applicable, or the affected method(s) are named below.
- [ ] The dataset, source/run ID, Python version, and relevant `uv.lock` revision are recorded.
- [ ] Native output and canonical prediction paths are listed.
- [ ] Logs are minimal and scrubbed of credentials, private URLs, and dataset content.
- [ ] The method's isolated `uv sync --frozen` and compile/smoke checks pass.
- [ ] `UPSTREAM.md` is updated when vendored source or local patches change.
