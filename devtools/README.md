# Release tools

`conda-recipe/meta.yaml` is the conda recipe for the
[mdtools](https://anaconda.org/mdtools) channel on anaconda.org. The Conda
workflow builds it from the committed tree and runs the test suite on the
built package.

## Releasing

1. Set the new version in all three places, in one pull request, and merge
   it:
   - `version` in `pyproject.toml`;
   - `__version__` in `src/openmm_opes/__init__.py`;
   - the expected version in `tests/test_packaging.py`.
2. Publish a GitHub Release from `main` with the tag `vX.Y.Z`:

   ```bash
   gh release create vX.Y.Z --target main --generate-notes
   ```

Publishing the release starts three workflows, each of which refuses a tag
that does not match the version in `pyproject.toml`:

| Workflow | Publishes |
|---|---|
| `publish.yml` | The package on PyPI, with the README's relative links pinned to the tag |
| `conda.yml` | The conda package on the mdtools channel |
| `docs.yml` | The documentation for the tag, which becomes `latest` |

Prereleases are not deployed. A failed upload fails its workflow, but check
[PyPI](https://pypi.org/project/openmm-opes/) and
[anaconda.org](https://anaconda.org/mdtools/openmm-opes) anyway.

## Rerunning a deployment

`conda.yml` and `docs.yml` can be run by hand from the Actions tab, with the
tag of an existing release, to rebuild and redeploy it. A manual run of
`conda.yml` without a tag builds and tests the current branch as a dry run,
with no upload. Pull requests that change the recipe or `conda.yml` get the
same dry run.

## Secrets

| Secret | Used by |
|---|---|
| `PYPI_TOKEN` | `publish.yml` |
| `ANACONDA_TOKEN` | `conda.yml`; an anaconda.org token allowed to upload to mdtools |
