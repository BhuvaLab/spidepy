# Releasing spidepy to PyPI

Releases are built and published by `.github/workflows/release.yaml` using PyPI *trusted
publishing* (OIDC): no API token is stored anywhere.

## One-time setup (needs the project owner's PyPI account)

1. On <https://pypi.org/manage/account/publishing/> add a **pending publisher**:
   project `spidepy`, owner `BhuvaLab`, repository `spidepy`, workflow `release.yaml`,
   environment `pypi`.
2. Do the same on <https://test.pypi.org/manage/account/publishing/> with environment `testpypi`.
3. The GitHub environments `pypi` (one required reviewer) and `testpypi` already exist.

## Dry run on TestPyPI

Run the `release` workflow manually (Actions -> release -> Run workflow). It builds, checks,
installs the wheel in a clean environment and publishes to TestPyPI. Then check from a clean
environment:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ spidepy
python -c "import spidepy; print(spidepy.__version__)"
```

A version can be uploaded to PyPI/TestPyPI only once, so bump `__version__` before every upload.

## Release

1. Update `__version__` in `src/spidepy/__init__.py` and `CHANGELOG.md`; merge to `main` with CI green.
2. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
3. The workflow checks that the tag matches `__version__`, builds, and publishes to PyPI.
4. Afterwards: create a GitHub release from the tag; the PyPI/pepy badges in the README start to work.
