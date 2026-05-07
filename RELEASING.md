# Releasing to PyPI

This repository is configured for GitHub-to-PyPI publishing with PyPI Trusted
Publishing.

## One-time setup on PyPI

1. Open the `brighteyes-flim` project on PyPI.
2. Go to `Manage -> Publishing`.
3. Add a GitHub trusted publisher with:
   - Owner: `VicidominiLab`
   - Repository: `BrightEyes-Flim`
   - Workflow file: `.github/workflows/release.yml`
   - Environment: `pypi`

If this is the first-ever PyPI release for the project, you can also create the
project using a pending trusted publisher from the same workflow file.

## Release flow

1. Update the version in `setup.cfg`.
2. Commit the release changes to `main`.
3. Create and push a version tag:

```bash
git tag v0.9.4
git push origin main --tags
```

The GitHub Actions workflow will:

- build the sdist and wheel
- run `twine check`
- publish the artifacts to PyPI

## Local preflight check

Before tagging, run:

```bash
python3 -m pip install -U build twine
rm -rf dist build src/*.egg-info
python3 -m build
python3 -m twine check dist/*
```

## Notes

- Publishing is triggered by tags matching `v*`.
- The GitHub environment is named `pypi`; you can protect it with required
  reviewers if you want manual approval before each release.
- If you want a TestPyPI lane later, add a second workflow or a second publish
  job with a separate trusted publisher configuration.
