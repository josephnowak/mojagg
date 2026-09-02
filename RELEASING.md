# Releasing mojagg

Releases are cut manually, on demand — never automatically on merge to main.

## One-time setup

### PyPI Trusted Publishing (OIDC) — do this once, no tokens needed

1. Go to <https://pypi.org> → your account → **Publishing** (or the project's **Settings → Publishing** once the project exists; for the very first release use "Add a new pending publisher").
2. Add a **GitHub** publisher with:
   - **Owner**: your GitHub user/org
   - **Repository**: `mojagg`
   - **Workflow name**: `release.yml`
   - **Environment**: `pypi`
3. Repeat on <https://test.pypi.org> for the sandbox.
4. In the GitHub repo: **Settings → Environments → New environment** named `pypi`. Optionally add protection rules (require review before publish).

That's it — no `PYPI_API_TOKEN` secret anywhere. The workflow exchanges a short-lived OIDC token.

### CodSpeed

1. Install the CodSpeed GitHub App on the repo: <https://github.com/apps/codspeed-hq>
2. Copy the repo's CodSpeed token into **Settings → Secrets → Actions → `CODSPEED_TOKEN`**.
3. Free for public/open-source repos.

## Cutting a release (the button)

1. Make sure `main` is green (CI + CodSpeed) and CHANGELOG-worthy changes use conventional commits.
2. Go to **Actions → Release → Run workflow**.
3. Type the version (e.g. `0.2.0`) → **Run**.

The workflow will:
- validate the version and refuse if the tag exists,
- bump `pyproject.toml`, commit, and tag `v<version>`,
- build wheels for each platform (linux x86_64, macOS arm64) + sdist,
- smoke-test the built wheel on its own platform,
- publish to **TestPyPI**, then **PyPI** via trusted publishing,
- create a GitHub Release with the wheels attached and auto-generated notes.

## Versioning

SemVer, strictly. Performance-sensitive downstream code pins against us.
- `MAJOR`: API/semantic breaks
- `MINOR`: new functions, backward-compatible
- `PATCH`: fixes and pure performance improvements

## Benchmarks per release

Before tagging a release, regenerate the publishable numbers on the pinned
benchmark host so the README table stays honest:

```bash
python benchmarks/full_matrix.py --out benchmarks/results/
```

Commit the results with the release. (A `benchmark-aws.yml` manual workflow to
do this on a dedicated instance is planned.)
