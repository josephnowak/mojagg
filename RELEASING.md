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

## Cutting a release

You can cut a release in either of two ways:

### Option A: Create a GitHub Release (Recommended)
1. In the GitHub repository, go to **Releases** → **Draft a new release**.
2. Click **Choose a tag**:
   - Type the version tag, e.g. `v0.1.0`.
   - Select **Create new tag: v0.1.0 on publish**.
   - Target: `main`.
3. Fill in the release title (e.g. `v0.1.0`) and click **Generate release notes**.
4. Click **Publish release**.

The `release.yml` workflow will automatically trigger, build the wheels and sdist, test them, publish to PyPI, and attach the assets to the GitHub Release.

### Option B: The "Run workflow" button
1. Go to **Actions → Release → Run workflow**.
2. Type the version without leading `v` (e.g. `0.1.0`) → **Run**.

The workflow will:
- validate the version and refuse if the tag exists,
- bump `pyproject.toml`, commit, and tag `v<version>`,
- build wheel (`manylinux_x86_64`) + sdist,
- smoke-test the built wheel,
- publish to **PyPI** via trusted publishing,
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
