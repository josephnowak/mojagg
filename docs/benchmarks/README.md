# Manual benchmark publication

Run the public profile on the publication machine:

```bash
python benchmarks/public_benchmark.py --profile public --output docs/benchmarks/latest/index.html
```

The repository's Windows/WSL wrapper exposes the equivalent local command as
`scripts/mojagg.ps1 bench-public`.

The command writes both `index.html` and `results.json`. Upload or commit
both files under `docs/benchmarks/latest/`. The future documentation site can
serve `index.html` directly or embed it from `docs/benchmarks/index.md`.

Use a custom suite when the hardware or audience needs a different matrix:

```python
from benchmarks.public_benchmark import Public, run_benchmark

suite = Public()
suite.reduction_functions = ["nansum", "nanmean", "nanvar"]
suite.groupby_tests[0].num_groups = 4096
suite.repeats = 7
run_benchmark(suite, "docs/benchmarks/latest/index.html")
```

The HTML contains a visible device profile with the AWS identity, instance
type, instance ID, region, availability zone, CPU, memory, operating system,
runtime, and thread environment. The same profile is stored in `results.json`
so the published result remains interpretable after the AWS instance is gone.

The benchmark queries EC2 IMDSv2 automatically. The EC2 `Name` tag appears
when EC2 Instance Metadata Tags are enabled; instance type and instance ID do
not depend on that tag. Use `--device-name` when the published report needs a
manual label.
