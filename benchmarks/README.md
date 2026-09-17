# Benchmark runners

`public_benchmark.py` is the manual, publication-oriented comparison. Run it
on the selected AWS host with the installed mojagg wheel or source checkout;
it writes a standalone HTML dashboard and JSON result file. Its `Quick`,
`Public`, and `Stress` suite classes can be customized by changing their
public case and function lists.

At startup the runner records a device profile in both outputs. On EC2 it
queries IMDSv2 for the instance ID, instance type, AMI, region, availability
zone, hostname, lifecycle, and the EC2 `Name` tag when instance metadata tags
are enabled. It also records the CPU model and counts, process CPU affinity,
total memory, operating system, Python and NumPy versions, and relevant thread
environment variables. The report prominently displays this profile and keeps
the raw JSON available for inspection.

The identity uses `--device-name` when supplied, then
`MOJAGG_BENCHMARK_DEVICE_NAME`, the EC2 `Name` tag, the instance type, and
the hostname. The EC2 `Name` tag is optional: enabling instance metadata tags
is required for it, while the instance ID and type remain available without
that tag. A publication run can therefore use an explicit stable label:

```bash
python benchmarks/public_benchmark.py \
  --profile public \
  --device-name "aws-c7i-public-run" \
  --output docs/benchmarks/latest/index.html
```

`codspeed/` is intentionally separate. It contains only moderate deterministic
inputs, is the suite used by `.github/workflows/codspeed.yml`, and runs on
every trusted push and pull request through CodSpeed's CPU simulation
instrument.

It covers each public family with one benchmark per operation and workload:

- `test_hot_paths.py` — the original representative paths, kept unchanged so
  their CodSpeed history stays continuous.
- `test_reductions.py` — every reduction on the contiguous, strided and
  full-collapse layouts, plus the float32 and int64 kernel instantiations.
- `test_grouped.py` — every grouped reduction at cache-resident (256) and
  out-of-cache (4096) cardinality, covering the int32 and int64 label kernels.
- `test_moving.py` — trailing windows with a short and a long window, and the
  exponentially weighted kernels, unary and pairwise.
- `test_matrix.py` — static, trailing and exponentially weighted pairwise
  matrices.
- `test_fill_selection.py` — `ffill`/`bfill` along the contiguous and strided
  layouts, and the non-streaming quantile selection path.

Shared inputs live in `codspeed/conftest.py`. Case names are part of the
CodSpeed benchmark identity: renaming a case or a test starts a new history,
so prefer adding a case over editing an existing one. Inputs stay in the low
megabytes because instrumented execution is far slower than native and larger
arrays lengthen every pull request without sharpening the signal; the broad
size/dtype/NaN/cardinality matrix belongs to `public_benchmark.py`.

```bash
pixi run pytest benchmarks/codspeed -q   # execute the suite once, no measurement
pixi run bench-codspeed                  # local wall-time measurement (minutes)
codspeed run --mode simulation -- pixi run pytest benchmarks/codspeed --codspeed -q
```

The last command is the one CI runs: the CPU simulation instrument only
records under the `codspeed` CLI or the GitHub action.
