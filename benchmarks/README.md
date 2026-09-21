# Benchmark runners

`public_benchmark.py` is the manual, publication-oriented comparison. Run it
on the selected AWS host with the installed mojagg wheel or source checkout;
it writes a standalone HTML dashboard and JSON result file. Its `Quick`,
`Public`, and `Stress` suite classes can be customized by changing their
public case and function lists.

At startup the runner records a privacy-safe device profile in both outputs.
It records only non-identifying runtime characteristics such as the CPU model
and counts, process CPU affinity, total memory, operating system, Python and
NumPy versions, and relevant thread environment variables. Cloud instance
identity, hostname, region, AMI, and EC2 tags are never queried or stored.
The report prominently displays this profile.

The identity uses `--device-name` when supplied, then
`MOJAGG_BENCHMARK_DEVICE_NAME`, and otherwise a generic `benchmark host`
label. A publication run can therefore use an explicit stable label:

```bash
python benchmarks/public_benchmark.py \
  --profile public \
  --device-name "aws-c7i-public-run" \
  --output docs/benchmarks/latest/index.html
```

`codspeed/` is intentionally separate. It contains only moderate deterministic
hot paths for release-to-release regression tracking and is the suite used by
`.github/workflows/codspeed.yml`.

## Parallel threshold probe

`parallel_threshold.mojo` compares the same row-sum work in a serial loop and
the current Mojo 1.0 parallel primitive, `max.algorithm.parallelize`. Older
Mojo examples spell the second form as `parallel_range(rows)`; the benchmark
uses the API available in this checkout.

Run it through the project wrapper on Windows:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 bench-parallel-threshold
```

On a Linux checkout, the equivalent direct command is:

```bash
pixi run -e default mojo run benchmarks/parallel_threshold.mojo
```

The first cases vary `rows` with a long row, and the second vary `columns` with
16 rows. Compare `parallel_ns` with `serial_ns` only after confirming
`match=True`. The crossover in the second sweep is the evidence to use when
choosing `parallel_threshold`; repeat the run with the same CPU affinity and
thread settings as the production benchmark.

## Quantile selection probe

`quantile_select.mojo` compares the two ways `NanQuantileKernel` can answer
several quantiles over one core span: ordering everything with `sort`, or
selecting only the requested order statistics with a bisecting multi-kth
partition, the Mojo equivalent of `np.partition(arr, kth=unique_indices)`.
Both variants run over the same restored buffer, and the restore cost is
measured separately and subtracted.

Run it through the project wrapper on Windows:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/mojagg.ps1 bench-quantile-select
```

On a Linux checkout, the equivalent direct command is:

```bash
pixi run -e default mojo build -O3 --mcpu x86-64-v3 -o .tmp/quantile_select benchmarks/quantile_select.mojo
.tmp/quantile_select
```

Read `speedup = sort_ns / select_ns`, and only after confirming `match=True`.
The sweep varies the span length and the number of quantiles; the resulting
crossover is the evidence behind the `SELECT_*` tiers in
`src/mojagg/nanfuncs/nanquantile.mojo`. Selection wins while few order
statistics are requested on a long span and loses once their count approaches
`log2(length)`, where the single sort is already optimal.
