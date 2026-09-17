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
hot paths for release-to-release regression tracking and is the suite used by
`.github/workflows/codspeed.yml`.
