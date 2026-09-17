# Benchmark report

The latest public comparison is generated manually on the selected AWS host.
After `docs/benchmarks/latest/index.html` is copied into the repository, a
documentation build can embed it as a static page:

<iframe
  src="latest/index.html"
  title="Mojagg benchmark report"
  style="width: 100%; min-height: 1200px; border: 0; border-radius: 12px;">
</iframe>

The report is standalone and does not require a CDN. Documentation CI only
needs to build and deploy this tracked file; it should never rerun the AWS
benchmark. The report includes the detected device profile, including the EC2
instance identity when it is available, so the hardware context travels with
the comparison.
