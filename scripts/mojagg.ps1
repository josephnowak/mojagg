[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "doctor",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$CommandArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$distro = if ($env:MOJAGG_WSL_DISTRO) { $env:MOJAGG_WSL_DISTRO } else { "Ubuntu" }
$repoWindows = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Convert-ToWslPath([string]$Path) {
    if ($Path -notmatch "^(?<drive>[A-Za-z]):\\(?<rest>.*)$") {
        throw "The repository must be on a Windows drive mounted by WSL: $Path"
    }

    return "/mnt/$($Matches.drive.ToLower())/$($Matches.rest.Replace('\', '/'))"
}

function Convert-ToBashLiteral([string]$Value) {
    $singleQuote = [string][char]39
    $doubleQuote = [string][char]34
    $replacement = $singleQuote + $doubleQuote + $singleQuote + $doubleQuote + $singleQuote
    $escaped = $Value.Replace($singleQuote, $replacement)
    return $singleQuote + $escaped + $singleQuote
}

function Show-Help {
    @"
Usage: scripts/mojagg.ps1 <command> [options]

Every command runs inside Ubuntu WSL. The default Pixi environment is py314;
choose another locked environment with --env py310, --env py311, --env py312,
or --env py313.

Commands:
  install       Install Pixi in WSL and resolve the locked project environment.
  lock          Regenerate pixi.lock from pixi.toml.
  doctor        Check WSL, Pixi, Mojo, Python, pytest, search tools, and binaries.
  files [text]  List files under python, src, and tests.
  search <expr> Search source text under python, src, and tests.
  build         Compile the native Mojo extension.
  test-mojo     Run the Mojo smoke tests.
  test-python   Build the extension and run Python parity tests.
  test          Build the extension, then run Mojo and Python tests.
  lint          Run the repository lint task.
  format        Format Python and Mojo sources through Pixi.
  bench-codspeed
                Build the extension and run the lightweight CodSpeed suite.
  bench-public  Build the extension and run the manual HTML comparison.
  bench-parallel-threshold
                Run the standalone Mojo serial/parallel threshold benchmark.
  bench-quantile-select
                Run the standalone Mojo sort vs multi-select quantile benchmark.
  bench-group-nanmin
                Compare grouped nanmin vectorization and unroll variants.
  bench-group-nanmin-cartesian
                Benchmark the grouped nanmin width/unroll Cartesian product.
  bench-ffill-cartesian
                Benchmark the forward-fill width/unroll Cartesian product.
  bench-nansum-cartesian
                Benchmark the nansum width/unroll Cartesian product.
  bench-allnan-cartesian
                Benchmark the allnan width/unroll Cartesian product.
  bench-move-mean-cartesian
                Benchmark the move_mean width/unroll Cartesian product.
  bench-tail-split-3gib
                Compare masked and split-tail mean loops on 3 GiB of input.
  bench-move-exp-nansum-cartesian
                Benchmark the move_exp_nansum width/unroll Cartesian product.
  bench-nancovmatrix-cartesian
                Benchmark the nancovmatrix width/unroll Cartesian product.
  clean-native  Remove generated native binaries under python/mojagg only.

Examples:
  scripts/mojagg.ps1 install
  scripts/mojagg.ps1 test
  scripts/mojagg.ps1 search "group_nansum"
  scripts/mojagg.ps1 files "native"
"@
}

$aliases = @{
    "help" = "help"
    "-h" = "help"
    "--help" = "help"
    "install" = "install"
    "lock" = "lock"
    "doctor" = "doctor"
    "files" = "files"
    "search" = "search"
    "build" = "build"
    "build-ext" = "build"
    "test-mojo" = "test-mojo"
    "test-python" = "test-python"
    "test" = "test"
    "lint" = "lint"
    "format" = "format"
    "bench-codspeed" = "bench-codspeed"
    "bench-public" = "bench-public"
    "bench-parallel-threshold" = "bench-parallel-threshold"
    "bench-quantile-select" = "bench-quantile-select"
    "bench-group-nanmin" = "bench-group-nanmin"
    "bench-group-nanmin-cartesian" = "bench-group-nanmin-cartesian"
    "bench-ffill-cartesian" = "bench-ffill-cartesian"
    "bench-nansum-cartesian" = "bench-nansum-cartesian"
    "bench-allnan-cartesian" = "bench-allnan-cartesian"
    "bench-move-mean-cartesian" = "bench-move-mean-cartesian"
    "bench-tail-split-3gib" = "bench-tail-split-3gib"
    "bench-move-exp-nansum-cartesian" = "bench-move-exp-nansum-cartesian"
    "bench-nancovmatrix-cartesian" = "bench-nancovmatrix-cartesian"
    "clean-native" = "clean-native"
}

if ($Command -eq "doctor" -and @($CommandArgs).Count -eq 1 -and @($CommandArgs)[0] -in @("-h", "--help")) {
    Show-Help
    exit 0
}

if (-not $aliases.ContainsKey($Command)) {
    Show-Help
    throw "Unknown command: $Command"
}

$command = $aliases[$Command]
if ($command -eq "help") {
    Show-Help
    exit 0
}

$environment = "default"
$forwardArgs = [System.Collections.Generic.List[string]]::new()
$rawArgs = @($CommandArgs)
for ($index = 0; $index -lt $rawArgs.Count; $index++) {
    $argument = $rawArgs[$index]
    if ($argument -eq "--env") {
        if ($index + 1 -ge $rawArgs.Count) {
            throw "--env requires a Pixi environment name"
        }
        $index++
        $environment = $rawArgs[$index]
        continue
    }
    if ($argument -like "--env=*") {
        $environment = $argument.Substring(7)
        continue
    }
    $forwardArgs.Add($argument)
}

if ($environment -notmatch "^[A-Za-z0-9_-]+$") {
    throw "Invalid Pixi environment name: $environment"
}

if (($command -eq "search" -or $command -eq "files") -and $forwardArgs.Count -gt 1) {
    throw "$command accepts at most one search or filename expression"
}
if ($command -eq "search" -and $forwardArgs.Count -eq 0) {
    throw "search requires an expression"
}

$repoWsl = Convert-ToWslPath $repoWindows
$repoLiteral = Convert-ToBashLiteral $repoWsl
$environmentLiteral = Convert-ToBashLiteral $environment
$argumentText = if ($forwardArgs.Count -eq 0) {
    "set --"
} else {
    "set -- " + (($forwardArgs | ForEach-Object { Convert-ToBashLiteral $_ }) -join " ")
}

$common = @'
set -euo pipefail
cd __MOJAGG_REPO__

ensure_pixi() {
    if [ -x "\$HOME/.pixi/bin/pixi" ]; then
        PIXI="\$HOME/.pixi/bin/pixi"
        return
    fi
    if ! command -v curl >/dev/null 2>&1; then
        printf '%s\n' 'curl is required in Ubuntu WSL to install Pixi.' >&2
        exit 127
    fi
    printf '%s\n' 'Pixi is missing in WSL; installing the official Pixi bootstrap.'
    curl -fsSL https://pixi.sh/install.sh | bash
    if [ ! -x "\$HOME/.pixi/bin/pixi" ]; then
        printf '%s\n' 'Pixi installation did not produce ~/.pixi/bin/pixi.' >&2
        exit 127
    fi
    PIXI="\$HOME/.pixi/bin/pixi"
}

run_pixi() {
    "\$PIXI" run -e __MOJAGG_ENV__ "\$@"
}
'@
$common = $common.Replace("__MOJAGG_REPO__", $repoLiteral)
$common = $common.Replace("__MOJAGG_ENV__", $environmentLiteral)

$body = switch ($command) {
    "install" {
        @'
ensure_pixi
printf '%s\n' 'Using WSL Pixi environment'
"\$PIXI" install --locked
if [ ! -x "\$HOME/.pixi/bin/rg" ] && ! command -v rg >/dev/null 2>&1; then
    "\$PIXI" global install ripgrep
fi
printf '%s\n' 'WSL project environment is ready.'
'@
        break
    }
    "lock" {
        @'
ensure_pixi
"\$PIXI" lock
'@
        break
    }
    "doctor" {
        @'
ensure_pixi
"\$PIXI" install --locked
printf '%s\n' 'WSL execution: confirmed'
printf '%s' 'Pixi: '
"\$PIXI" --version
printf '%s\n' 'Mojo:'
run_pixi mojo --version
printf '%s\n' 'Python:'
run_pixi python --version
printf '%s\n' 'pytest:'
run_pixi pytest --version
if [ -x "\$HOME/.pixi/bin/rg" ]; then
    printf '%s' 'Search: '
    "\$HOME/.pixi/bin/rg" --version | head -1
elif command -v rg >/dev/null 2>&1; then
    printf '%s' 'Search: '
    rg --version | head -1
else
    printf '%s\n' 'Search: ripgrep unavailable; the wrapper will use git grep.'
fi
printf '%s\n' 'Source roots: python src tests'
printf '%s' 'Generated native binaries: '
find python/mojagg -maxdepth 1 -type f \( -name 'nanfuncs_native*.so' -o -name 'nanfuncs_native*.pyd' -o -name 'nanfuncs_native*.dylib' -o -name 'groupby_native*.so' -o -name 'groupby_native*.pyd' -o -name 'groupby_native*.dylib' \) | wc -l
printf '%s\n' 'Working tree:'
git -c core.autocrlf=true status --short
'@
        break
    }
    "files" {
        @'
if [ -x "\$HOME/.pixi/bin/rg" ]; then
    if [ "\$#" -eq 0 ]; then
        "\$HOME/.pixi/bin/rg" --files python src tests
    else
        "\$HOME/.pixi/bin/rg" --files python src tests | "\$HOME/.pixi/bin/rg" --fixed-strings --ignore-case -- "\$1"
    fi
elif command -v rg >/dev/null 2>&1; then
    if [ "\$#" -eq 0 ]; then
        rg --files python src tests
    else
        rg --files python src tests | rg --fixed-strings --ignore-case -- "\$1"
    fi
else
    if [ "\$#" -eq 0 ]; then
        git ls-files --cached --others --exclude-standard -- python src tests
    else
        git ls-files --cached --others --exclude-standard -- python src tests | grep --fixed-strings --ignore-case -- "\$1"
    fi
fi
'@
        break
    }
    "search" {
        @'
if [ -x "\$HOME/.pixi/bin/rg" ]; then
    "\$HOME/.pixi/bin/rg" --hidden --glob '!**/__pycache__/**' --glob '!**/*.pyc' -- "\$1" python src tests
elif command -v rg >/dev/null 2>&1; then
    rg --hidden --glob '!**/__pycache__/**' --glob '!**/*.pyc' -- "\$1" python src tests
else
    git -c core.autocrlf=true grep --line-number --extended-regexp -- "\$1" -- python src tests
fi
'@
        break
    }
    "build" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi build-ext
'@
        break
    }
    "test-mojo" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi test-mojo
'@
        break
    }
    "test-python" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi build-ext
run_pixi test
'@
        break
    }
    "test" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi build-ext
run_pixi test-mojo
run_pixi test
'@
        break
    }
    "lint" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi lint
'@
        break
    }
    "format" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi format
'@
        break
    }
    "bench-codspeed" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi build-ext
run_pixi bench-codspeed "\$@"
'@
        break
    }
    "bench-public" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi build-ext
PYTHONPATH=python run_pixi python benchmarks/public_benchmark.py "\$@"
'@
        break
    }
    "bench-parallel-threshold" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi mojo run benchmarks/parallel_threshold.mojo "\$@"
'@
        break
    }
    "bench-quantile-select" {
        @'
ensure_pixi
"\$PIXI" install --locked
mkdir -p .tmp
run_pixi mojo build -O3 --mcpu x86-64-v3 -o .tmp/quantile_select benchmarks/quantile_select.mojo
.tmp/quantile_select "\$@"
'@
        break
    }
    "bench-group-nanmin" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi mojo run benchmarks/group_nanmin_variants.mojo "\$@"
'@
        break
    }
    "bench-group-nanmin-cartesian" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi mojo run benchmarks/group_nanmin_cartesian.mojo "\$@"
'@
        break
    }
    "bench-ffill-cartesian" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi mojo run benchmarks/ffill_cartesian.mojo "\$@"
'@
        break
    }
    "bench-nansum-cartesian" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi mojo run benchmarks/nansum_cartesian.mojo "\$@"
'@
        break
    }
    "bench-allnan-cartesian" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi mojo run benchmarks/allnan_cartesian.mojo "\$@"
'@
        break
    }
    "bench-move-mean-cartesian" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi mojo run benchmarks/move_mean_cartesian.mojo "\$@"
'@
        break
    }
    "bench-tail-split-3gib" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi mojo run benchmarks/tail_split_3gib.mojo "\$@"
'@
        break
    }
    "bench-move-exp-nansum-cartesian" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi mojo run benchmarks/move_exp_nansum_cartesian.mojo "\$@"
'@
        break
    }
    "bench-nancovmatrix-cartesian" {
        @'
ensure_pixi
"\$PIXI" install --locked
run_pixi mojo run benchmarks/nancovmatrix_cartesian.mojo "\$@"
'@
        break
    }
    "clean-native" {
        @'
printf '%s\n' 'Removing generated native binaries under python/mojagg:'
find python/mojagg -maxdepth 1 -type f \( -name 'nanfuncs_native*.so' -o -name 'nanfuncs_native*.pyd' -o -name 'nanfuncs_native*.dylib' -o -name 'groupby_native*.so' -o -name 'groupby_native*.pyd' -o -name 'groupby_native*.dylib' \) -print -delete
'@
        break
    }
}

$bash = $common + "`n" + $argumentText + "`n" + $body

& wsl.exe -d $distro -- bash -lc $bash
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    exit $exitCode
}
