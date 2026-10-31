# Interactive Query beyond Linear Utility Model

Core implementations of **Anchor Tracking (AT)** and **Range Shrink (RS)** for
interactive preference queries with a weighted distance function. The algorithms
seek a user's favorite tuple by learning an unknown expected point and attribute
weights from pairwise comparisons.

For a tuple `p`, expected point `e`, and nonnegative weights `u` summing to one,
the implementation evaluates the weighted squared distance
`sum(u * (p - e) ** 2)`. Smaller distances indicate stronger preferences.

The included runner simulates users: it draws an expected point uniformly within
the selected data's coordinate-wise bounds and weights from
`Dirichlet(1, ..., 1)`. These hidden parameters answer comparisons and evaluate
the returned tuple.

## Repository contents

| Path | Purpose |
| --- | --- |
| `run.py` | Command-line entry point for AT and RS. |
| `AT.py` | Anchor tracking, question selection, and alternating updates. |
| `at_feasibility.py` | Feedback constraints, feasibility recovery, and LP checks for AT. |
| `RS.py` | Sample-based range shrinking and candidate selection. |
| `utils.py` | Loading, normalization, simulated feedback, evaluation, logging, and JSON export. |
| `datasets/` | Included numeric datasets. |
| `generator/` | Optional C++ generators; `generate.cpp` is the CMake target. |

## Installation

Use **Python 3.12** in a virtual or Conda environment. With that environment's
Python interpreter, install the core dependencies:

```sh
python -m pip install numpy==1.26.4 scipy==1.13.1 torch==2.6.0
```

These versions were used for the local runtime checks with Python 3.12.7 on
Windows. The runner selects CUDA when a compatible GPU and CUDA-enabled PyTorch
installation are available; otherwise it uses the CPU. Set
`CUDA_VISIBLE_DEVICES=-1` before launching Python to force CPU execution.

Run all commands below from the repository root, using that environment's
Python. The included Python sources and datasets are sufficient for these runs.

## Quick start

Run one small trial with each algorithm:

```sh
python run.py --alg AT --data_size 64 --rounds 1 --output-dir experiments/smoke-at
python run.py --alg RS --data_size 64 --rounds 1 --N 256 --output-dir experiments/smoke-rs
```

These commands use the first 64 tuples of `datasets/4d_100000.txt`. The smaller
RS sample count keeps this a quick execution check. Each command prints results
and saves JSON files. A specified output directory must not already exist; use
a new name when rerunning, or omit `--output-dir` to create a fresh timestamped
directory automatically.

For larger experiments, select the input, prefix size, and number of trials:

```sh
python run.py --alg AT --data datasets/car.txt --data_size 10000 --rounds 100
python run.py --alg RS --data datasets/car.txt --data_size 10000 --rounds 100 --N 5000
```

For higher precision and detailed AT diagnostics, or to disable JSON export:

```sh
python run.py --alg AT --data_size 64 --rounds 1 --dtype float64 --debug
python run.py --alg RS --data_size 64 --rounds 1 --N 256 --no-save-json
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--alg` | `AT` | `AT` or `RS`. |
| `--data` | `./datasets/4d_100000.txt` | Input text dataset. |
| `--data_size` | `10000` | Number of tuples taken from the beginning of the normalized dataset, up to the available count. |
| `--rounds` | `100` | Independent simulated-user trials; not a question budget. |
| `--dtype` | `float32` | `float32` or `float64`. |
| `--N` | `5000` | Initial RS sample count; unused by AT. |
| `--eps` | `0.0` | Approximation tolerance; see evaluation notes below. |
| `--save-json` / `--no-save-json` | enabled | Enable or disable result and route JSON export. |
| `--output-dir` | automatic | New directory for JSON output. |
| `--debug` / `--no-debug` | disabled | Extra AT state and recovery diagnostics; unused by RS. |

Use `python run.py --help` for the argument list. Runtime and memory use increase
with dataset size, dimension, and sample count. RS computes sample-to-tuple
distances, so large `--N` and `--data_size` values can require substantial memory.
Start with the small examples before increasing them.

## Datasets

| File | Tuples | Dimensions |
| --- | ---: | ---: |
| `datasets/2d_100000.txt` | 100,000 | 2 |
| `datasets/3d_100000.txt` | 100,000 | 3 |
| `datasets/4d_100000.txt` | 100,000 | 4 |
| `datasets/5d_100000.txt` | 100,000 | 5 |
| `datasets/car.txt` | 69,047 | 4 |
| `datasets/flight.txt` | 439,643 | 5 |
| `datasets/island.txt` | 63,383 | 2 |
| `datasets/nba.txt` | 21,960 | 4 |

An input file starts with `n d`, followed by `n` rows of `d` numeric attributes:

```text
3 2
0.0 1.0
0.5 0.4
1.0 0.0
```

The loader min-max normalizes each column using the **entire input file**, then
selects the first `--data_size` tuples. Constant columns become zero. Output
indices are zero-based indices within this selected prefix.

### Optional data generation

The included datasets can be used directly. To generate additional numeric data,
compile `generator/generate.cpp` with a C++17 compiler, for example GCC:

```sh
g++ -O2 -std=c++17 generator/generate.cpp -o generator/generate
./generator/generate 4 0 1 1 E 10000 datasets/synthetic_4d.txt
```

On Windows, use `generator/generate.exe` as the executable path. The positional
arguments are numeric dimensions, categorical dimensions, class count, Zipfian
theta, distribution, tuple count, and output filename. Use zero categorical
dimensions for the Python runner. Distributions are `E` (independent), `C`
(correlated), and `A` (anti-correlated). Choose a fresh output filename to preserve
existing datasets. The same source can also be built with
`generator/CMakeLists.txt`, CMake, and a suitable build tool.

## Outputs and evaluation

By default, each invocation creates:

```text
experiments/<AT-or-RS>/run_<timestamp>/
  results.json
  routes/
    eu_route_0.json       # AT, one file per trial
    # or rs_route_0.json  # RS, one file per trial
```

`--output-dir` replaces the invocation's output directory. File logs go to
`logs/experiment_<timestamp>.log`, including when `--no-save-json` is used.

`results.json` contains the arguments and a `results` list with one entry per
trial. Fields include `queries` (actual questions asked), `time` (elapsed seconds,
including monitoring), `success`, `rank` (zero-based rank under the true distance),
`dis_bias` (relative squared-distance error with a small denominator offset), and
`termination` (stopping reason). AT also reports `algorithm_valid` and
`final_anchor_valid`.

Route files retain the true simulated parameters, initial state, comparison
history, and one `E-U Route` / `Time` entry after each actual question. AT debug
mode additionally retains intermediate state and recovery diagnostics.

When interpreting results:

- With `--eps 0`, success requires the returned item to attain the minimum true
  weighted distance. An AT trial explicitly marked invalid is unsuccessful even
  if the returned index happens to be optimal. With ties, several items can be
  successful even though their reported ranks differ.
- AT evaluates squared distance against a factor of `1 + eps`. RS uses
  `(1 + eps)^2` on squared distance, corresponding to its distance tolerance; RS
  also uses this tolerance during candidate intersection. AT's runner does not
  change its stopping rule with `--eps`.
- RS uses sampled parameter pairs and can finish without returning the true
  favorite. Inspect `success`, `rank`, and `termination` together. AT's LP
  decisions also operate under finite-precision tolerances; inspect validity
  fields as well.
- Each run draws random users and initial states. The CLI has no global seed
  option, and separate AT and RS invocations do not automatically share users.

## License

The repository includes the [Apache License 2.0](LICENSE).
