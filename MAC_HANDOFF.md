# Mac handoff

Moving development from the Windows PC (MSYS2/UCRT64) to a Mac. Read
[`docs/working-context/PROJECT_NOTES.md`](docs/working-context/PROJECT_NOTES.md) first: its hard
constraints, verification loop and backlog all still apply. This file only covers what changes on a
Mac.

## Setup from zero

```bash
xcode-select --install                      # Apple Clang (CI's macOS job uses it)
brew install git gh cmake ninja libomp python@3.12
pip3 install clang-format==21.1.2 clang-tidy   # pinned so local and CI formatting agree
gh auth login                               # as Lukey-7 (the repo owner)
git clone https://github.com/Lukey-7/AxiomQuant.git && cd AxiomQuant
```

Eigen and SQLite are vendored in `third_party/`, so there is nothing else to install.

## Build and test

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DOpenMP_ROOT="$(brew --prefix libomp)"
cmake --build build --parallel
ctest --test-dir build --output-on-failure      # 64 tests
./build/bin/axiomquant --data sample_data --no-db --export-dir out
```

- **OpenMP:** Apple Clang ships without it. With `libomp` and `OpenMP_ROOT` set, the Monte Carlo
  runs in parallel. Without them, `find_package(OpenMP)` fails quietly and the build is single
  threaded, which is exactly what CI's *macOS (Apple Clang)* job does. Results are identical either
  way because seeding is per path.
- **Binaries** have no `.exe`: `./build/bin/quant_tests`, `./build/bin/axiomquant`.
- **Apple Silicon:** leave `AXIOM_NATIVE` off for anything you compare against CI numbers.
- The sanitizer build from CI works locally too: add
  `-DCMAKE_BUILD_TYPE=Debug -DAXIOM_ENABLE_OPENMP=OFF "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined"`.

## What changes from the Windows notes

| PROJECT_NOTES.md says | On the Mac |
|---|---|
| MSYS2 tools at `C:\msys64\ucrt64\bin` on PATH | Homebrew + Xcode command line tools (above) |
| `./build/bin/quant_tests.exe` | `./build/bin/quant_tests` |
| Files on disk are CRLF (`autocrlf`) | Set `git config --global core.autocrlf input`; files are LF |
| `gh` keeps reverting to VD0WQ, which can't push | Only sign in as Lukey-7 on the Mac and the problem goes away |
| Python 3.14 with NumPy available | `pip3 install numpy` (plus `cvxpy` to run `check_optimizer.py` locally) |

Update the *Hard constraints* and *Machine gotchas* sections in `PROJECT_NOTES.md` once the Mac
setup is confirmed working.

## Secrets

None are stored locally. `KAGGLE_API_TOKEN` (set) and `TIINGO_API_KEY` (optional, not set) are
GitHub Actions secrets on the repo. Yahoo data needs no key.

## State at the move (10 Oct 2026)

- `main` is at `a34473d` *ci: update checkout and upload-artifact to v7* and green on all jobs.
- `improvements` (the push-first branch) has nothing that isn't on `main`.
- Nothing from the backlog is started. Suggested order, from the notes: **1** Reality Check / SPA
  test, **8** faster correlated-GBM Monte Carlo, **7** Python bindings.

## First steps on the Mac

1. Build, run the 64 tests and the CLI showcase on `sample_data`, and confirm the output matches
   the latest CI run of the *macOS (Apple Clang)* job.
2. Run `clang-format --dry-run --Werror` over the sources to check the pinned formatter works.
3. Update `PROJECT_NOTES.md` for the Mac, push it to `improvements`, watch CI, then merge to `main`.
4. Start backlog item 1.
