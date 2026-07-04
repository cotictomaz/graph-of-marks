# Running the Ablation Studies on the SLURM Cluster

This guide explains, end to end, how to take this repository from your
laptop to a queued, GPU-backed job on the university's SLURM cluster, and
how to change *what* an ablation run does without touching any code.

It assumes you have already read the cluster's own "SLURM Web Guide" (SSH
access, `install_rootless_docker.sh`, disk quotas, byobu, etc.) — this
document only covers what is specific to this project: `build/Dockerfile`,
`build/Dockerfile.rtx5090`, `train.sh`, `run_docker.sh`, `sbatch_train.sh`,
and the `slurm_configs/*.yaml` files.

The only code this setup runs is `src/gom/ablations/main.py`
(`python -m gom.ablations.main --config <file>`), which dispatches to the
three experiment types implemented in `src/gom/ablations/run_experiments.py`:
ablation grid experiments, VLM comparison, and prompting-strategy
experiments. Nothing else in the package (preprocessing CLI, other scripts)
is wired into these Docker/SLURM scripts.

---

## 1. The moving parts, at a glance

```
laptop / GitHub                 40 (faretra, master node)          compute node (40, 43, 153, 232)
──────────────────              ──────────────────────────         ───────────────────────────────
git push  ───────────────────►  git pull (repo on disk)
                                 docker build (per node!)   ──────► image cached on THAT node
                                 sbatch sbatch_train.sh
                                        │
                                        ▼
                                 SLURM queue ─────────────────────► run_docker.sh runs on the
                                                                    node SLURM picked
                                                                       │
                                                                       ▼
                                                                 docker run (mounts repo + /llms)
                                                                       │
                                                                       ▼
                                                                 train.sh inside the container
                                                                       │
                                                                       ▼
                                                       python -m gom.ablations.main --config ...
```

Three separate things happen at three separate times:

1. **Getting the code onto a node** — `git clone`/`git pull` (or WinSCP). Not
   automatic; SLURM does not sync files between servers (see guide §7).
2. **Building the Docker image** — `docker build`, a slow one-off step you
   repeat only when dependencies or the Dockerfile change, **on every node**
   you plan to run on.
3. **Running a job** — `sbatch run_docker.sh <config.yaml>`, a fast step you
   repeat every time you want to launch (or relaunch) an experiment. This is
   what actually starts a container and executes `python -m gom.ablations.main`.

---

## 2. One-time setup per server

Do this once on **each** server you intend to use (faretra/40, moro43,
deeplearn2/153, moro232), because there is no shared filesystem across nodes
(guide §7) — code and Docker images both need to exist locally on whichever
node will run your job.

```bash
# 1. SSH in (repeat per server, using the ports/IPs from the guide's server table)
ssh username@137.204.107.40 -p 37335

# 2. Clone (only once) / pull (every time you push new changes)
git clone https://github.com/<you>/graph-of-marks.git
cd graph-of-marks
# on later visits: git pull

# 3. Put your dataset JSON under the repo so it's visible inside the container
#    at /workspace (the repo root is bind-mounted there by run_docker.sh — see
#    §4). The dataset is now a SINGLE flat JSON list of records, each
#    {image_path (basename only), question, answers: [...]}:
#    graph-of-marks/vqav1_limited_1000.json
#
#    The images themselves live ONLY on node 40 (faretra). You do NOT copy the
#    ~19 GB image folder to every node — see §2.1 for how other nodes fetch the
#    handful of images they need over HTTP from node 40.

# 4. Build the image (WHEN: once per node, and again any time
#    build/Dockerfile, build/Dockerfile.rtx5090, or build/requirements.txt
#    change — NOT before every job run)
docker build -f build/Dockerfile -t gom:latest .          # RTX 3090 / Titan Xp nodes
# on moro43 (RTX 5090) use the CUDA 12.8 variant instead:
docker build -f build/Dockerfile.rtx5090 -t gom:latest .

# 5. Sanity-check
docker images                 # confirm gom:latest exists
docker run --rm gom:latest python3 -m gom.ablations.main --help
```

**Why build per node, and why does it take so long?** Each server has its
own local Docker image cache (again, no shared filesystem — guide §7). The
build compiles/downloads torch, detectron2, vllm, spaCy/NLTK model data,
etc. — expect this to take a while the first time; subsequent builds reuse
Docker's layer cache and are much faster unless `build/requirements.txt`
changed.

**RTX 5090 (moro43 / server 43) uses a different, non-shared dependency
stack.** The RTX 5090 is Blackwell (sm_120) and needs CUDA 12.8 wheels, so
`build/Dockerfile.rtx5090` does **not** just reuse `build/requirements.txt`
with a torch swap. It strips the six pins that are cu124-specific
(`torch`/`torchvision`/`torchaudio` **and** `vllm`/`transformers`/`tokenizers`)
and installs a Blackwell-compatible set instead: `torch==2.8.0+cu128` +
`vllm==0.11.0` (which pulls `transformers==4.57.x`). This is required, not
optional — the standard image's `vllm==0.8.5` is built against torch
2.6.0/cu124, has no sm_120 kernels, and **fails to `import` against the cu128
torch**, so building the RTX 5090 node from the cu124 pins gives you a vLLM
that dies at startup. Verified building + running on server 43 as of
2026-07-04 (`torch.cuda` sees the RTX 5090 at sm_120; `from vllm import LLM`
imports cleanly). If you change either Dockerfile's torch/vllm versions,
re-verify with a `pip install --dry-run` first (see the troubleshooting note
on `ResolutionImpossible` below).

**When do you rebuild?** Only when:
- You edit `build/Dockerfile`, `build/Dockerfile.rtx5090`, or
  `build/requirements.txt` (added/removed a Python package).
- You change something in `setup.py` that the editable install depends on
  (rare — the image installs the package with `pip install --no-deps -e .`,
  which just registers `src/` on the path; you normally don't need to
  rebuild for ordinary code edits, see §3).

You do **not** rebuild when you only edit `slurm_configs/*.yaml` or the
Python source under `src/gom/` — see next section.

---

## 2.1 Images live only on node 40 — how other nodes get them

The dataset JSON (`vqav1_limited_1000.json`) only stores image **basenames**
(e.g. `COCO_train2014_000000487025.jpg`); the actual `.jpg` files live **only
on node 40 (faretra)**. Because there is no shared filesystem (guide §7), a
job that SLURM lands on node 43/153/232 cannot see node 40's disk. Rather than
copying the whole ~19 GB image folder to every node, node 40 **serves the
images over HTTP** and other nodes fetch (and locally cache) only the ~1000
images they actually use.

`src/gom/ablations/main.py::build_vqa_examples` resolves each image, per node,
in this order: **(1)** a local file at `images_dir/<basename>` (node 40) →
**(2)** a file already in `image_cache_dir/<basename>` (fetched on a previous
run) → **(3)** download `images_base_url/<basename>` into `image_cache_dir`.
Whatever it resolves, it hands the rest of the pipeline a normal local path,
so preprocessing/caching/evaluation are identical on every node.

**One-time setup on node 40 — start the image server** (host side, outside the
container; run it in a `byobu`/`tmux` session so it survives your SSH logout):

```bash
# on node 40 (faretra), in the directory that contains the COCO_train2014_*.jpg files
cd /path/to/train2014
python3 -m http.server 8000            # serves http://137.204.107.40:8000/<basename>.jpg
# quick check from another node:  curl -sI http://137.204.107.40:8000/COCO_train2014_000000487025.jpg
```

**Then, per config:**

- Running **on node 40**: set `images_dir` to the local image folder and leave
  `images_base_url: ""` — nothing is downloaded.
- Running **on any other node**: set
  `images_base_url: "http://137.204.107.40:8000"`. The `images_dir` value is
  harmless there (it just won't exist, so resolution falls through to the
  download). Fetched images are cached under `image_cache_dir` (default
  `{base_dir}/image_cache`, which is under the bind-mounted `/workspace`, so
  they persist on that node and are reused across grid points and reruns).

**Docker networking:** no change to `run_docker.sh` or the Dockerfiles is
needed. The container's default bridge network already allows outbound HTTP to
`137.204.107.40`, and `requests` (used to download) is already installed. Only
if your node blocks Docker's bridge NAT would you add `--network host` to the
`docker run` in `run_docker.sh`. The image cache dir must stay under
`/workspace` (the default does) so it's writable and survives the `--rm`
container.

---

## 3. Why editing code/config doesn't require a rebuild

The Dockerfiles deliberately do **not** bake your live source tree into the
image. They `COPY` just enough (`setup.py`, `pyproject.toml`, `src/`) at
build time to run `pip install --no-deps -e .`, which registers `gom` (and
`gom.ablations`) as an editable package pointing at `/workspace/src`.

At `docker run` time, `run_docker.sh` bind-mounts your **actual, current**
project directory over `/workspace` (`-v "$PHYS_DIR":/workspace`). Because
the mount uses the same path the editable install points to, the container
always executes whatever is on disk on that node right now — not whatever
was on disk when the image was built.

Practical consequence:
- `git pull` on the node, then just re-submit a job (`sbatch run_docker.sh
  slurm_configs/....yaml`) — **no rebuild needed** — to pick up source
  changes in `src/gom/ablations/*.py` or a new/edited `slurm_configs/*.yaml`.
- Rebuild only for actual dependency/environment changes (new pip package,
  changed CUDA/torch version, etc.).

---

## 4. What happens when you submit a job

```bash
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh slurm_configs/ablation_experiments.yaml
```

1. `sbatch` puts this in SLURM's shared queue and returns immediately
   (non-blocking — you can log out). Track it with `squeue`, cancel with
   `scancel <job_id>`.
2. When your turn comes and a matching GPU is free, SLURM runs
   `run_docker.sh slurm_configs/ablation_experiments.yaml` **on whichever
   node it picked** (which is why the code/data/image must already exist on
   every node you might target — no distributed filesystem).
3. `run_docker.sh` (repo root) starts the container:
   ```bash
   docker run \
       -v "$PHYS_DIR":/workspace \        # your repo, live, read-write
       -v /llms:/llms \                   # cluster's shared HF model cache
       -e HF_HOME=/llms \                 # vllm/transformers reuse cached weights
       --rm \                             # container removed after exit; outputs
                                           # survive because they're written under
                                           # the mounted /workspace
       --memory="30g" \
       --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \   # only the GPU SLURM assigned you
       gom:latest \
       /workspace/train.sh "$1"           # "$1" = the config path you passed to sbatch
   ```
4. Inside the container, `train.sh` runs:
   ```bash
   python3 -m gom.ablations.main --config "$1"
   ```
   i.e. exactly `src/gom/ablations/main.py`, nothing else.
5. Standard output/error go to `slurm-<job_id>.out` **on the node that ran
   the job**, per the cluster guide. Results (JSON metrics, preprocessed
   images) go wherever `base_dir` in your YAML points to — see §5 — which,
   since it's normally a path under `/workspace`, ends up back in your
   project directory on that node once the job finishes.

`GOM_IMAGE_NAME` (env var, default `gom:latest`) lets you point
`run_docker.sh` at a differently-tagged image if you build more than one.

To queue several experiments at once, edit/duplicate the `sbatch` lines in
`sbatch_train.sh` and run `./sbatch_train.sh` — each line queues one
independent job.

---

## 5. Customizing ablation studies (the part you actually asked about)

**You change what runs by editing a YAML file, not by editing `train.sh`,
`run_docker.sh`, `sbatch_train.sh`, or the Dockerfiles.** Those four files
are fixed plumbing; every experiment is fully described by one config
passed as `--config`.

Three ready-to-edit templates ship under `slurm_configs/`, one per
experiment type in `run_experiments.py`. Copy one to make a new experiment
set (e.g. `cp slurm_configs/ablation_experiments.yaml
slurm_configs/my_experiment.yaml`), edit it, then run:

```bash
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh slurm_configs/my_experiment.yaml
```

### 5.1 Fields shared by every config

```yaml
base_dir: "/workspace/ablation_studies/<name>"   # where preprocessed images + results are written
backend: "vllm"        # "vllm" (recommended on the cluster) or "ollama" (needs a local Ollama server — not available here)
n_runs: 3              # repeat each (model, config) combination this many times, report mean/std
num_examples: -1       # -1 = full dataset, or an integer to subsample (unique images)
force_reprocess: false # true = regenerate preprocessed images even if the folder already exists

dataset_path: "/workspace/vqav1_limited_1000.json"   # single flat VQAv1 JSON (list of {image_path, question, answers})
images_dir: "/workspace/data/train2014"              # local images — used only on node 40
images_base_url: ""    # "" on node 40; "http://137.204.107.40:8000" on every other node (see §2.1)
image_cache_dir: ""    # "" ⇒ {base_dir}/image_cache (must be under /workspace)
```

Use `/workspace/...` paths (the bind-mounted project dir) for anything you
want visible inside the container. See §2.1 for the `images_dir` /
`images_base_url` / `image_cache_dir` split (images live only on node 40). Model checkpoints (e.g.
`"Qwen/Qwen2.5-VL-7B-Instruct"`) are Hugging Face repo IDs — vllm downloads
them into `HF_HOME=/llms` (the shared cluster cache), so if a colleague
already pulled the same model you won't re-download it (see the guide's
"HELP US SAVE SOME MEMORY DISK" section).

### 5.2 `slurm_configs/ablation_experiments.yaml` — grid ablations

Turns on the `ablations:` block (`enabled: true`) and leaves
`vlm_comparison`/`prompting` disabled. Each entry under `experiments:` is
one ablation study: it preprocesses images across a grid of values, then
evaluates every model in `models:` against every grid point.

```yaml
ablations:
  enabled: true
  skip_preprocessing: false   # true = images already exist, jump straight to inference
  run_vlm: true                # false = only preprocess, useful to inspect images before spending GPU time
  models:
    - "Qwen/Qwen2.5-VL-7B-Instruct"     # add more lines to compare several models
  experiments:
    ablate_edge_thickness:
      ablation_grid:
        rel_arrow_linewidth: [0.5, 3.0]        # add/remove values to test more points
    ablate_max_relations_per_object:
      ablation_grid:
        max_relations_per_object: [0, 1, 2]
    # add a new experiment by adding a new key here, e.g.:
    # my_new_experiment:
    #   ablation_grid:
    #     some_param: [valueA, valueB]
```

**Important:** each key under `experiments:` must be handled by
`apply_experiment_config()` in `src/gom/ablations/main.py` (it maps an
experiment name to preprocessing-config overrides, e.g. which
`PreprocessorConfig` fields to force before generating images). If you add a
brand-new experiment name that isn't one of the existing `elif exp_name ==
...` branches there, it silently falls back to baseline-only overrides —
open `main.py` and add a branch if your new experiment needs specific
preprocessing overrides beyond the baseline defaults.

Results land in:
```
{base_dir}/results/{experiment_name}/{model}/{ablation_value}/summary_metrics.json
{base_dir}/results/{experiment_name}/{model}/{ablation_value}/run_N/raw_results.json
```

### 5.3 `slurm_configs/vlm_comparison.yaml` — compare models on one config

Enables `vlm_comparison:` only. Preprocesses images once (default config +
optional `preprocessing_overrides`), then runs every model in `models:`
against that fixed set of images.

```yaml
vlm_comparison:
  enabled: true
  skip_preprocessing: false
  models:
    - "Qwen/Qwen2.5-VL-3B-Instruct"
    - "Qwen/Qwen2.5-VL-7B-Instruct"
    - "llava-hf/llava-1.5-7b-hf"        # add/remove HF repo IDs to change which models are compared
  preprocessing_overrides: {}            # any gom.config.PreprocessorConfig field, e.g. {"label_mode": "numeric"}
```

Results land in:
```
{base_dir}/results/vlm_comparison/{model}/summary_metrics.json
{base_dir}/results/vlm_comparison/{model}/run_N/raw_results.json
```

### 5.4 `slurm_configs/prompting_experiments.yaml` — compare prompting strategies

Enables `prompting:` only. Preprocesses images once, then evaluates each
`enabled: true` strategy under `strategies:` against every model in
`models:`. The model is loaded once (expensive for vllm) and reused across
all enabled strategies.

```yaml
prompting:
  enabled: true
  models:
    - "Qwen/Qwen2.5-VL-7B-Instruct"
  strategies:
    baseline:
      enabled: true
    few_shot:
      enabled: true
      n_shots: 3          # tune per-strategy parameters here
    chain_of_thought:
      enabled: true
    graph_guided:
      enabled: true
```

Toggle a strategy off with `enabled: false` instead of deleting its block
(keeps the parameters around for later). The available strategies and their
parameters are defined in `src/gom/ablations/prompts.py`
(`build_prompt_template`) — add a new strategy there first if you want one
beyond the four shipped ones, then reference it by name here.

Results land in:
```
{base_dir}/results/prompting/{strategy}/{model}/summary_metrics.json
{base_dir}/results/prompting/{strategy}/{model}/run_N/raw_results.json
```

### 5.5 Running more than one experiment type in a single job

Nothing stops you from setting `enabled: true` on more than one of
`ablations` / `vlm_comparison` / `prompting` in the same YAML — `main.py`
runs whichever blocks are enabled, in that order, within a single process
(reusing the same loaded dataset and, where applicable, the same
preprocessor instance). This is useful for a smaller one-off run; for
larger jobs it's usually easier to keep them in separate config files (as
shipped) so each becomes its own SLURM job and can be retried/rescheduled
independently.

---

## 6. Day-to-day workflow once everything is set up

```bash
# on your laptop
git add slurm_configs/my_experiment.yaml
git commit -m "Add my_experiment ablation"
git push

# on the target node (e.g. faretra)
cd graph-of-marks
git pull
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh slurm_configs/my_experiment.yaml

# check on it
squeue                    # see PD/R status, job id
# tail -f slurm-<job_id>.out   once it's running, on whichever node it landed on

# if something's wrong and you need to stop it
scancel <job_id>
nvidia-smi                # double-check the GPU is actually freed after scancel
```

No Docker rebuild is needed anywhere in this loop — only `git pull` +
`sbatch`. You'd only add a `docker build` step back in if you'd also
changed `build/Dockerfile` or `build/requirements.txt`.

---

## 7. Troubleshooting checklist

- **`docker: command not found` / permission errors** — you haven't run
  `install_rootless_docker.sh` on this node yet, or need to restart your
  shell / run `systemctl --user start docker` (see the cluster guide §4).
- **Job fails immediately with a missing-file error** — the code, image, or
  dataset files under `data/` don't exist on the node SLURM picked. Re-run
  `git pull` and `docker build` on that specific node (guide §7).
- **`ImportError: vllm`/`ollama` inside the container** — you built the
  image before `build/requirements.txt` included these packages, or edited
  requirements without rebuilding; run `docker build` again.
- **vLLM import fails with an `undefined symbol` / torch-ABI error on the
  RTX 5090** — you built the 5090 node with the standard `build/Dockerfile`
  (or an old `Dockerfile.rtx5090` that kept `vllm==0.8.5`). That vLLM is
  compiled against torch 2.6.0/cu124 and cannot load against the cu128 torch
  the 5090 needs. Rebuild with the current `build/Dockerfile.rtx5090`
  (`torch==2.8.0+cu128` + `vllm==0.11.0`) — see §2.
- **`ResolutionImpossible` after changing a torch/vllm/transformers pin** —
  those three are tightly coupled (vLLM hard-pins an exact torch and a
  minimum transformers). Before rebuilding, verify the whole set resolves
  with `pip install --dry-run` (for the 5090 stack, include
  `--extra-index-url https://download.pytorch.org/whl/cu128`). Note the 5090
  image overrides `requirements.txt`'s torch/vllm/transformers/tokenizers
  pins — change them in `build/Dockerfile.rtx5090`, not just requirements.
- **Model re-downloads every run instead of reusing the cache** —
  double-check `/llms` actually exists on that node and `HF_HOME` is set (it
  is, automatically, by `run_docker.sh`); don't override `HF_HOME` in your
  YAML/environment.
- **Want a different GPU** — change `--gpus=nvidia_geforce_rtx_3090:1` in
  your `sbatch` call (`titan_xp` or `nvidia_geforce_rtx_5090` are also
  valid) and, for the RTX 5090, make sure you built with
  `build/Dockerfile.rtx5090` on that node, not the standard `build/Dockerfile`.
