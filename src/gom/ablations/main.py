import os
import time
import yaml
import json
import argparse
from collections import Counter

from .ablate_preprocessing import generate_ablated_dataset, generate_default_dataset
from .utils import update_cfg_correct, release_preprocessor
from .run_experiments import run_ablation_experiments, run_vlm_comparison, run_prompting_experiments
from .logging_utils import (
    setup_logging,
    log_section,
    log_key_values,
    log_models,
    log_preprocessor_config,
)
from gom.vqa.runner import VQAExample
from gom.config import default_config

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _majority_answer(answers) -> str:
    """VQA-style ground truth: the most frequent human answer (ties broken by
    first occurrence). Returns "" if there are no answers."""
    if not answers:
        return ""
    counts = Counter(a.strip() for a in answers if isinstance(a, str))
    if not counts:
        return ""
    # Counter.most_common preserves insertion order among equal counts (Py3.7+).
    return counts.most_common(1)[0][0]


def _resolve_local_image(
    basename: str,
    images_dir: str | None,
    image_cache_dir: str | None,
    images_base_url: str | None,
    *,
    session=None,
    timeout: float = 30.0,
    retries: int = 3,
) -> str | None:
    """
    Resolve one image basename (e.g. "COCO_train2014_000000487025.jpg") to a
    local absolute path, in this order:

      1. ``images_dir/basename`` if it already exists on this node (the node-40
         case where the images live locally — no download).
      2. ``image_cache_dir/basename`` if it was downloaded on a previous run.
      3. Download ``images_base_url/basename`` into ``image_cache_dir`` (used on
         every node that does NOT hold the images locally, i.e. everything but
         node 40; see README_SLURM.md — there is no shared filesystem, so images
         are served over HTTP by node 40 and fetched on demand).

    Returns the resolved absolute path, or None if the image cannot be located
    or downloaded (the caller skips such examples instead of crashing the run).
    """
    # 1) Already present locally (node 40, or manually staged).
    if images_dir:
        local = os.path.join(images_dir, basename)
        if os.path.isfile(local):
            return os.path.abspath(local)

    # 2) Previously cached on this node.
    cache_path = None
    if image_cache_dir:
        cache_path = os.path.join(image_cache_dir, basename)
        if os.path.isfile(cache_path):
            return os.path.abspath(cache_path)

    # 3) Fetch from node 40's HTTP image server into the local cache.
    if images_base_url and cache_path:
        import requests  # already a project dependency (see gom.vqa.io)

        url = f"{images_base_url.rstrip('/')}/{basename}"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        sess = session or requests
        for attempt in range(1, retries + 1):
            try:
                resp = sess.get(url, timeout=timeout, stream=True)
                resp.raise_for_status()
                tmp_path = cache_path + ".part"
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
                # Atomic rename so a partial download is never mistaken for a
                # complete cache hit on a later run.
                os.replace(tmp_path, cache_path)
                return os.path.abspath(cache_path)
            except Exception as exc:  # noqa: BLE001 — retry any transient failure
                if attempt == retries:
                    print(f"  ⚠️  Download fallito per {url}: {exc}")
                    return None
                time.sleep(min(2 ** attempt, 10))

    return None


def build_vqa_examples(
    dataset_path: str,
    images_dir: str | None = None,
    images_base_url: str | None = None,
    image_cache_dir: str | None = None,
) -> list[VQAExample]:
    """
    Build VQAExamples from the flat VQAv1 JSON (a list of records, each with
    ``image_path`` [basename only], ``question`` and ``answers`` [list of 10]).

    Every example's ``image_path`` is rewritten to a resolved LOCAL absolute
    path (see ``_resolve_local_image``) so that the rest of the pipeline —
    basename-keyed preprocessing caches, image grouping, evaluation — runs
    unchanged regardless of which SLURM node executes the job.
    """
    print("Caricamento JSON VQA...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(
            f"Formato dataset non valido: atteso una lista di record, trovato {type(records).__name__}. "
            f"File: {dataset_path}"
        )

    # Fail fast with an actionable message when there is provably NO image
    # source, instead of silently skipping every record and raising a generic
    # error at the end. Sources, in resolution order (see _resolve_local_image):
    #   1. images_dir      — local dir (node 40; bind-mounted at /images)
    #   2. image_cache_dir — images fetched on a previous run (counted only if
    #                        it already holds files: a bare default path isn't a
    #                        source)
    #   3. images_base_url — HTTP download from node 40
    # Raising only when none of the three can serve an image means this never
    # false-positives on a legitimate local/HTTP/warm-cache run.
    def _dir_has_files(path) -> bool:
        if not path:
            return False
        try:
            with os.scandir(path) as it:
                return any(True for _ in it)
        except (FileNotFoundError, NotADirectoryError):
            return False

    have_local = bool(images_dir) and os.path.isdir(images_dir)
    have_http = bool(images_base_url)
    have_cache = _dir_has_files(image_cache_dir)
    if not (have_local or have_http or have_cache):
        raise RuntimeError(
            "Nessuna sorgente immagini disponibile:\n"
            f"  • images_dir={images_dir!r}: assente o non è una directory nel container.\n"
            f"  • image_cache_dir={image_cache_dir!r}: assente o vuota.\n"
            f"  • images_base_url={images_base_url!r}: non impostato.\n"
            "→ Su node 40 (faretra): imposta GOM_IMAGES_DIR (o ~/.gom_images_dir) così che "
            "run_docker.sh monti il dataset immagini su /images (README_SLURM.md §2.1).\n"
            "→ Su ogni altro nodo: imposta images_base_url al server immagini di node 40 "
            "(es. http://137.204.107.40:8000)."
        )

    if images_base_url:
        print(f"🌐 Immagini servite via HTTP da: {images_base_url} → cache: {image_cache_dir}")
    if images_dir:
        print(f"📁 Directory immagini locale: {images_dir}")

    # Resolve each distinct image once (records share images: ~3 questions/image),
    # so a given image is downloaded/looked up a single time.
    import requests
    session = requests.Session()
    resolved_cache: dict[str, str | None] = {}

    def resolve(basename: str) -> str | None:
        if basename not in resolved_cache:
            resolved_cache[basename] = _resolve_local_image(
                basename, images_dir, image_cache_dir, images_base_url, session=session,
            )
        return resolved_cache[basename]

    print("Costruzione dataset VQAExample...")
    examples: list[VQAExample] = []
    missing = 0
    for rec in records:
        raw_path = rec.get("image_path", "")
        basename = os.path.basename(raw_path)
        question = rec.get("question", "")
        answers = rec.get("answers", []) or []

        local_path = resolve(basename)
        if local_path is None:
            missing += 1
            continue

        image_id = os.path.splitext(basename)[0]  # e.g. COCO_train2014_000000487025
        ex = VQAExample(
            image_path=local_path,
            question=question,
            answer=_majority_answer(answers),
            image_id=image_id,
            metadata={"answers": answers, "image_file": basename, "dataset": "vqav1"},
        )
        examples.append(ex)

    n_images = len({ex.image_id for ex in examples})
    print(f"Costruiti con successo {len(examples)} VQAExamples su {n_images} immagini!")
    if missing:
        print(f"⚠️  {missing} record saltati: immagine non trovata né scaricabile.")
    if not examples:
        raise RuntimeError(
            "Nessuna immagine risolta (tutti i record saltati). Verifica che i basename "
            "del JSON esistano nella sorgente configurata:\n"
            f"  • node 40 (faretra): il mount /images ({images_dir!r}) deve contenere i .jpg — "
            "controlla GOM_IMAGES_DIR / ~/.gom_images_dir (README_SLURM.md §2.1);\n"
            f"  • altri nodi: images_base_url ({images_base_url!r}) deve puntare al server "
            "immagini di node 40 e i file devono essere scaricabili."
        )
    return examples

def apply_experiment_config(preproc_obj, exp_name: str):
    """
    Resets the preprocessor to the baseline config, then applies experiment-specific overrides.
    Updates the existing object in memory (avoids reloading model weights).
    Unknown exp_name values fall through with config_changes = {} and receive baseline only.
    """
    base_cfg_updates = {
        "apply_question_filter": True,
        "aggressive_pruning": False,
        "auto_scale_styles": False,
        "rel_arrow_linewidth": 2.0,
        "auto_adjust_relation_cap": True,
        "cap_relations_per_object": False,
        "ablate_max_per_object": False,
        "ablate_max_global": False,
        "min_relations_per_object": 1,
        "max_relations_per_object": 3,
        "color_edge": "head"
    }

    config_changes = {}

    if exp_name == "ablate_edge_thickness":
        pass  # baseline only

    elif exp_name == "ablate_max_relations_per_object":
        config_changes = {
            "rel_arrow_linewidth": 2.0,
            "auto_adjust_relation_cap": False,
            "cap_relations_per_object": True,
            "ablate_max_per_object": True
        }

    elif exp_name == "ablate_edge_color":
        config_changes = {
            "rel_arrow_linewidth": 2.0,
            "auto_adjust_relation_cap": False,
            "cap_relations_per_object": True,
            "ablate_max_per_object": False,
            "ablate_max_global": False,
            "min_relations_per_object": 0,
            "max_relations_per_object": 3
        }

    elif exp_name == "ablate_max_relations_global":
        config_changes = {
            "auto_adjust_relation_cap": False,
            "cap_relations_per_object": False,
            "min_relations_per_object": 0,
            "max_relations_per_object": 3,
            "ablate_max_per_object": False,
            "rel_arrow_linewidth": 2.0,
            "ablate_max_global": True,
            "color_edge": "head"
        }

    final_updates = {**base_cfg_updates, **config_changes}

    # Grid ablations keep question-guided filtering ON (apply_question_filter is
    # True in base_cfg_updates above), matching vlm_comparison / prompting and
    # the upstream repo default. This trims the marked set to question-relevant
    # objects so the images aren't overcrowded; the swept hyperparameter is still
    # the only *config* knob varying across grid points.

    return update_cfg_correct(final_updates, preproc_obj)

def main():
    parser = argparse.ArgumentParser(description="Lancia la pipeline degli studi ablativi via YAML")
    parser.add_argument("--config", type=str, required=True, help="Percorso al file config.yaml")
    args = parser.parse_args()

    print(f"📄 Lettura configurazione da: {args.config}")
    cfg = load_config(args.config)

    # --- Global settings ---
    base_dir         = cfg.get("base_dir", "ablation_studies")

    # Initialize the file logger as early as possible (needs base_dir).
    logger = setup_logging(base_dir)
    logger.info("Config file: %s", args.config)
    backend          = cfg.get("backend", "ollama")
    n_runs           = cfg.get("n_runs", 3)
    # num_images         : how many unique images to keep      (-1 = all images).
    # questions_per_image: how many questions to keep per image (-1 = all questions).
    # ``num_examples`` is the legacy name for ``num_images`` and is still honoured.
    num_images       = cfg.get("num_images", cfg.get("num_examples", -1))
    questions_per_image = cfg.get("questions_per_image", -1)
    force_reprocess  = cfg.get("force_reprocess", False)
    dataset_path     = cfg.get("dataset_path")
    images_dir       = cfg.get("images_dir")
    images_base_url  = cfg.get("images_base_url")
    # Where images fetched from node 40 are cached locally. Must live under the
    # bind-mounted /workspace so it is writable and persists across reruns on
    # the node (see README_SLURM.md). Defaults to {base_dir}/image_cache.
    image_cache_dir  = cfg.get("image_cache_dir") or os.path.join(base_dir, "image_cache")

    # --- Ablations section ---
    ablations_cfg            = cfg.get("ablations", {})
    ablations_enabled        = ablations_cfg.get("enabled", False)
    ablations_skip_preproc   = ablations_cfg.get("skip_preprocessing", False)
    ablations_run_vlm        = ablations_cfg.get("run_vlm", True)
    ablations_models         = ablations_cfg.get("models", [])
    experiments              = ablations_cfg.get("experiments", {})
    # Static preprocessing overrides applied to EVERY grid point on top of the
    # per-experiment baseline (mirrors vlm_comparison / prompting). Intended for
    # cross-cutting knobs like aggressive_pruning / auto_scale_styles that should
    # stay constant across the swept grid; the grid params still win on collision.
    ablations_overrides      = ablations_cfg.get("preprocessing_overrides", {}) or {}

    # --- VLM comparison section ---
    vlm_comparison_cfg          = cfg.get("vlm_comparison", {})
    vlm_comparison_enabled      = vlm_comparison_cfg.get("enabled", False)
    vlm_comparison_skip_preproc = vlm_comparison_cfg.get("skip_preprocessing", False)
    vlm_comparison_models       = vlm_comparison_cfg.get("models", [])
    vlm_preprocessing_overrides = vlm_comparison_cfg.get("preprocessing_overrides", {}) or {}
    # Which image the VLM sees, and whether the scene-graph triples are prepended
    # to the prompt. The defaults reproduce the GoM setup (annotated render + graph
    # text); "raw" + include_scene_graph: false is the no-GoM baseline, and
    # "raw" + true is the textual-only ablation. See run_vlm_comparison.
    # NB: a run with non-default values needs its OWN base_dir — run_vqa resumes
    # from an existing raw_results.json and would otherwise reuse the previous
    # run's answers (see slurm_configs/vlm_comparison_raw.yaml).
    vlm_inference_image         = vlm_comparison_cfg.get("inference_image", "preprocessed")
    vlm_include_scene_graph     = vlm_comparison_cfg.get("include_scene_graph", True)

    # --- Prompting section ---
    prompting_cfg           = cfg.get("prompting", {})
    prompting_enabled       = prompting_cfg.get("enabled", False)
    prompting_skip_preproc  = prompting_cfg.get("skip_preprocessing", False)
    prompting_models        = prompting_cfg.get("models", [])
    prompting_overrides     = prompting_cfg.get("preprocessing_overrides", {}) or {}
    prompting_strategies    = prompting_cfg.get("strategies", {})

    # Shared prompts (identical for both experiment types).
    # IMPORTANT: we do NOT forbid explanation/reasoning. Doing so would cripple
    # reasoning models (LlamaV-o1, *-Thinking) and contradict chain_of_thought
    # prompting. Instead the model may reason freely and is only asked to
    # CONCLUDE with a concise, parseable final answer ("Answer: <word/phrase>").
    # The official VQA metric (ablations/evaluation.py) scores that final answer,
    # which extract_final_answer pulls out after the last "Answer:" marker — so
    # reasoning is preserved AND short-answer scoring works. Note the prompt does
    # not END with "Answer:", which would pre-empt a reasoning model's thinking;
    # the model emits the "Answer:" line itself.
    system_prompt     = "You are a multimodal assistant capable of understanding both visual and textual scene graphs. Use the image and the accompanying graph description to answer the question accurately."
    multimodal_prompt = "Answer the question based on the spatial configuration in the image and the graph description. Conclude with your final answer on a new line in the form:\nAnswer: <one word or a short phrase>\n\nQuestion: {question}"

    # Baseline variants, used only when a run sends no scene graph (see
    # include_scene_graph above). The shared prompts tell the model to use "the
    # graph description", which does not exist in that run — a control condition
    # must not instruct the model to consult something absent. Same "Answer:"
    # contract, so ablations/evaluation.py extracts and scores them identically.
    baseline_system_prompt     = "You are a multimodal assistant. Use the image to answer the question accurately."
    baseline_multimodal_prompt = "Answer the question based on the image. Conclude with your final answer on a new line in the form:\nAnswer: <one word or a short phrase>\n\nQuestion: {question}"

    # Record the global run settings and which experiment types are enabled.
    log_section("GLOBAL RUN SETTINGS")
    log_key_values("Global settings", {
        "base_dir": base_dir,
        "backend": backend,
        "n_runs": n_runs,
        "num_images": num_images,
        "questions_per_image": questions_per_image,
        "force_reprocess": force_reprocess,
        "dataset_path": dataset_path,
        "images_dir": images_dir,
        "images_base_url": images_base_url,
        "image_cache_dir": image_cache_dir,
    })
    log_key_values("Enabled experiment types", {
        "ablations": ablations_enabled,
        "vlm_comparison": vlm_comparison_enabled,
        "prompting": prompting_enabled,
    })

    print("\n📦 Caricamento Dataset in corso...")
    dataset = build_vqa_examples(
        dataset_path=dataset_path,
        images_dir=images_dir,
        images_base_url=images_base_url,
        image_cache_dir=image_cache_dir,
    )

    logger.info("Dataset built: %d VQA examples from %d unique images.",
                len(dataset), len({ex.image_id for ex in dataset}))

    # Subsample by unique image and (independently) by questions per image.
    #   • num_images > 0          → keep only the first ``num_images`` distinct
    #     images (in first-seen order); -1 keeps every image.
    #   • questions_per_image > 0 → within each kept image, keep at most that
    #     many questions (in first-seen order); -1 keeps every question.
    # The flat dataset holds several questions per image, and they are not
    # guaranteed to be contiguous, so we scan the whole list rather than
    # breaking early: an already-selected image may reappear later.
    if num_images > 0 or questions_per_image > 0:
        dataset_examples = []
        seen_ids: set[str] = set()
        per_image_count: dict[str, int] = {}
        for ex in dataset:
            img = ex.image_id
            if img not in seen_ids:
                # A new image: admit it only if we still have image budget.
                if num_images > 0 and len(seen_ids) >= num_images:
                    continue
                seen_ids.add(img)
                per_image_count[img] = 0
            # ``img`` is now a selected image — enforce the per-image quota.
            if questions_per_image > 0 and per_image_count[img] >= questions_per_image:
                continue
            dataset_examples.append(ex)
            per_image_count[img] += 1
        print(
            f"✂️  Dataset limitato a {len(seen_ids)} immagini uniche"
            f" (num_images={num_images}, questions_per_image={questions_per_image})"
            f" → {len(dataset_examples)} esempi."
        )
    else:
        dataset_examples = dataset
        print(f"📊 Utilizzo dell'intero dataset: {len(dataset_examples)} esempi.")

    n_unique_images = len({ex.image_id for ex in dataset_examples})
    logger.info(
        "Dataset in use: %d examples across %d unique images "
        "(num_images=%d, questions_per_image=%d).",
        len(dataset_examples), n_unique_images, num_images, questions_per_image,
    )

    # Initialize the shared preprocessor only when at least one preprocessing phase will run.
    needs_preprocessing = (
        (ablations_enabled and not ablations_skip_preproc) or
        (vlm_comparison_enabled and not vlm_comparison_skip_preproc) or
        (prompting_enabled and not prompting_skip_preproc)
    )
    if needs_preprocessing:
        preprocessor = update_cfg_correct(None)
    else:
        preprocessor = None

    # ==========================================
    # ABLATION EXPERIMENTS
    # ==========================================
    if ablations_enabled:
        log_section("ABLATION EXPERIMENTS")
        log_models(ablations_models, backend)
        log_key_values("Ablation settings", {
            "skip_preprocessing": ablations_skip_preproc,
            "run_vlm": ablations_run_vlm,
            "experiments": list(experiments.keys()),
        })
        log_key_values("Preprocessing overrides", ablations_overrides)

        if not ablations_skip_preproc:
            print("\n" + "═"*50)
            print("🛠️  ABLATIONS — FASE 1: PREPROCESSING")
            print("═"*50)
            logger.info("Ablations — phase 1: preprocessing started.")

            if preprocessor is None:  # a prior experiment may have released it
                preprocessor = update_cfg_correct(None)
            for exp_name, exp_data in experiments.items():
                ablation_grid = exp_data.get("ablation_grid")
                if not ablation_grid:
                    logger.info("Experiment '%s' has no ablation_grid; skipping.", exp_name)
                    continue
                preprocessor = apply_experiment_config(preprocessor, exp_name)
                log_section(f"[PREPROCESSING] Experiment: {exp_name}")
                log_key_values("Ablation grid", ablation_grid)
                log_preprocessor_config(preprocessor)
                print(f"\n[Fase 1] Generazione dataset per: {exp_name.upper()}")
                generate_ablated_dataset(
                    experiment_name=exp_name,
                    ablation_grid=ablation_grid,
                    examples=dataset_examples,
                    preproc_obj=preprocessor,
                    preprocessing_overrides=ablations_overrides,
                    base_dir=base_dir,
                    force_reprocess=force_reprocess
                )
        else:
            print("\n⏭️  [Ablations Fase 1 SKIP] Preprocessing saltato da configurazione.")
            logger.info("Ablations — phase 1: preprocessing SKIPPED by configuration.")

        if ablations_run_vlm:
            print("\n" + "═"*50)
            print("🧠  ABLATIONS — FASE 2: INFERENZA E VALUTAZIONE")
            print("═"*50)
            logger.info("Ablations — phase 2: VLM inference and evaluation started.")

            # Preprocessing done; free the preprocessor's GPU models before
            # loading any VLM so it runs on the full card in bf16.
            preprocessor = release_preprocessor(preprocessor)
            for exp_name, exp_data in experiments.items():
                ablation_grid = exp_data.get("ablation_grid")
                if not ablation_grid:
                    continue
                logger.info("Ablations — phase 2: running experiment '%s'.", exp_name)
                print(f"\n[Fase 2] Esecuzione modelli per: {exp_name.upper()}")
                run_ablation_experiments(
                    experiment_name=exp_name,
                    ablation_grid=ablation_grid,
                    models_list=ablations_models,
                    examples=dataset_examples,
                    multimodal_prompt=multimodal_prompt,
                    system_prompt=system_prompt,
                    n_runs=n_runs,
                    base_dir=base_dir,
                    backend=backend,
                )
        else:
            print("\n⏭️  [Ablations Fase 2 SKIP] Inferenza saltata da configurazione.")
            logger.info("Ablations — phase 2: VLM inference SKIPPED by configuration.")

    # ==========================================
    # VLM COMPARISON EXPERIMENT
    # ==========================================
    if vlm_comparison_enabled:
        print("\n" + "═"*50)
        print("🔬 VLM COMPARISON EXPERIMENT")
        print("═"*50)
        log_section("VLM COMPARISON EXPERIMENT")
        log_models(vlm_comparison_models, backend)
        log_key_values("VLM comparison settings", {
            "skip_preprocessing": vlm_comparison_skip_preproc,
            "inference_image": vlm_inference_image,
            "include_scene_graph": vlm_include_scene_graph,
        })
        log_key_values("Preprocessing overrides", vlm_preprocessing_overrides)

        if not vlm_comparison_skip_preproc:
            print("\n[VLM Comparison] Generating default preprocessed images...")
            if preprocessor is None:  # a prior experiment may have released it
                preprocessor = update_cfg_correct(None)
            preprocessor = apply_experiment_config(preprocessor, "vlm_comparison")
            log_preprocessor_config(preprocessor)
            generate_default_dataset(
                experiment_name="vlm_comparison",
                examples=dataset_examples,
                preproc_obj=preprocessor,
                preprocessing_overrides=vlm_preprocessing_overrides,
                base_dir=base_dir,
                force_reprocess=force_reprocess,
            )
        else:
            print("\n⏭️  [VLM Comparison Preprocessing SKIP] Preprocessing skipped by configuration.")
            logger.info("VLM comparison — preprocessing SKIPPED by configuration.")

        # Preprocessing (if any) is done and the images are on disk; free the
        # preprocessor's ~6GB of GPU models so each VLM can use the full card in
        # bf16. Inference never uses the preprocessor (run_vqa skip_preproc=True).
        preprocessor = release_preprocessor(preprocessor)

        run_vlm_comparison(
            experiment_name="vlm_comparison",
            models_list=vlm_comparison_models,
            examples=dataset_examples,
            # A run that sends no scene graph must not be told to use one.
            multimodal_prompt=multimodal_prompt if vlm_include_scene_graph else baseline_multimodal_prompt,
            system_prompt=system_prompt if vlm_include_scene_graph else baseline_system_prompt,
            n_runs=n_runs,
            base_dir=base_dir,
            backend=backend,
            inference_image=vlm_inference_image,
            include_scene_graph=vlm_include_scene_graph,
        )

    # ==========================================
    # PROMPTING TECHNIQUES EXPERIMENT
    # ==========================================
    if prompting_enabled:
        print("\n" + "═"*50)
        print("📝 PROMPTING TECHNIQUES EXPERIMENT")
        print("═"*50)
        log_section("PROMPTING TECHNIQUES EXPERIMENT")
        log_models(prompting_models, backend)
        log_key_values("Prompting settings", {
            "skip_preprocessing": prompting_skip_preproc,
            "enabled_strategies": [
                name for name, s in prompting_strategies.items()
                if isinstance(s, dict) and s.get("enabled", False)
            ],
        })
        log_key_values("Preprocessing overrides", prompting_overrides)

        if not prompting_skip_preproc:
            print("\n[Prompting] Generating default preprocessed images...")
            if preprocessor is None:  # a prior experiment may have released it
                preprocessor = update_cfg_correct(None)
            preprocessor = apply_experiment_config(preprocessor, "prompting")
            log_preprocessor_config(preprocessor)
            generate_default_dataset(
                experiment_name="prompting",
                examples=dataset_examples,
                preproc_obj=preprocessor,
                preprocessing_overrides=prompting_overrides,
                base_dir=base_dir,
                force_reprocess=force_reprocess,
            )
        else:
            print("\n⏭️  [Prompting Preprocessing SKIP] Preprocessing skipped by configuration.")
            logger.info("Prompting — preprocessing SKIPPED by configuration.")

        # Preprocessing done; free the preprocessor's GPU models before loading
        # any VLM so it runs on the full card in bf16.
        preprocessor = release_preprocessor(preprocessor)

        run_prompting_experiments(
            experiment_name="prompting",
            strategies=prompting_strategies,
            models_list=prompting_models,
            examples=dataset_examples,
            system_prompt=system_prompt,
            n_runs=n_runs,
            base_dir=base_dir,
            backend=backend,
        )

    print("\n🏁 PIPELINE COMPLETATA CON SUCCESSO! 🏁")
    log_section("PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
