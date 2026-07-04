import os
import time
import yaml
import json
import argparse
from collections import Counter

from .ablate_preprocessing import generate_ablated_dataset, generate_default_dataset
from .utils import update_cfg_correct
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
            "Nessuna immagine risolta. Verifica 'images_dir' (nodo 40) oppure "
            "'images_base_url'/'image_cache_dir' (altri nodi) nel file di config."
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
    num_examples     = cfg.get("num_examples", -1)
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

    # --- VLM comparison section ---
    vlm_comparison_cfg          = cfg.get("vlm_comparison", {})
    vlm_comparison_enabled      = vlm_comparison_cfg.get("enabled", False)
    vlm_comparison_skip_preproc = vlm_comparison_cfg.get("skip_preprocessing", False)
    vlm_comparison_models       = vlm_comparison_cfg.get("models", [])
    vlm_preprocessing_overrides = vlm_comparison_cfg.get("preprocessing_overrides", {}) or {}

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

    # Record the global run settings and which experiment types are enabled.
    log_section("GLOBAL RUN SETTINGS")
    log_key_values("Global settings", {
        "base_dir": base_dir,
        "backend": backend,
        "n_runs": n_runs,
        "num_examples": num_examples,
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

    if num_examples > 0:
        dataset_examples = []
        seen_ids = set()
        for ex in dataset:
            if ex.image_id not in seen_ids:
                dataset_examples.append(ex)
                seen_ids.add(ex.image_id)
            if len(dataset_examples) >= num_examples:
                break
        print(f"✂️  Dataset limitato a {num_examples} immagini uniche.")
    else:
        dataset_examples = dataset
        print(f"📊 Utilizzo dell'intero dataset: {len(dataset_examples)} esempi.")

    n_unique_images = len({ex.image_id for ex in dataset_examples})
    logger.info(
        "Dataset in use: %d examples across %d unique images (num_examples=%d).",
        len(dataset_examples), n_unique_images, num_examples,
    )

    # Initialize the shared preprocessor only when at least one preprocessing phase will run.
    needs_preprocessing = (
        (ablations_enabled and not ablations_skip_preproc) or
        (vlm_comparison_enabled and not vlm_comparison_skip_preproc) or
        (prompting_enabled and not prompting_skip_preproc)
    )
    if needs_preprocessing:
        preprocessor = update_cfg_correct()
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

        if not ablations_skip_preproc:
            print("\n" + "═"*50)
            print("🛠️  ABLATIONS — FASE 1: PREPROCESSING")
            print("═"*50)
            logger.info("Ablations — phase 1: preprocessing started.")

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
        })
        log_key_values("Preprocessing overrides", vlm_preprocessing_overrides)

        if not vlm_comparison_skip_preproc:
            print("\n[VLM Comparison] Generating default preprocessed images...")
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

        run_vlm_comparison(
            experiment_name="vlm_comparison",
            models_list=vlm_comparison_models,
            examples=dataset_examples,
            multimodal_prompt=multimodal_prompt,
            system_prompt=system_prompt,
            n_runs=n_runs,
            base_dir=base_dir,
            backend=backend,
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
