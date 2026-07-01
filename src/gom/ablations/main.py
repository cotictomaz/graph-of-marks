import os
import yaml
import json
import argparse

from .ablate_preprocessing import generate_ablated_dataset, generate_default_dataset
from .utils import update_cfg_correct
from .run_experiments import run_ablation_experiments, run_vlm_comparison
from gom.vqa.runner import VQAExample
from gom.config import default_config

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_vqa_examples(questions_json_path: str, annotations_json_path: str, images_dir: str) -> list[VQAExample]:
    print("Caricamento JSON VQA...")
    with open(questions_json_path, 'r') as f:
        questions_data = json.load(f)['questions']
    with open(annotations_json_path, 'r') as f:
        annotations_data = json.load(f)['annotations']

    answer_map = {ann['question_id']: ann['multiple_choice_answer'] for ann in annotations_data}
    examples = []

    print("Costruzione dataset VQAExample...")
    for q in questions_data:
        q_id = q['question_id']
        img_id = q['image_id']
        img_filename = f"COCO_val2014_{str(img_id).zfill(12)}.jpg"
        full_image_path = os.path.join(images_dir, img_filename)

        ex = VQAExample(
            image_path=full_image_path,
            question=q['question'],
            answer=answer_map.get(q_id, ""),
            image_id=str(img_id),
            metadata={"question_id": q_id, "dataset": "vqav1"}
        )
        examples.append(ex)

    print(f"Costruiti con successo {len(examples)} VQAExamples!")
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
    backend          = cfg.get("backend", "ollama")
    n_runs           = cfg.get("n_runs", 3)
    num_examples     = cfg.get("num_examples", -1)
    force_reprocess  = cfg.get("force_reprocess", False)
    question_path    = cfg.get("questions_path")
    annotations_path = cfg.get("annotations_path")
    images_dir       = cfg.get("images_dir")

    # --- Ablations section ---
    ablations_cfg            = cfg.get("ablations", {})
    ablations_enabled        = ablations_cfg.get("enabled", False)
    ablations_skip_preproc   = ablations_cfg.get("skip_preprocessing", False)
    ablations_run_vlm        = ablations_cfg.get("run_vlm", True)
    ablations_models         = ablations_cfg.get("models", [])
    experiments              = ablations_cfg.get("experiments", {})

    # --- VLM comparison section ---
    vlm_comparison_cfg       = cfg.get("vlm_comparison", {})
    vlm_comparison_enabled   = vlm_comparison_cfg.get("enabled", False)
    vlm_comparison_skip_preproc = vlm_comparison_cfg.get("skip_preprocessing", False)
    vlm_comparison_models    = vlm_comparison_cfg.get("models", [])
    vlm_preprocessing_overrides = vlm_comparison_cfg.get("preprocessing_overrides", {}) or {}

    # Shared prompts (identical for both experiment types)
    system_prompt     = "You are a multimodal assistant capable of understanding both visual and textual scene graphs. Use the image and the accompanying graph description to answer the question accurately."
    multimodal_prompt = "Answer the question based on the spatial configuration in the image and the graph description.\n\nQuestion: {question}"

    print("\n📦 Caricamento Dataset in corso...")
    dataset = build_vqa_examples(
        questions_json_path=question_path,
        annotations_json_path=annotations_path,
        images_dir=images_dir
    )

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

    # Initialize the shared preprocessor only when at least one preprocessing phase will run.
    needs_preprocessing = (
        (ablations_enabled and not ablations_skip_preproc) or
        (vlm_comparison_enabled and not vlm_comparison_skip_preproc)
    )
    if needs_preprocessing:
        print("\n🤖 Inizializzazione del Preprocessor Globale (YOLO)...")
        preprocessor = update_cfg_correct({"detectors_to_use": ("yolov8",)})
    else:
        preprocessor = None

    # ==========================================
    # ABLATION EXPERIMENTS
    # ==========================================
    if ablations_enabled:
        if not ablations_skip_preproc:
            print("\n" + "═"*50)
            print("🛠️  ABLATIONS — FASE 1: PREPROCESSING")
            print("═"*50)

            for exp_name, exp_data in experiments.items():
                ablation_grid = exp_data.get("ablation_grid")
                if not ablation_grid:
                    continue
                preprocessor = apply_experiment_config(preprocessor, exp_name)
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

        if ablations_run_vlm:
            print("\n" + "═"*50)
            print("🧠  ABLATIONS — FASE 2: INFERENZA E VALUTAZIONE")
            print("═"*50)

            for exp_name, exp_data in experiments.items():
                ablation_grid = exp_data.get("ablation_grid")
                if not ablation_grid:
                    continue
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

    # ==========================================
    # VLM COMPARISON EXPERIMENT
    # ==========================================
    if vlm_comparison_enabled:
        print("\n" + "═"*50)
        print("🔬 VLM COMPARISON EXPERIMENT")
        print("═"*50)

        if not vlm_comparison_skip_preproc:
            print("\n[VLM Comparison] Generating default preprocessed images...")
            preprocessor = apply_experiment_config(preprocessor, "vlm_comparison")
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

    print("\n🏁 PIPELINE COMPLETATA CON SUCCESSO! 🏁")

if __name__ == "__main__":
    main()
