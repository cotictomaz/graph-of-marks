"""
The file ablate_preprocessing.py contains the logic to generate the preprocessed images 
the VLM will consume for the VQA task during ablation studies. 

Instead of duplicating logic for each parameter, it uses a unified generation function:
- generate_ablated_dataset(ablation_type, ablation_value, examples, base_config, ...)

This function updates the configuration dynamically based on the `ablation_type` 
(e.g., 'edge_thickness', 'max_relations', 'edge_color') and delegates the processing 
to the core IGP pipeline.

To save disk space and I/O time, preprocessing is strictly deterministic. Therefore, 
it does not split artifacts by "run". The unified directory structure is:
- ablation_studies
    - preprocessed_images
        - experiment_name (e.g., ablate_edge_thickness)
            - ablation_value (e.g., thickness_0.5)
                - all generated files (jpeg, scene_graph.txt, etc.)

The functions in run_experiments.py are then designed to call the VLM, pointing it 
to these shared preprocessed directories. Since inference can be non-deterministic 
(e.g., temperature > 0), the evaluation results DO include the "run" level:
- ablation_studies
    - results
        - experiment_name
            - model_name
                - ablation_value
                    - current_run (e.g., run_1, run_2)
                        - raw inference files
                    - summary_metrics.json (mean and std across runs)
"""

import os
import shutil
from typing import List, Dict, Any, Optional

from .utils import run_preprocessing


def generate_default_dataset(
    experiment_name: str,
    examples: List[Any],
    preproc_obj: Optional[Any] = None,
    preprocessing_overrides: Optional[Dict[str, Any]] = None,
    base_dir: str = "ablation_studies",
    force_reprocess: bool = False,
) -> None:
    """
    Generates a single preprocessed dataset using the default (or lightly overridden) config.

    Unlike generate_ablated_dataset, there is no ablation grid — every image is processed
    once with a fixed configuration. Artifacts are saved to:
        {base_dir}/preprocessed_images/{experiment_name}/default/

    Parameters:
        experiment_name: Name used to locate the output folder (e.g. "vlm_comparison").
        examples: VQA examples to process.
        preproc_obj: Optional pre-loaded preprocessor (avoids reloading models).
        preprocessing_overrides: Config overrides applied on top of whatever base config
            the preproc_obj was reset to before this call.
        base_dir: Root directory for all ablation outputs.
        force_reprocess: If True, deletes and rebuilds an existing folder.
    """
    out_dir = os.path.join(base_dir, "preprocessed_images", experiment_name, "default")

    if os.path.exists(out_dir):
        if force_reprocess:
            print(f"\n  [♻️ FORCE REPROCESS] Clearing existing folder and recomputing: {out_dir}")
            shutil.rmtree(out_dir)
            os.makedirs(out_dir)
        else:
            if len(os.listdir(out_dir)) > 0:
                print(f"\n  [⏭️ SKIP] Default dataset already present in {out_dir}. Skipping.")
                return
            else:
                print(f"\n  [▶️ RESUME] Empty folder found at {out_dir}. Starting processing.")
    else:
        os.makedirs(out_dir)

    print(f"\n  ↳ 🔧 Processing with default config (overrides: {preprocessing_overrides or {}})")
    print(f"  ↳ 💾 Saving to: {out_dir}")

    run_preprocessing(
        examples=examples,
        preproc_folder=out_dir,
        preproc_obj=preproc_obj,
        cfg_overrides=preprocessing_overrides or {},
        max_imgs=-1,
        max_qpi=-1,
    )

    print(f"\n✅ Default dataset generation for {experiment_name} done!")
    
def generate_ablated_dataset(
    experiment_name: str,
    ablation_grid: Dict[str, List[Any]],
    examples: List[Any],
    preproc_obj: Optional[Any] = None,
    base_dir: str = "ablation_studies",
    force_reprocess: bool = False 
) -> None:
    """
    Generates preprocessed datasets for a specific ablation study.

    Parameters:
        experiment_name (str): 
            Name of the experiment (e.g., 'ablate_edge_thickness').
        ablation_grid (Dict[str, List[Any]]): 
            A dictionary mapping configuration parameters to lists of values to test.
            If multiple keys are provided (e.g., parameters that must be ablated together), 
            their value lists must have the exact same length and will be iterated in parallel.
        examples (List[VQAExample]): 
            The dataset examples to process.
        preproc_obj (Optional[Preprocessor]): 
            An optionally pre-initialized preprocessor object to avoid reloading models.
        base_dir (str): 
            Base directory for saving all outputs.

    Description:
        The function loops over the provided ablation values in parallel, dynamically updates 
        the preprocessor configuration, and delegates the execution to `run_preprocessing`. 
        Artifacts are saved strictly in:
        {base_dir}/preprocessed_images/{experiment_name}/{param1_value1_param2_value2}/
    """
    keys = list(ablation_grid.keys())
    value_lists = list(ablation_grid.values())
    
    lengths = [len(v) for v in value_lists]
    if len(set(lengths)) > 1:
        raise ValueError(f"Tutte le liste in ablation_grid devono avere la stessa lunghezza. Trovate: {lengths}")
    
    print(f"🚀 Avvio Generazione Dataset: {experiment_name} (force_reprocess={force_reprocess})")

    for values_tuple in zip(*value_lists):
        current_overrides = dict(zip(keys, values_tuple))
        
        folder_parts = []
        for k, v in current_overrides.items():
            safe_v = str(v).replace(" ", "").replace("[", "").replace("]", "").replace(",", "_")
            folder_parts.append(f"{k}_{safe_v}")
            
        folder_suffix = "_".join(folder_parts)
        out_dir = os.path.join(base_dir, "preprocessed_images", experiment_name, folder_suffix)

        if os.path.exists(out_dir):
            if force_reprocess:
                print(f"\n  [♻️ FORCE REPROCESS] Cancello la cartella esistente e ricalcolo: {folder_suffix}")
                shutil.rmtree(out_dir)
                os.makedirs(out_dir)
            else:
                # Controlliamo se la cartella ha già dei file dentro.
                # Se è vuota (magari creata per errore e script interrotto), procediamo.
                if len(os.listdir(out_dir)) > 0:
                    print(f"\n  [⏭️ SKIP] Dataset già presente in {folder_suffix}. Passo al prossimo parametro.")
                    continue
                else:
                    print(f"\n  [▶️ RESUME] Cartella vuota trovata per {folder_suffix}. Inizio processamento.")
        else:
            os.makedirs(out_dir)

        print(f"\n  ↳ 🔧 Processamento configurazione: {current_overrides}")
        print(f"  ↳ 💾 Salvataggio in: {out_dir}")
        
        run_preprocessing(
            examples=examples,
            preproc_folder=out_dir,
            preproc_obj=preproc_obj,
            cfg_overrides=current_overrides,
            max_imgs=-1,
            max_qpi=-1
        )
        
    print(f"\n✅ Operazioni su {experiment_name} terminate!")