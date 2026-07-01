import os
import yaml
import json
import argparse

# Importa le tue funzioni
from .ablate_preprocessing import generate_ablated_dataset
from .utils import update_cfg_correct
from .run_experiments import run_ablation_experiments
from gom.vqa.runner import VQAExample
from gom.config import default_config

def load_config(yaml_path: str) -> dict:
    """Legge e parsa il file YAML di configurazione."""
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
    Ripristina la configurazione di base e applica le modifiche dell'esperimento,
    aggiornando lo STESSO oggetto preprocessor in memoria.
    """
    # 1. BASELINE: Inserisci qui TUTTI i valori di default che potresti 
    # aver modificato in uno degli esperimenti, così li resetti sempre a zero!
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
    
    # 2. OVERRIDE SPECIFICI DELL'ESPERIMENTO
    config_changes = {}
    
    if exp_name == "ablate_edge_thickness":
        pass # Usa solo il baseline
        
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
        
    # 3. Uniamo il baseline (reset) con i nuovi cambiamenti
    final_updates = {**base_cfg_updates, **config_changes}
    
    # 4. Chiamiamo update_cfg_correct PASSANDO l'oggetto esistente 
    # affinché lo aggiorni senza ricaricare i pesi del modello.
    return update_cfg_correct(final_updates, preproc_obj)

def main():
    parser = argparse.ArgumentParser(description="🚀 Lancia la pipeline degli studi ablativi via YAML")
    parser.add_argument("--config", type=str, required=True, help="Percorso al file config.yaml")
    args = parser.parse_args()

    print(f"📄 Lettura configurazione da: {args.config}")
    cfg = load_config(args.config)

    base_dir = cfg.get("base_dir", "ablation_studies")
    n_runs = cfg.get("n_runs", 3)
    num_examples = cfg.get("num_examples", -1)
    force_reprocess = cfg.get("force_reprocess", False)
    skip_preprocessing = cfg.get("skip_preprocessing", False)
    run_vlm = cfg.get("run_vlm", True)
    backend = cfg.get("backend", "ollama")
    models_list = cfg.get("models", [])
    experiments = cfg.get("experiments", {})
    question_path = cfg.get("questions_path")
    annotations_path = cfg.get("annotations_path")
    images_dir = cfg.get("images_dir")

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

    system_prompt = "You are a multimodal assistant capable of understanding both visual and textual scene graphs. Use the image and the accompanying graph description to answer the question accurately."
    multimodal_prompt = "Answer the question based on the spatial configuration in the image and the graph description.\n\nQuestion: {question}"

    print("\n🤖 Inizializzazione del Preprocessor Globale (YOLO)...")
    preprocessor = update_cfg_correct({"detectors_to_use": ("yolov8", )})

    if not skip_preprocessing:
        print("\n" + "═"*50)
        print("🛠️  FASE 1: AVVIO PREPROCESSING DELLE IMMAGINI")
        print("═"*50)
        
        for exp_name, exp_data in experiments.items():
            ablation_grid = exp_data.get("ablation_grid")
            if not ablation_grid:
                continue
            
            # Aggiorna la configurazione DENTRO lo stesso oggetto
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
        print("\n⏭️  [Fase 1 SKIP] Preprocessing saltato da configurazione YAML.")

    if run_vlm:
        print("\n" + "═"*50)
        print("🧠  FASE 2: AVVIO INFERENZA ED EVALUATION SUI VLM")
        print("═"*50)
        
        for exp_name, exp_data in experiments.items():
            ablation_grid = exp_data.get("ablation_grid")
            if not ablation_grid:
                continue

            print(f"\n[Fase 2] Esecuzione modelli per: {exp_name.upper()}")
            run_ablation_experiments(
                experiment_name=exp_name,
                ablation_grid=ablation_grid,
                models_list=models_list,
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