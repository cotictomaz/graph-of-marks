import os
import sys

# 2. Diciamo a Python dove trovare il codice sorgente della repository
# Usiamo il percorso relativo alla cartella in cui ti trovi
sys.path.append(os.path.abspath("src"))

from gom.vqa.runner import run_vqa
from gom.vqa.types import VQAExample


# 1. Definiamo il Mock Model per fare il debug su CPU senza occupare RAM
class MockVLModel:
    def __init__(self):
        print("Mock Model inizializzato correttamente in locale su CPU.")

    def generate(self, prompt: str, image_path: str = None) -> str:
        """Simula l'inferenza del modello stampando il prompt generato da Graph of Marks"""
        print("\n====================================================")
        print("--- PROMPT RICEVUTO DAL MODELLO (CON GRAFO SPAZIALE) ---")
        print(prompt)
        print("====================================================\n")
        return "Risposta simulata dal modello locale."


# 3. Crea una cartella locale per salvare le immagini preprocessate con i "Marks"
os.makedirs("preprocessed_images", exist_ok=True)

# 4. Configura l'immagine di test
percorso_immagine_reale = "/home/tomaz_cotic/BDTM/graph-of-marks/images/vqav2_sample.png"

if not os.path.exists(percorso_immagine_reale):
    print(f"⚠️ ATTENZIONE: Il file '{percorso_immagine_reale}' non esiste!")

examples = [
    VQAExample(
        image_path=percorso_immagine_reale,
        question="Quali oggetti ci sono nell'immagine e come sono posizionati?",
    )
]

# 5. Inizializziamo il modello fantoccio e lanciamo la pipeline
model = MockVLModel()

print("Avvio della pipeline Graph of Marks in corso...")

results = run_vqa(
    examples=examples,
    model=model,
    out_json="risultati_locali.json",       # Salverà l'output in un file JSON nella cartella corrente
    prompt_tpl="Domanda: {question}\nRisposta:",
    preproc_folder="preprocessed_images",
    include_scene_graph=True,                # Forza l'estrazione delle relazioni spaziali (Graph of Marks)
)

print("\n--- PIPELINE ESEGUITA CON SUCCESSO SU CPU! ---")
print("Risultati registrati:", results)