import os
import ollama
from typing import Optional

class OllamaVLM:
    """
    Wrapper per i modelli Vision-Language gestiti tramite server locale Ollama.
    
    Questa classe fornisce un'interfaccia unificata per comunicare con modelli 
    come Qwen, Llama-Vision o Moondream. Supporta l'iniezione del System Prompt,
    fondamentale per replicare le configurazioni SOTA descritte nel paper.
    """
    
    def __init__(self, model_name: str = "qwen2.5vl:3b", system_prompt: str = ""):
        """
        Inizializza il wrapper del modello.
        
        Args:
            model_name (str): Il tag esatto del modello su Ollama (es. "moondream", "qwen2.5vl:3b").
            system_prompt (str): Istruzioni di sistema da passare al modello prima della domanda.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        print(f"[OllamaVLM] Modello inizializzato: {self.model_name}")

    def generate(self, prompt: str, image_path: str) -> str:
        """
        Genera una risposta testuale data un'immagine e un prompt.
        
        Args:
            prompt (str): La domanda dell'utente (che includerà anche il grafo testuale).
            image_path (str): Il percorso sul disco dell'immagine pre-processata.
            
        Returns:
            str: La risposta generata dal modello.
        """
        if not os.path.exists(image_path):
            print(f"[OllamaVLM Errore] Immagine non trovata: {image_path}")
            return "Error: Image not found"

        # Ollama richiede le immagini sotto forma di byte
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
            print(f"[OllamaVLM Errore] Impossibile leggere l'immagine: {e}")
            return "Error reading image"

        # Prepariamo la coda dei messaggi rispettando i ruoli
        messages = []
        
        if self.system_prompt:
            messages.append({
                'role': 'system',
                'content': self.system_prompt
            })

        messages.append({
            'role': 'user',
            'content': prompt,
            'images': [img_bytes]
        })

        try:
            # Chiamata sincrona al server Ollama locale
            response = ollama.chat(model=self.model_name, messages=messages)
            
            # Estraiamo e restituiamo solo il testo puro generato
            return response['message']['content']
            
        except Exception as e:
            print(f"[OllamaVLM Errore] Fallimento durante l'inferenza: {e}")
            return f"Error during inference {e}"