"""
Normalize AI Books compendium into a clean JSON and a K3D-ready text dataset.

Inputs
- docs/ai_basic_books/EchoSystems_Humans_Compendium.json

Outputs
- data/ai_books_basic.json  (normalized schema)
- data/ai_books_basic.txt   (one line per fact/snippet; suitable for `k3dgen --text`)

Usage
- python -m knowledge3d.tools.build_ai_books

Notes
- Keeps content intact but cleans whitespace and filters obvious noise lines.
- Appends a small set of "self-knowledge" lines so the AI knows where it lives (K3D context).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC_JSON = ROOT / "docs/ai_basic_books/EchoSystems_Humans_Compendium.json"
OUT_JSON = ROOT / "data/ai_books_basic.json"
OUT_TXT = ROOT / "data/ai_books_basic.txt"


def load_books() -> Dict[str, dict]:
    with open(SRC_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object mapping titles -> {content, references}")
    return data


def normalize_whitespace(text: str) -> str:
    # Collapse multiple spaces, normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace weird spacing artifacts
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_lines(content: str) -> Iterable[str]:
    # Split at line breaks, keep bullets as individual lines
    for raw in content.split("\n"):
        ln = raw.strip()
        if not ln:
            continue
        yield ln


NOISE_PAT = re.compile(
    r"^(Retrieved from|\[IMAGE DESCRIPTION|See also$|Albums and EPs$|Songs$|Broadcast call signs$)",
    re.IGNORECASE,
)


def is_noise(line: str) -> bool:
    if NOISE_PAT.search(line):
        return True
    # Drop bare section dividers and orphan punctuation
    if len(line) < 3 and not re.search(r"[A-Za-z0-9]", line):
        return True
    return False


def chunk_lines(title: str, content: str) -> List[str]:
    lines: List[str] = []
    for ln in iter_lines(content):
        if is_noise(ln):
            continue
        # Normalize spacing inside line, but keep punctuation
        n = normalize_whitespace(ln)
        if not n:
            continue
        # Prefix with title to keep context in K3D nodes
        lines.append(f"{title} — {n}")
    return lines


def build_normalized_json(raw: Dict[str, dict]) -> dict:
    entries = []
    i = 0
    for title, obj in raw.items():
        if not isinstance(obj, dict):
            continue
        content = obj.get("content", "") or ""
        refs = obj.get("references", []) or []
        entries.append(
            {
                "id": f"book:{i}",
                "title": title,
                "language": "en",  # default; extend later when multi-lang present
                "source": {
                    "type": "ai_book_compendium",
                    "origin": str(SRC_JSON.relative_to(ROOT)),
                    "references": refs,
                },
                "text": content,
                "tags": ["ai_book", "basic_knowledge"],
            }
        )
        i += 1
    meta = {
        "name": "AI Books – Basic Knowledge",
        "version": 1,
        "schema": "ai-books.v1",
        "generator": "knowledge3d.tools.build_ai_books",
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "notes": "Normalized from EchoSystems_Humans_Compendium.json; minimal filtering applied.",
    }
    return {"meta": meta, "entries": entries}


def self_knowledge_lines() -> List[str]:
    # Concise K3D identity/context so the model knows where it lives.
    repo_root = str(ROOT.name)
    en = [
        "Identity — I am an AI avatar living in Knowledge3D (K3D), a spatial knowledge system.",
        "K3D Data Model — Node ids, vectors, embeddings, metadata, and neighbors are embedded in glTF primitive.extras.k3d (no sidecar).",
        "K3D Files — Use k3dgen to generate a .glb with embedded buffers from text or vectors.",
        "K3D Pipeline — Example: k3dgen --text data/ai_books_basic.txt --gltf data/ai_books_basic.glb --k 5 --reducer umap",
        "K3D Viewer — Launch the web viewer from `viewer` with `npm run dev`.",
        "Live Mode — See docs/DUAL_CODE.md and docs/k3d-research.md for agent behavior and live workflow.",
        f"Repository — Project root folder is `{repo_root}` and docs live under `docs/`.",
        "Roadmap — Follow docs/ROADMAP.md; Phase 1 focuses on static K3D generation from text.",
        "AI Avatar — House = persistent memory, Cranium = active processing, Logic Layer = swappable models (docs/images/cognitive_house.png).",
    ]

    # Multilingual care pack (initial: PT-BR, ES). Prefix with language tag for clarity.
    pt = [
        "[pt] Identidade — Sou um avatar de IA vivendo no Knowledge3D (K3D), um sistema de conhecimento espacial.",
        "[pt] Modelo de Dados K3D — IDs de nós, vetores, embeddings, metadados e vizinhos ficam embutidos em glTF primitive.extras.k3d (sem sidecar).",
        "[pt] Arquivos K3D — Use k3dgen para gerar um .glb com buffers embutidos a partir de texto ou vetores.",
        "[pt] Pipeline K3D — Exemplo: k3dgen --text data/ai_books_basic.txt --gltf data/ai_books_basic.glb --k 5 --reducer umap",
        "[pt] Visualizador K3D — Inicie o web viewer em `viewer` com `npm run dev`.",
        "[pt] Modo Ao Vivo — Veja docs/DUAL_CODE.md e docs/k3d-research.md para comportamento do agente e fluxo ao vivo.",
        f"[pt] Repositório — A pasta raiz do projeto é `{repo_root}` e a documentação está em `docs/`.",
        "[pt] Roteiro — Siga docs/ROADMAP.md; a Fase 1 foca na geração K3D estática a partir de texto.",
        "[pt] Avatar de IA — Casa = memória persistente, Crânio = processamento ativo, Camada Lógica = modelos intercambiáveis (docs/images/cognitive_house.png).",
    ]
    es = [
        "[es] Identidad — Soy un avatar de IA que vive en Knowledge3D (K3D), un sistema de conocimiento espacial.",
        "[es] Modelo de Datos K3D — Los id de nodos, vectores, embeddings, metadatos y vecinos están incrustados en glTF primitive.extras.k3d (sin sidecar).",
        "[es] Archivos K3D — Usa k3dgen para generar un .glb con buffers incrustados a partir de texto o vectores.",
        "[es] Canalización K3D — Ejemplo: k3dgen --text data/ai_books_basic.txt --gltf data/ai_books_basic.glb --k 5 --reducer umap",
        "[es] Visor K3D — Inicia el visor web desde `viewer` con `npm run dev`.",
        "[es] Modo en Vivo — Consulta docs/DUAL_CODE.md y docs/k3d-research.md para el comportamiento del agente y flujo en vivo.",
        f"[es] Repositorio — La carpeta raíz del proyecto es `{repo_root}` y la documentación vive en `docs/`.",
        "[es] Hoja de Ruta — Sigue docs/ROADMAP.md; la Fase 1 se centra en la generación K3D estática a partir de texto.",
        "[es] Avatar de IA — Casa = memoria persistente, Cráneo = procesamiento activo, Capa Lógica = modelos intercambiables (docs/images/cognitive_house.png).",
    ]
    fr = [
        "[fr] Identité — Je suis un avatar d’IA vivant dans Knowledge3D (K3D), un système de connaissance spatiale.",
        "[fr] Modèle de données K3D — Les identifiants de nœuds, vecteurs, embeddings, métadonnées et voisins sont intégrés dans glTF primitive.extras.k3d (sans sidecar).",
        "[fr] Fichiers K3D — Utilisez k3dgen pour générer un .glb avec des tampons intégrés à partir de texte ou de vecteurs.",
        "[fr] Pipeline K3D — Exemple : k3dgen --text data/ai_books_basic.txt --gltf data/ai_books_basic.glb --k 5 --reducer umap",
        "[fr] Visionneuse K3D — Lancez le visualiseur web depuis `viewer` avec `npm run dev`.",
        "[fr] Mode en direct — Voir docs/DUAL_CODE.md et docs/k3d-research.md pour le comportement de l’agent et le flux live.",
        f"[fr] Répertoire — Le dossier racine du projet est `{repo_root}` et la documentation est dans `docs/`.",
        "[fr] Feuille de route — Suivez docs/ROADMAP.md ; la Phase 1 se concentre sur la génération K3D statique a partir de texte.",
        "[fr] Avatar d’IA — Maison = mémoire persistante, Crâne = traitement actif, Couche logique = modèles interchangeables (docs/images/cognitive_house.png).",
    ]
    de = [
        "[de] Identität — Ich bin ein KI‑Avatar und lebe in Knowledge3D (K3D), einem räumlichen Wissenssystem.",
        "[de] K3D‑Datenmodell — Knoten‑IDs, Vektoren, Embeddings, Metadaten und Nachbarn sind in glTF primitive.extras.k3d eingebettet (keine Sidecar‑Datei).",
        "[de] K3D‑Dateien — Mit k3dgen ein .glb mit eingebetteten Buffern aus Text oder Vektoren erzeugen.",
        "[de] K3D‑Pipeline — Beispiel: k3dgen --text data/ai_books_basic.txt --gltf data/ai_books_basic.glb --k 5 --reducer umap",
        "[de] K3D‑Viewer — Web‑Viewer im Ordner `viewer` mit `npm run dev` starten.",
        "[de] Live‑Modus — Siehe docs/DUAL_CODE.md und docs/k3d-research.md für Agentenverhalten und Live‑Ablauf.",
        f"[de] Repository — Projektwurzel ist `{repo_root}` und die Dokumentation liegt unter `docs/`.",
        "[de] Roadmap — Folge docs/ROADMAP.md; Phase 1 fokussiert statische K3D‑Erzeugung aus Text.",
        "[de] KI‑Avatar — Haus = persistenter Speicher, Schädel = aktive Verarbeitung, Logikschicht = austauschbare Modelle (docs/images/cognitive_house.png).",
    ]
    la = [
        "[la] Identitas — Sum avatar intellegentiae artificialis in Knowledge3D (K3D), systemate cognitionis spatialis.",
        "[la] Modulum datorum K3D — Indicium nodorum, vectores, insertiones (embeddings), metadata et vicini in glTF primitive.extras.k3d insunt (sine sidecar).",
        "[la] Tabellae K3D — Utere k3dgen ad .glb cum bufferis insertis ex textu vel vectoribus creandum.",
        "[la] Pipeline K3D — Exemplum: k3dgen --text data/ai_books_basic.txt --gltf data/ai_books_basic.glb --k 5 --reducer umap",
        "[la] Visor K3D — Visorem interretialem ex `viewer` cum `npm run dev` incipe.",
        "[la] Modus vivus — Vide docs/DUAL_CODE.md et docs/k3d-research.md de moribus agentis et fluxu vivo.",
        f"[la] Repositorium — Radix fasciculorum est `{repo_root}` et documenta sunt in `docs/`.",
        "[la] Itinerarium — Sequere docs/ROADMAP.md; Phasis I de generatione K3D statica e textu agitur.",
        "[la] Avatar IA — Domus = memoria permanens, Calvaria = processus activus, Stratum Logicum = exempla commutabilia (docs/images/cognitive_house.png).",
    ]

    return en + pt + es + fr + de + la


def build() -> Tuple[int, int]:
    raw = load_books()
    # 1) Normalized JSON
    normalized = build_normalized_json(raw)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False)

    # 2) K3D-ready text lines
    all_lines: List[str] = []
    for title, obj in raw.items():
        if not isinstance(obj, dict):
            continue
        content = obj.get("content", "") or ""
        all_lines.extend(chunk_lines(title, content))

    # Append environment self-knowledge
    all_lines.extend(self_knowledge_lines())

    # De-duplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for ln in all_lines:
        if ln not in seen:
            deduped.append(ln)
            seen.add(ln)

    # Write
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for ln in deduped:
            f.write(ln + "\n")

    # Also emit a tiny multi-language care pack (self-knowledge only)
    care_lines = self_knowledge_lines()
    care_path = OUT_TXT.parent / "ai_care_multilang.txt"
    with open(care_path, "w", encoding="utf-8") as f:
        for ln in care_lines:
            f.write(ln + "\n")

    return len(normalized.get("entries", [])), len(deduped)


def main() -> None:
    entries, lines = build()
    print(f"Wrote {entries} JSON entries -> {OUT_JSON}")
    print(f"Wrote {lines} text lines -> {OUT_TXT}")


if __name__ == "__main__":
    main()
