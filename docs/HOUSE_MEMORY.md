# House Memory: Rooms, Books, and Garden

The **house metaphor** is the core unit of persistent memory. A house has rooms, shelves, a virtual computer, and a Knowledge Garden. As a conversation unfolds, the avatar walks through these rooms, retrieving and organizing knowledge objects.

## LLM Heads to K3D Nodes

K3D externalizes those rooms into a spatial scene. Head embeddings become **K3D nodes**, allowing long‑term memories to live outside the core model. Each node represents a concept/book/paper/file and lives on shelves or trees in the Garden. Doors are strictly intra‑house (room transitions), not network links.

## Offloading to Reduce Core Model Size

By offloading seldom‑used knowledge to the K3D house, the core model can remain lightweight. Instead of expanding parameters to remember everything, the model fetches relevant rooms from the spatial web when needed, reducing overall size while retaining access to rich context.

## LLM ↔ House (Internal) Flow

```mermaid
graph LR
    LLM["LLM Head Embeddings"] <--> House["House Memory\n(K3D Nodes)"]
    House --> Garden["Knowledge Garden"]

## On Doors and Condos
- Doors: internal room transitions (e.g., Library ↔ Workshop). They do not link to other houses.
- Condo: future multi‑house orchestration where a user may assemble several AI houses. Inter‑house navigation is a higher‑level network and is not modeled as in‑scene doors.

## Knowledge Objects
- Books and papers are placed in the Library.
- Files and a virtual computer desk live in the Study/Office.
- Trees of knowledge (ontology) grow in the Knowledge Garden.
```
