"""Compare PyMuPDF CPU parser with PTX GPU parser on a sample PDF."""

from __future__ import annotations

import time

from knowledge3d.cranium.bridges.pdf_ingestion_bridge import PDFIngestionBridge


def run_parser(bridge: PDFIngestionBridge, pdf_path: str, label: str) -> dict:
    start = time.perf_counter()
    page = bridge.ingest_pdf_page(pdf_path, page_num=0)
    elapsed = (time.perf_counter() - start) * 1000.0
    print(f"[{label}] objects={page['object_count']}, time={elapsed:.2f} ms")
    return page


def main() -> None:
    pdf_path = "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to think/Algorithmic.Thinking.BASE.pdf"

    cpu_bridge = PDFIngestionBridge()
    cpu_bridge.enable_gpu_parser(False)
    cpu_page = run_parser(cpu_bridge, pdf_path, "CPU")

    gpu_bridge = PDFIngestionBridge()
    gpu_bridge.enable_gpu_parser(True)
    gpu_page = run_parser(gpu_bridge, pdf_path, "GPU")

    print("\nSample CPU text:")
    cpu_nodes = cpu_page["layout_graph"].get("nodes", [])
    for node in cpu_nodes[:3]:
        if node.get("type") == 1.0:
            idx = int(node.get("data_index", -1))
            if 0 <= idx < len(cpu_bridge._temp_text_storage):
                print("  •", cpu_bridge._temp_text_storage[idx][:120])
            break

    print("\nSample GPU text:")
    gpu_nodes = gpu_page["layout_graph"].get("nodes", [])
    for node in gpu_nodes[:3]:
        if node.get("type") == 1.0:
            idx = int(node.get("data_index", -1))
            if 0 <= idx < len(gpu_bridge._temp_text_storage):
                print("  •", gpu_bridge._temp_text_storage[idx][:120])
            break


if __name__ == "__main__":
    main()
