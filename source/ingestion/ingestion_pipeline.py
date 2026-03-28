import os
import json
from pypdf import PdfReader
from traffic_rag.source.chunking.hierarchical_splitter import HierarchicalLegalSplitter

def run_ingestion(pdf_path, output_json):
    print(f"Loading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
        
    doc_name = os.path.basename(pdf_path)
    splitter = HierarchicalLegalSplitter(doc_name)
    chunks = splitter.split_text(full_text)
    
    print(f"Created {len(chunks)} chunks.")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_json}")

if __name__ == "__main__":
    pdf_file = "/media/pphong/D:/Do_An_Tot_Nghiep/GitHub1/Retrieval_Information_System_With_Agentic_RAG-main/traffic_rag/data/traffic_law_qa.pdf"
    output_file = "/media/pphong/D:/Do_An_Tot_Nghiep/GitHub1/Retrieval_Information_System_With_Agentic_RAG-main/traffic_rag/data/traffic_chunks.json"
    run_ingestion(pdf_file, output_file)
