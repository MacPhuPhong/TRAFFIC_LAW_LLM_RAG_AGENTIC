from pypdf import PdfReader
import sys

def inspect_pdf(file_path):
    reader = PdfReader(file_path)
    # Print first 5 pages to see structure
    for i in range(min(5, len(reader.pages))):
        print(f"--- PAGE {i+1} ---")
        print(reader.pages[i].extract_text())
        print("\n")

if __name__ == "__main__":
    inspect_pdf(sys.argv[1])
