import re
from typing import List, Dict

class HierarchicalLegalSplitter:
    def __init__(self, document_name: str):
        self.document_name = document_name
        # Regex patterns for Vietnamese legal structure
        self.re_chuong = re.compile(r'^(Chương\s+[IVXLCDM]+.*)$', re.MULTILINE)
        self.re_dieu = re.compile(r'^(Điều\s+\d+[\.\s].*)$', re.MULTILINE)
        self.re_khoan = re.compile(r'^(\d+)\.\s+(.*)$', re.MULTILINE)
        self.re_diem = re.compile(r'^([a-z])\)\s+(.*)$', re.MULTILINE)
        self.re_cau_hoi = re.compile(r'^(Câu\s+\d+[\.\s].*)$', re.MULTILINE)

    def split_text(self, text: str) -> List[Dict]:
        chunks = []
        current_chuong = "N/A"
        current_dieu = "N/A"
        current_cau = "N/A"
        
        lines = text.splitlines()
        current_chunk_text = []
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Check for high-level structures to update metadata
            if self.re_chuong.match(line):
                current_chuong = line
                continue
            
            if self.re_dieu.match(line):
                # Save previous Article if exists as a general chunk or flush
                current_dieu = line
                continue
                
            if self.re_cau_hoi.match(line):
                current_cau = line
                continue

            # Check for Khoản (Clause)
            khoan_match = self.re_khoan.match(line)
            if khoan_match:
                khoan_num = khoan_match.group(1)
                chunks.append({
                    "content": line,
                    "metadata": {
                        "ten_van_ban": self.document_name,
                        "chuong": current_chuong,
                        "dieu": current_dieu,
                        "khoan": khoan_num,
                        "diem": "N/A",
                        "cau_hoi": current_cau,
                        "type": "khoan"
                    }
                })
                continue
                
            # Check for Điểm (Point)
            diem_match = self.re_diem.match(line)
            if diem_match:
                diem_let = diem_match.group(1)
                # Attach to last chunk if it was a Khoản
                chunks.append({
                    "content": line,
                    "metadata": {
                        "ten_van_ban": self.document_name,
                        "chuong": current_chuong,
                        "dieu": current_dieu,
                        "khoan": chunks[-1]["metadata"]["khoan"] if chunks else "N/A",
                        "diem": diem_let,
                        "cau_hoi": current_cau,
                        "type": "diem"
                    }
                })
                continue

            # For general text, append to the last specific chunk or create a general one
            if chunks:
                chunks[-1]["content"] += " " + line
            else:
                chunks.append({
                    "content": line,
                    "metadata": {
                        "ten_van_ban": self.document_name,
                        "chuong": current_chuong,
                        "dieu": current_dieu,
                        "khoan": "N/A",
                        "diem": "N/A",
                        "cau_hoi": current_cau,
                        "type": "general"
                    }
                })
                
        return chunks
