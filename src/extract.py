import os
from typing import List, Dict, Tuple
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEMP_DIR = os.path.join(ROOT, 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)


class DocumentProcessor:
    """Extract text paragraphs and embedded images (with OCR) from PDFs."""

    def __init__(self):
        self.text_chunks: List[str] = []
        self.image_chunks: List[Dict] = []
        self.metadata: List[Dict] = []

    def extract_from_pdfs(self, pdf_paths: List[str]) -> Tuple[List[str], List[Dict], List[Dict]]:
        for pdf_path in pdf_paths:
            print(f"Processing: {pdf_path}")
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text and chunk into paragraphs
                text = page.get_text()
                if text and text.strip():
                    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
                    for para in paragraphs:
                        self.text_chunks.append(para)
                        self.metadata.append({
                            'type': 'text',
                            'source': os.path.basename(pdf_path),
                            'page': page_num + 1
                        })

                # Extract images from page
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image.get('image')

                        # Save a temporary image file
                        img_filename = f"img_{os.path.basename(pdf_path)}_p{page_num+1}_{img_index}.png"
                        img_path = os.path.join(TEMP_DIR, img_filename)
                        with open(img_path, 'wb') as f:
                            f.write(image_bytes)

                        pil_img = Image.open(img_path).convert('RGB')
                        ocr_text = pytesseract.image_to_string(pil_img)

                        self.image_chunks.append({'path': img_path, 'ocr_text': ocr_text, 'image': pil_img})
                        self.metadata.append({
                            'type': 'image',
                            'source': os.path.basename(pdf_path),
                            'page': page_num + 1,
                            'index': img_index
                        })
                    except Exception:
                        # ignore images we cannot extract
                        continue

            doc.close()

        print(f"Extracted {len(self.text_chunks)} text chunks and {len(self.image_chunks)} images")
        return self.text_chunks, self.image_chunks, self.metadata


if __name__ == '__main__':
    # quick test runner
    data_dir = os.path.join(ROOT, 'data', 'pdfs')
    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print('No PDFs found in data/pdfs. Place your PDFs there and re-run.')
    else:
        proc = DocumentProcessor()
        proc.extract_from_pdfs(pdf_files)
