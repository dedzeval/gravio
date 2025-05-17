import fitz  # PyMuPDF
import sys

def extract_pdf_data(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)

    # Extract metadata
    metadata = doc.metadata

    # Extract visual text blocks from each page
    pages_text = []
    for page_number in range(len(doc)):
        page = doc[page_number]
        # "blocks" returns a list of tuples containing:
        # (x0, y0, x1, y1, "text", block_type, block_no)
        blocks = page.get_text("blocks")
        pages_text.append({
            "page": page_number + 1,
            "blocks": blocks
        })

    return metadata, pages_text

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_pdf.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        metadata, pages_text = extract_pdf_data(pdf_path)
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)

    # Print metadata
    print("PDF Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    # Print visual text information
    print("\nVisual Text Blocks:")
    for page_info in pages_text:
        print(f"\n--- Page {page_info['page']} ---")
        for block in page_info['blocks']:
            # Each block is a tuple: (x0, y0, x1, y1, text, block_type)
            x0, y0, x1, y1, text, block_type = block[:6]
            print(f"Block at ({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}) [Type {block_type}]:")
            print(text.strip())
            print("-" * 40)

if __name__ == "__main__":
    main()