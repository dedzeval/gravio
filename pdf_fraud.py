import fitz  # PyMuPDF
import sys

def extract_pdf_data(pdf_path):
    doc = fitz.open(pdf_path)
    metadata = doc.metadata
    pages_text = []
    for page_number in range(len(doc)):
        page = doc[page_number]
        # get_text("blocks") returns tuples: (x0, y0, x1, y1, text, block_type, block_no)
        blocks = page.get_text("blocks")
        pages_text.append({
            "page": page_number + 1,
            "blocks": blocks
        })
    return metadata, pages_text

def modify_last_block(pdf_path, output_path, custom_text):
    doc = fitz.open(pdf_path)
    # Get the last page and its blocks
    last_page = doc[-1]
    blocks = last_page.get_text("blocks")
    if not blocks:
        print("No text blocks found on the last page.")
        doc.close()
        return

    # Choose the last block from the last page
    last_block = blocks[-1]
    # The block is a tuple: (x0, y0, x1, y1, text, block_type, ...)
    rect = fitz.Rect(last_block[0], last_block[1], last_block[2], last_block[3])

    # Add a redaction annotation over the block to cover its content with a white fill.
    last_page.add_redact_annot(rect, fill=(1, 1, 1))
    last_page.apply_redactions()  # This removes the redacted content

    # Insert the custom text into the same rectangle.
    # Setting overlay=True forces the new text to be drawn over existing content.
    last_page.insert_textbox(
        rect,
        custom_text,
        fontname="helv",
        fontsize=12,
        color=(0, 0, 0),  # Explicitly set text color to black
        align=1,
        overlay=True
    )

    # Save the modified PDF to the desired output path
    doc.save(output_path)
    doc.close()
    print(f"Modified PDF saved as {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python modify_pdf.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = "./1.pdf"

    # Extract metadata and visual text blocks
    metadata, pages_text = extract_pdf_data(pdf_path)

    print("PDF Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    print("\nVisual Text Blocks:")
    for page_info in pages_text:
        print(f"\n--- Page {page_info['page']} ---")
        for block in page_info['blocks']:
            x0, y0, x1, y1, text, block_type = block[:6]
            print(f"Block at ({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}) [Type {block_type}]:")
            print(text.strip())
            print("-" * 40)

    # Set the custom text for the last block.
    # (For example, replacing "Дата проведення\n07.03.2025\n..." with a new string.)
    custom_text = """Дата проведення
19.03.2025

Сума: 110 999,00 грн"""

    # Modify the last block in the PDF and store the modified PDF
    modify_last_block(pdf_path, output_path, custom_text)

if __name__ == "__main__":
    main()