import fitz  # PyMuPDF


# Функція для перетворення PDF в текст
def extract_text_from_pdf(pdf, output):
    doc = fitz.open(pdf)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()

    with open(output, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    pdf_file = "roman_empire.pdf"
    output_file = "roman_empire.txt"
    extract_text_from_pdf(pdf_file, output_file)
    print(f"Text extracted from {pdf_file} and saved to {output_file}")
