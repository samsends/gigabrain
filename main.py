import pytesseract
import pdf2image
import pdftotext
import os


def extract_pdf_text(pdf_path):
    """Extracts the text from a PDF file.

    Args:
      pdf_path: The path to the PDF file.

    Returns:
      The text from the PDF file, or OCR text if there is no text.
    """

    # Try to read the text from the PDF file.
    pdf_text = read_pdf_text(pdf_path)

    # If the PDF file has no text, perform OCR on the PDF file.
    if pdf_text is None or pdf_text.isspace():
        pages_as_images = pdf2image.convert_from_path(pdf_path)
        ocr_output = []
        for page_as_image in pages_as_images:
            ocr_output.append(pytesseract.image_to_string(page_as_image))
        pdf_text = "\n\n".join(ocr_output)

    return pdf_text


def read_pdf_text(pdf_path):
    """Reads the text from a PDF file.

    Args:
      pdf_path: The path to the PDF file.

    Returns:
      The text from the PDF file, or None if there is no text.
    """
    # Open the PDF file.
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = pdftotext.PDF(pdf_file)

    pdf_text = "\n\n".join(pdf_reader)

    return pdf_text


def main():
    input_folder = "inputs"
    output_folder = "outputs"

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            txt_filename = f"{filename[:-4]}.txt"
            txt_path = os.path.join(output_folder, txt_filename)

            extracted_text = extract_pdf_text(pdf_path)

            with open(txt_path, "w") as output_text_file:
                output_text_file.write(extracted_text)


if __name__ == "__main__":
    main()
