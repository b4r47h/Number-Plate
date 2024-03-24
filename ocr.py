import easyocr

def perform_ocr(image_path):
    reader = easyocr.Reader(['en'])
    output = reader.readtext(image_path)
    plate_text = ' '.join([item[1] for item in output])
    return plate_text
