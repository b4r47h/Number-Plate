import cv2
import uuid
import pandas as pd
import ocr

def detect_plates(img, harcascade, min_area):
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
    return plates

def save_image(img_roi):
    uid = str(uuid.uuid4())
    image_path = "plates/scanned_img_" + uid + ".jpg"
    cv2.imwrite(image_path, img_roi)
    return uid, image_path

def display_image(img, plates):
    for (x,y,w,h) in plates:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.imshow("Result", img)

def main():
    harcascade = "model\haarcascade_russian_plate_number.xml"

    cap = cv2.VideoCapture(0)
    cap.set(3, 640) #w
    cap.set(4, 480) #h

    min_area = 500
    data = {'UUID': [], 'Image_Path': [], 'Plate_Text': []}

    while True:
        success, img = cap.read()
        plates = detect_plates(img, harcascade, min_area)

        for (x,y,w,h) in plates:
            area = w * h
            if area > min_area:
                img_roi = img[y: y+h, x:x+w]
                display_image(img, plates)

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    uid, image_path = save_image(img_roi)
                    plate_text = ocr.perform_ocr(image_path)
                    data['UUID'].append(uid)
                    data['Image_Path'].append(image_path)
                    data['Plate_Text'].append(plate_text)

        display_image(img, plates)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(data)
    excel_file = 'number_plates_data.xlsx'
    df.to_excel(excel_file, index=False)
    print("Data saved to Excel file:", excel_file)

if __name__ == "__main__":
    main()
