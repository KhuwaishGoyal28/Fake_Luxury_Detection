
import cv2
import requests

url = "http://127.0.0.1:5000/predict"
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    cv2.imshow("Press 's' to scan, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        img_path = "captured_product.jpg"
        cv2.imwrite(img_path, frame)
        print("Image saved! Sending for verification...")

        files = {'file': open(img_path, 'rb')}
        data = {"product_name": "Gucci Bag", "brand": "Gucci"}

        response = requests.post(url, files=files, data=data)
        print(response.json())

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
