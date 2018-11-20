import flower_detection
import predict

img_path = input("Enter the image filename that contains the flower: ")

cropped = flower_detection.crop(img_path)

predict.predict(cropped, 4)

