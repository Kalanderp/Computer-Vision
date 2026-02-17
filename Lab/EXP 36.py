import cv2
import numpy as np
import os

# Main image path
main_img_path = r"D:\d\hj.png"

# Template image path (CHANGE THIS TO YOUR TEMPLATE LOCATION)
template_path = r"D:\d\watch_template.png"

# Check if files exist
if not os.path.isfile(main_img_path):
    print("Main image not found!")
    exit()

if not os.path.isfile(template_path):
    print("Template image not found! Check the path.")
    exit()

# Read images
img = cv2.imread(main_img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread(template_path, 0)

w, h = template.shape[::-1]

# Template matching
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.6
locations = np.where(result >= threshold)

detected = False

for pt in zip(*locations[::-1]):
    detected = True
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
    cv2.putText(img, "Watch Detected",
                (pt[0], pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

if not detected:
    print("Watch not detected. Try lowering threshold.")

cv2.imshow("Final Output - Watch Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()