import cv2


print(cv2.__version__)

image = cv2.imread('resources/image/bitmex.png', cv2.IMREAD_COLOR)
cv2.imshow("bitmex", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
