import cv2
import numpy as np

root_resource = 'C:/GitHub/Swift_Video_Coloring/resources/image/'

image = cv2.imread(root_resource + 'bitmex.png', cv2.IMREAD_UNCHANGED)
cv2.imshow("BIT", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
