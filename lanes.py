import cv2
import numpy as np
import matplotlib.pyplot as plt


def Canny(lane_image):
    gray=cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray, (5,5), 0)
    canny=cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(lane_image):
    # triangle=np.array([[(84, 537), (876, 530), (468, 251)]], dtype=np.int32)
    vertices = np.array([[(84, 537),
                          (876, 530),
                          (468, 251),]],
                        dtype=np.int32)
    mask=np.zeros_like(lane_image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image=cv2.bitwise_and(canny, mask)
    return masked_image

def display_lines(lane_image, lines):
    line_image=np.zeros_like(lane_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2=line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return line_imageo

#
# def make_coords(lane_image, lines_parameters):
#     slope, intercept = line_parameters
#     y1=image.shape[0]
#     y2=int(y1*(3/5))
#     x1=(y1-intercept)/slope
#     x2=(y2-intercept)/slope
#     return np.array([x1,y1,x2,y2])


# def average(lane_image, lines):
#     left_fit=[]
#     right_fit=[]
#     for line in lines:
#         x1,y1,x2,y2 = line.reshape(4)
#         parameters=np.polyfit((x1,x2), (y1,y2), 1)
#         slope=parameters[0]
#         intercept=parameters[1]
#         if slope>0:
#             left_fit.append((slope, intercept))
#         else:
#             right_fit.append((slope, intercept))
#     left_fit_average=np.average(left_fit, axis=0)
#     right_fit_average=np.average(right_fit, axis=0)
#     left_line=make_coords(image, left_fit_average)
#     right_line=make_coords(image, right_fit_average)
#     return np.array(left_line, right_line)


#
image=cv2.imread("lanes.jpg")
cv2.imshow("Intial", image)
lane_image=np.copy(image)
lane_image=cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB)
img_h, img_w = lane_image[0].shape[0], lane_image[0].shape[1]
canny=Canny(lane_image)
cropped_image=region_of_interest(canny)
cv2.imshow("canny", canny)
plt.imshow(canny)
# cv2.imshow("result" ,region_of_interest(canny))
cv2.imshow("masked", region_of_interest(canny))
lines=cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=5)
# #average_lines=average(lane_image, lines)
line_image=display_lines(lane_image, lines)
combo=cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow("result", combo)
cv2.waitKey(0)
plt.show()

'''
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, lane_image = cap.read()
    img_h, img_w = lane_image[0].shape[0], lane_image[0].shape[1]
    canny=Canny(lane_image)
    cropped_image=region_of_interest(canny)
    cv2.imshow("canny", canny)
    plt.imshow(canny)
    #cv2.imshow("result" ,region_of_interest(canny))
    cv2.imshow("masked", region_of_interest(canny))
    lines=cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=5)
    #average_lines=average(lane_image, lines)
    line_image=display_lines(lane_image, lines)
    combo=cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo)
    cv2.waitKey(1)
    '''
