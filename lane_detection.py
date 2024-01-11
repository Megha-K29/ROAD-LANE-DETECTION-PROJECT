




import cv2
import numpy as np

# Define a function to display detected lines on an image
def display_lines(image, lines):
   
    line_image = np.zeros_like(image)
    
    if lines is not None:
        for line in lines:
           
            x1, y1, x2, y2 = line.reshape(4)
            
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    
    return line_image

def canny(image):
   
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply the Canny edge detection algorithm to detect edges in the blurred image
    canny = cv2.Canny(blur, 50, 150)
    
    return canny

def region_of_interest(image):

    height, width = image.shape[:2]
    
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ], dtype=np.int32)
    
    mask = np.zeros_like(image)
    
    cv2.fillPoly(mask, polygons, 255)
    
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def averaged_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    
    for line in lines:
       
        x1, y1, x2, y2 = line.reshape(4)
        
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        
        slope = parameters[0]
        intercept = parameters[1]
        
        # Determine whether the line belongs to the left or right lane
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.mean(left_fit, axis=0) if left_fit else (0, 0)
    right_fit_average = np.mean(right_fit, axis=0) if right_fit else (0, 0)
    
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    
    return np.array([left_line, right_line])


prev_left_line = None
prev_right_line = None

def make_coordinates(image, line_parameters, min_slope=0.1):
    global prev_left_line, prev_right_line
    
    slope, intercept = line_parameters
    
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    
    if abs(slope) > min_slope:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        
       
        prev_left_line = [x1, y1, x2, y2] if slope < 0 else prev_left_line
        prev_right_line = [x1, y1, x2, y2] if slope >= 0 else prev_right_line
    else:

        x1, y1, x2, y2 = prev_left_line if slope < 0 else prev_right_line

    return np.array([x1, y1, x2, y2], dtype=np.int32)


image = cv2.imread('Road Lane Detection/test_img.jpg')
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)

lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)


averaged_lines = averaged_slope_intercept(lane_image, lines)

line_image = display_lines(lane_image, averaged_lines)


combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

cv2.imshow('result', combo_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Open a video file for processing
cap = cv2.VideoCapture('Road Lane Detection/test2.mp4')

while cap.isOpened():
   
    ret, frame = cap.read()
   
    if not ret:
        break

    
    canny_image = canny(frame)
    
    
    cropped_image = region_of_interest(canny_image)
    
   
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    
    averaged_lines = averaged_slope_intercept(frame, lines)
    
  
    line_image = display_lines(frame, averaged_lines)
    
   
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    
    cv2.imshow('result', combo_image)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
