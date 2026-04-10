#code for dynamic region of interest

import cv2
import numpy as np
import torch
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Load YOLOv5 model
MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def FocalLength(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):
    return (real_object_width * Focal_Length) / object_width_in_frame

# Mapping of class labels to known widths
class_width_mapping = {
    0: 1.5,  #class label 0 corresponds to a known width of 1.5 meters
    1: 1.3,
    2: 3.5,
    3: 1.3,
    4: 8,
    5: 6.2,  
    7: 5.8,
    15: 0.8,
    16: 1.5,
    19: 2,
    62:6.2
}

def object_data(frame):
    results = MODEL(frame)
    coords = results.xyxy[0].numpy()
    object_list = []
    for coord in coords:
        x1, y1, x2, y2, confidence, class_label = coord
        object_width = x2 - x1
        object_list.append((object_width, [x1, y1, x2, y2], confidence, int(class_label)))
    return object_list

def calculate_intersection_area(rect, poly,frame):
    rect_pts = cv2.boxPoints(rect)
    rect_pts = np.int0(rect_pts)
    poly_mask = np.zeros_like(frame)
    cv2.fillPoly(poly_mask, [poly], (255, 255, 255))
    rect_mask = np.zeros_like(frame)
    cv2.fillPoly(rect_mask, [rect_pts], (255, 255, 255))
    intersection_mask = cv2.bitwise_and(poly_mask, rect_mask)
    intersection_area = np.sum(intersection_mask) / 255
    total_rect_area = cv2.contourArea(rect_pts)
    return intersection_area / total_rect_area


def draw_lane(frame, direction, lane_width=105, lane_bottom_y=None):
    height, width = frame.shape[:2]
    
    # Adjust the lane height dynamically based on the average flow in x direction
    lane_height = 140  # Default lane height

    if direction in ['left', 'right']:
        lane_height = 110

    if lane_bottom_y is None:
        lane_bottom_y = height   # Default to bottom of the frame

    # Adjust lane width based on the provided parameter
    

    if direction == 'left':
        pt1 = (max(0, width // 2 - lane_width), lane_bottom_y)
        pt2 = (width // 2 - lane_width + int(lane_height / 3), lane_bottom_y - lane_height)
        pt3 = (width // 2 + lane_width + int(lane_height / 3), lane_bottom_y - lane_height)
        pt4 = (min(width - 1, width // 2 + lane_width), lane_bottom_y)
    elif direction == 'right':
        pt1 = (max(0, width // 2 - lane_width - int(lane_height / 3)), lane_bottom_y - lane_height)
        pt2 = (width // 2 - lane_width, lane_bottom_y)
        pt3 = (width // 2 + lane_width, lane_bottom_y)
        pt4 = (min(width - 1, width // 2 + lane_width - int(lane_height / 3)), lane_bottom_y - lane_height)
    else:  # straight
        pt1 = (max(0, width // 2 - lane_width), lane_bottom_y - lane_height)
        pt2 = (max(0, width // 2 - lane_width), lane_bottom_y)
        pt3 = (min(width - 1, width // 2 + lane_width), lane_bottom_y)
        pt4 = (min(width - 1, width // 2 + lane_width), lane_bottom_y - lane_height)

    points = np.array([pt1, pt2, pt3, pt4], np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=5)
    return points

def calculate_intersection_area(rect, poly, frame):
    # Convert rectangle points to a cv2.RotatedRect object
    # Since the rectangle is not rotated, the angle is 0
    rect_rotated = cv2.minAreaRect(rect)
    
    # Get the points of the rectangle from the RotatedRect object
    rect_pts = cv2.boxPoints(rect_rotated)
    rect_pts = rect_pts.astype(np.int32) # Updated line to resolve the deprecation warning
    
    # Create a mask for the rectangle
    rect_mask = np.zeros_like(frame)
    cv2.fillPoly(rect_mask, [rect_pts], (255, 255, 255))
    
    # Create a mask for the polygon (lane)
    poly_mask = np.zeros_like(frame)
    cv2.fillPoly(poly_mask, [poly], (255, 255, 255))
    
    # Find the intersection mask
    intersection_mask = cv2.bitwise_and(rect_mask, poly_mask)
    
    # Calculate the intersection area
    intersection_area = np.sum(intersection_mask) / 255
    
    # Calculate the total area of the lane
    total_lane_area = cv2.contourArea(poly)
    
    # Return the ratio of the intersection area to the total lane area
    return intersection_area / total_lane_area


def process_video(input_video):
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_frame_width = frame_width // 2  # Medium-sized frame
    new_frame_height = frame_height // 2
    # Output video settings
    out = cv2.VideoWriter('output_video1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (new_frame_width, new_frame_height))

    ret, prev_frame = cap.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    total = 0 
    collided = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:  # Skip every alternate frame
            continue

        # Resize frame for processing
        frame = cv2.resize(frame, (new_frame_width, new_frame_height))

        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Ensure the grayscale frames have the same dimensions and are single-channel images
        prev_frame_gray = cv2.resize(prev_frame_gray, (new_frame_width, new_frame_height))

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, current_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate the average flow in x direction
        avg_flow_x = np.mean(flow[..., 0])

        # Determine direction based on x component of flow
        if abs(avg_flow_x) > 5:  # Threshold to ignore minor movements
            if avg_flow_x > 0:
                direction = 'right'
            else:
                direction = 'left'
        else:
            direction = 'straight'

        # Draw the dynamic lane and get lane points
        lane_points = draw_lane(frame, direction)

        # Get object data
        detected_objects = object_data(frame)

        # Check for collision between object bounding boxes and lane lines
        for object_width_in_frame, Objects, confidence, class_label in detected_objects:
            x1, y1, x2, y2 = Objects
            object_box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)

            # Retrieve known width based on class label
            known_width = class_width_mapping.get(class_label, 1.5)

            Focal_length_dynamic = FocalLength(known_width, known_width, 182)
            Distance = Distance_finder(Focal_length_dynamic, known_width, object_width_in_frame)

            # is_collision = False
            # for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
            #     if cv2.pointPolygonTest(lane_points, (x, y), False) >= 0:
            #         is_collision = True
            #         break

            # for lx, ly in lane_points:
            #     if x1 <= lx <= x2 and y1 <= ly <= y2:
            #         is_collision = True
            #         break
           
            # if is_collision and Distance < 10:

                    # Calculate the intersection area with the lane
            intersection_area_ratio = calculate_intersection_area(object_box, lane_points, frame)

            # Check if the intersection area ratio is more than 25%
            if intersection_area_ratio > 0.40:
                # Collision detected, draw bounding box in red and print collision alert
                cv2.polylines(frame, [object_box], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.putText(frame, 'Collision Alert!', (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, f"Distance: {Distance:.2f}m", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                collided += 1
            else:
                # No collision, draw bounding box in green
                cv2.polylines(frame, [object_box], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f"Distance: {Distance:.2f}m", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            total += 1

        out.write(frame)  # Write frame to output video

        cv2.imshow('Dynamic Lane', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame_gray = current_frame_gray

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculate accuracy
    accuracy = (collided / total) * 100
    print("Accuracy: {:.2f}%".format(accuracy))

# Example usage
input_video = "D:\\AHMS\\Project_N\\vedio.mp4"  # Specify the path to your video file
process_video(input_video)