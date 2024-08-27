import cv2
import numpy as np
import imutils
import argparse
from paddleocr import PaddleOCR
from rembg import remove, new_session
import math


def find_rect_cnt(sorted_contours):
    '''finds the biggest contour similar to a rectangle'''
    for contour in sorted_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return contour
    return None


def draw_line(line,image):
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def avg_lines(lines, angle_threshold=(np.pi / 180) * 30, distance_threshold=50):
    '''merges similar lines'''
    if len(lines) == 0:
        return []
    final_lines = []
    used = [False] * len(lines)
    for i, line1 in enumerate(lines):
        if used[i]:
            continue
        rho1, theta1 = line1[0]
        similar_lines = [(rho1, theta1)]
        used[i] = True
        for j, line2 in enumerate(lines):
            if i != j and not used[j]:
                rho2, theta2 = line2[0]
                # Check if the lines are close in both angle and distance
                if abs(theta1 - theta2) < angle_threshold and abs(rho1 - rho2) < distance_threshold:
                    similar_lines.append((rho2, theta2))
                    used[j] = True
        rhos, thetas = zip(*similar_lines)
        avg_rho = np.mean(rhos)
        avg_theta = np.mean(thetas)
        final_lines.append([(avg_rho, avg_theta)])    
    return final_lines


def line_intersection(line1, line2):
    '''returns the point of intersection between two lines'''
    rho1, theta1 = line1
    rho2, theta2 = line2
    a1 = np.cos(theta1)
    b1 = np.sin(theta1)
    c1 = rho1
    a2 = np.cos(theta2)
    b2 = np.sin(theta2)
    c2 = rho2
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-10:
        return None
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return (int(round(x)), int(round(y)))


def find_intersections(lines, img_width, img_height):
    '''finds all intersections between lines in a list'''
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]
            intersection = line_intersection(line1, line2)
            if intersection:
                x, y = intersection
                if -50 <= x <= img_width+50 and -50 <= y <= img_height+50:
                    if img_width <= x <=img_width+50 : x=img_width-2
                    if  -50 <= x <=0 : x=1
                    if img_height <= y <=img_height+50 : y=img_height-2
                    if  -50 <= y <=0 : y=1
                    intersections.append((x,y))
    return intersections


def distance_between_points(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def two_most_distant_pairs(points, min_distance):
    '''
    finds the two most distant pairs of points(4 points in total) in a list of points
    it's used to determines the corners of a rectangle
    '''
    if len(points) < 4:
        raise ValueError("At least four points are required to find two distinct pairs")
    points = np.array(points)
    pairs = [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]
    pair_distances = [(distance_between_points(points[i], points[j]), (i, j)) for i, j in pairs]
    pair_distances.sort(reverse=True, key=lambda x: x[0])
    first_pair = pair_distances[0][1]
    first_pair_points = [tuple(points[first_pair[0]]), tuple(points[first_pair[1]])]
    second_pair = None
    for _, (i, j) in pair_distances[1:]:
        second_pair_points = [tuple(points[i]), tuple(points[j])]
        if (distance_between_points(first_pair_points[0], second_pair_points[0]) >= min_distance and
            distance_between_points(first_pair_points[0], second_pair_points[1]) >= min_distance and
            distance_between_points(first_pair_points[1], second_pair_points[0]) >= min_distance and
            distance_between_points(first_pair_points[1], second_pair_points[1]) >= min_distance):
            second_pair = (i, j)
            break
    if second_pair is None:
        raise ValueError("No valid second pair found that meets the minimum distance requirement")
    return first_pair_points , second_pair_points


def adapt_point(original_size, resized_size,point):
    '''adjusts points in a resized image to the original image'''
    original_width, original_height = original_size
    resized_width, resized_height = resized_size
    x_scale = original_width / resized_width
    y_scale = original_height / resized_height
    return (point[0] * x_scale, point[1] * y_scale)


def adapt_corners(original_size, resized_size, pair1, pair2):
    '''adjusts corners in a resized image to the original image'''
    adapted_pair1 = [adapt_point(original_size, resized_size,pair1[0]), adapt_point(original_size, resized_size,pair1[1])]
    adapted_pair2 = [adapt_point(original_size, resized_size,pair2[0]), adapt_point(original_size, resized_size,pair2[1])]
    return adapted_pair1, adapted_pair2


def order_points(pair1, pair2):
    '''organizes four points in the following order : top left, top right, bottom right, bottom left'''
    rect = np.zeros((4, 2), dtype="float32")
    points = np.array([pair1[0], pair1[1], pair2[0], pair2[1]])
    s = points.sum(axis=1)
    top_left_index = np.argmin(s)
    rect[0] = points[top_left_index]
    if top_left_index<2:
        if top_left_index==0:
            rect[2]=points[1]
        else:
            rect[2]=points[0]
        points=points[2:]
    else:
        if top_left_index==2:
            rect[2]=points[3]
        else:
            rect[2]=points[2]
        points=points[:-2]

    top_right_index = np.argmin(np.diff(points, axis=1))
    rect[1] = points[top_right_index]
    if top_right_index==0:
        rect[3]=points[1]
    else:
        rect[3]=points[0]
    return rect


def four_point_transform(image, couple1,couple2):
    '''applies four point tranform on a portion of a image determined by it's four corners'''
    card_aspect_ratio=0.63
    rect = order_points(couple1,couple2)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    larger_side=max(maxHeight,maxWidth)
    if larger_side==maxWidth:
        maxHeight=int(maxWidth*card_aspect_ratio)
    else:
        maxWidth=int(maxHeight*card_aspect_ratio)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def adapt_rect(original_size, resized_size,rect):
    original_width, original_height = original_size
    resized_width, resized_height = resized_size
    x_scale = original_width / resized_width
    y_scale = original_height / resized_height
    center = (int(rect[0][0] * x_scale), int(rect[0][1] * y_scale))
    size = (int(rect[1][0] * x_scale), int(rect[1][1] * y_scale))
    angle = rect[2]
    scaled_rect = (center, size, angle)
    return scaled_rect


def adapt_box(original_size, resized_size,box):
    original_width, original_height = original_size
    resized_width, resized_height = resized_size
    x_scale = original_width / resized_width
    y_scale = original_height / resized_height
    scaled_box = box.copy()
    scaled_box[:, 0] = (scaled_box[:, 0] * x_scale).astype(np.int64)
    scaled_box[:, 1] = (scaled_box[:, 1] * y_scale).astype(np.int64)
    return scaled_box


def crop_rect(image,box,width,height):
    '''crops out a rectangle determined by a box from a image'''
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def surface(points):
    '''
    returns the surface of the rectangle formed by 4 points
    '''
    if len(points) != 4:
        raise ValueError("Exactly four points are required")
    # Sort points in clockwise or counterclockwise order
    points = sorted(points, key=lambda p: (p[0], p[1]))
    centroid = (sum(p[0] for p in points) / 4, sum(p[1] for p in points) / 4)
    points = sorted(points, key=lambda p: (math.atan2(p[1] - centroid[1], p[0] - centroid[0])))

    # Calculate area using Shoelace formula
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    area = 0.5 * abs(
        x_coords[0]*y_coords[1] + x_coords[1]*y_coords[2] + x_coords[2]*y_coords[3] + x_coords[3]*y_coords[0]
        - y_coords[0]*x_coords[1] - y_coords[1]*x_coords[2] - y_coords[2]*x_coords[3] - y_coords[3]*x_coords[0]
    )
    return area


def largest_rect(result):
    '''returns the coordinates of the largest rectangle detected using OCR'''
    max_surface=0
    max_points=None
    for line in result:
        surf=surface(line[0])
        if surf>max_surface: 
            max_surface=surf
            max_points=line[0]
    return max_points


def isVerticle(points):
    '''checks if a rectangle is vertical (height bigger than width)'''
    p1=points[0]
    p3=points[2]
    x=p3[0]-p1[0]
    y=p3[1]-p1[1]
    if x<y: return True
    else: return False


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated_image


def avg_conf(result):
    '''returns average confidence score of the results obtained using ocr'''
    conf=[line[1][1] for line in result]
    return sum(conf)/len(conf)


def correct_orientation(warped,ocr_ar):
    '''corrects the image orientation if it's sideways or upside down'''
    rotated=warped.copy()
    result = ocr_ar.ocr(rotated)[0]
    if result == None: 
        rotated=rotate_image(rotated,45)
        result= ocr_ar.ocr(rotated)[0]
    if result == None:
        raise ValueError("Text not readable")
    largest=largest_rect(result)
    if isVerticle(largest) :
        rotated=rotate_image(warped,90)
    correct_angle=0
    max_conf=0
    for i in range (2):
        rot=rotate_image(rotated,i*180)
        res=ocr_ar.ocr(rot)[0] 
        if res is not None:
            conf=avg_conf(res)
            if max_conf<conf:
                max_conf=conf
                correct_angle=i*180
    rotated=rotate_image(rotated,correct_angle)
    return rotated


def box_rect(boxes):
    '''returns the coordinates of the bounding box of all the boxes, meaning the rectangle that includes all of them'''
    all_points = np.array(boxes).reshape(-1, 2)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    return x_min, y_min,x_max, y_max


def isCropped(image,ocr_ar):
    '''determins if the card is already cropped or not'''
    width = image.shape[0]
    height = image.shape[1]
    aspect_ratio=height/width
    if 0.61<=aspect_ratio<=0.65 or 1.538<=aspect_ratio<=1.64:
        try:
            rotated = correct_orientation(image,ocr_ar)
            result=ocr_ar.ocr(rotated)[0]
            boxes = [line[0] for line in result]
        except : 
            return False
        x_min, y_min,x_max, y_max = box_rect(boxes)

        rected=image.copy()
        cv2.rectangle(rected, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

        rect_area= (x_max - x_min) * (y_max - y_min)
        orig_area=width*height
        rect_ratio=rect_area/orig_area
        if rect_ratio>0.5: return True
        else : return False
    else : return False


def scan_card(image,ocr_ar,debug=False):
    
    # open a session for rembg(the tool used to remove the background
    session = new_session()
    orig = image.copy()
    # resize the image to better visualize the results
    image = imutils.resize(image, height=750)
    if debug==True:
        cv2.imshow("image", image)
        cv2.waitKey(0)
    # check if the card is already cropped 
    is_cropped=isCropped(image,ocr_ar)
    if is_cropped:
        if debug:
            print("image is already cropped")
        warped=orig
    else:
        if debug==True: 
            print("image is not cropped")

        # remove the backgroung using rembg
        cropped=remove(image, session=session)
        if debug==True:
            cv2.imshow("cropped",cropped)
            cv2.waitKey(0)
        
        # apply binary thresholding on the forth channel of the RGBA image
        # this displays the transparent part of the image(the removed background) in black and the opaque part(the foreground) in white
        alpha_channel = cropped[:, :, 3]
        _, binary = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
        if debug==True:
            cv2.imshow('binary', binary)
            cv2.waitKey(0)

        # dilate the image to remove the noise around the extracted foreground
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        if debug==True:
            cv2.imshow("dilated",dilated)
            cv2.waitKey(0)

        # draw countours around the extarcted foreground
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        contoured = cv2.cvtColor(image , cv2.COLOR_BGRA2BGR)
        cv2.drawContours(contoured, cnts, -1, (0, 255, 0), 2)  
        if debug==True:
            cv2.imshow('Contours', contoured)
            cv2.waitKey(0)

        # in case more than one object is selected in the foreground, we look for the closest one to a rectangle
        height, width, channels = image.shape
        rectangled= np.zeros((height, width, channels), dtype=np.uint8)
        rect=find_rect_cnt(cnts)

        # if no rectangle is found (for example if some noise is picked up along with the card)
        if rect is None:
            if debug:
                print("no rectangle found ")

            # draw a bounding rectangle around the biggest object in the foreground and crop it out
            cnt=cnts[0]
            contoured = cv2.cvtColor(image , cv2.COLOR_BGRA2BGR)
            cv2.drawContours(contoured, [cnt], -1, (0, 255, 0), 2)  
            if debug==True:
                cv2.imshow('Contour', contoured)
                cv2.waitKey(0)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)  
            rectangled=cv2.cvtColor(image , cv2.COLOR_BGRA2BGR)
            cv2.drawContours(rectangled, [box], 0, (0, 255, 0), 2)
            if debug==True:
                cv2.imshow('Bounding Box', rectangled)
                cv2.waitKey(0)

            # warped=crop_rect(image,box,width=int(rect[1][0]),height=int(rect[1][1]))
            box = adapt_box(orig.shape[:2],image.shape[:2],box)
            rect=adapt_rect(orig.shape[:2],image.shape[:2],rect)
            warped=crop_rect(orig,box,width=int(rect[1][0]),height=int(rect[1][1]))
            if debug==True:
                cv2.imshow('warped',warped)
                cv2.waitKey(0)
        
        # if we do find a rectangle( which would be the card)
        else :
            # use hough transform to draw lines on it's sides and determine it's corners
            cv2.drawContours(rectangled,[rect], -1, (0, 255,0), 2)
            if debug==True:
                cv2.imshow("rect",rectangled)
                cv2.waitKey(0)

            rectangled=cv2.cvtColor(rectangled, cv2.COLOR_BGR2GRAY)
            lines = cv2.HoughLines(rectangled, 1, np.pi / 180, 120)
            if lines is not None:
                lined=image.copy()
                for line in lines:
                    draw_line(line,lined)
            if debug==True:
                print("line count : ",len(lines))
                cv2.imshow("lined",lined)
                cv2.waitKey(0)

            avglines =avg_lines(lines,(np.pi / 180)*30, 50)
            if avglines is not None:
                avglined=image.copy()
                for line in avglines:
                    draw_line(line,avglined)
            if debug==True:
                print("avgline count : ",len(avglines))
                cv2.imshow("avglined",avglined)
                cv2.waitKey(0)

            inter=image.copy()
            intersections=find_intersections(avglines,image.shape[1],image.shape[0])
            for point in intersections:
                cv2.circle(inter, point, 5, (0, 0, 255), -1)
            if debug==True:
                print("intersection count : ",len(intersections))
                cv2.imshow("inter",inter)
                cv2.waitKey(0)

            cornered=image.copy()
            couple1,couple2=two_most_distant_pairs(intersections,80)
            for point in couple1:
                cv2.circle(cornered, point, 10, (0, 0, 255), -1)
            for point in couple2:
                cv2.circle(cornered, point, 10, (0, 0, 255), -1)
            if debug==True:
                print("corners : ",couple1,couple2)
                cv2.imshow("cornered",cornered)
                cv2.waitKey(0)

            # use the found corners to apply four point transform on the image
            adapted_couple1,adapted_couple2=adapt_corners(orig.shape[:2],image.shape[:2],couple1,couple2)
            warped = four_point_transform(orig,adapted_couple1,adapted_couple2)
            # warped = four_point_transform(image,couple1,couple2)
            if debug==True:
                cv2.imshow("warped",warped)
                cv2.waitKey(0)

    # correct the orientation of the cropped out image    
    rotated=correct_orientation(warped,ocr_ar)
    if debug:
        cv2.imshow("rotated",rotated)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return rotated



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="scan_card")
    parser.add_argument('--input', type=str, help='input path to image')
    args = parser.parse_args()

    ocr_ar=PaddleOCR(lang="ar")



    image = cv2.imread(args.input)
    final = scan_card(image,ocr_ar=ocr_ar,debug=True)
