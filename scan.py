import cv2
import numpy as np
import imutils
import argparse
from paddleocr import PaddleOCR
import Rectangle
from deskew import determine_skew
import math


def order_points(pair1, pair2):
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


def correct_orientation(
        image: np.ndarray
) -> np.ndarray:
    '''
    rotates the image according to the angle determined by "determine_deskew"
    '''
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    if angle == None: return image
    
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=(0,0,0))


def four_point_transform(image, couple1,couple2):
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


def find_rect_cnt(sorted_contours):
    for contour in sorted_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return contour
    return None


def line_intersection(line1, line2):
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


def adapt_coordinates(original_size, resized_size, pair1, pair2):
    original_width, original_height = original_size
    resized_width, resized_height = resized_size
    x_scale = original_width / resized_width
    y_scale = original_height / resized_height
    def adapt_point(point):
        return (point[0] * x_scale, point[1] * y_scale)
    adapted_pair1 = [adapt_point(pair1[0]), adapt_point(pair1[1])]
    adapted_pair2 = [adapt_point(pair2[0]), adapt_point(pair2[1])]
    return adapted_pair1, adapted_pair2


def avg_conf(result):
    conf=[line[1][1] for line in result]
    return sum(conf)/len(conf)


def scan_card(image,ocr_fr,ocr_ar,debug=False):
    orig = image.copy()
    image = imutils.resize(image, height=750)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug==True:
        cv2.imshow("gray", gray)
        cv2.waitKey(0)
    
    blurThresh=29
    while True:
        try :
            if blurThresh>4: blurThresh-=4
            elif blurThresh>1: blurThresh=1
            else : 
                warped=correct_orientation(orig)
                break
            
            blurred = cv2.medianBlur(gray ,blurThresh)
            if debug==True:
                print("blur thresh : ",blurThresh)
                cv2.imshow("blurred",blurred)
                cv2.waitKey(0)

            thresh=20
            edged = cv2.Canny(blurred ,thresh, thresh*3, L2gradient = True)
            if debug==True:
                cv2.imshow("edged"+str(thresh),edged)
                cv2.waitKey(0)

            kernel = np.ones((5,5), np.uint8)
            dilated = cv2.dilate(edged, kernel, iterations=1)
            if debug==True:
                cv2.imshow("dilated"+str(thresh),dilated)
                cv2.waitKey(0)

            cnts, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            height, width, channels = image.shape
            contoured= np.zeros((height, width, channels), dtype=np.uint8)
            rect=find_rect_cnt(cnts)

            cv2.drawContours(contoured,[rect], -1, (0, 255,0), 2)
            if debug==True:
                cv2.imshow("rect",contoured)
                cv2.waitKey(0)

            contoured=cv2.cvtColor(contoured, cv2.COLOR_BGR2GRAY)
            lines = cv2.HoughLines(contoured, 1, np.pi / 180, 120)
            if lines is not None:
                lined=image.copy()
                for line in lines:
                    draw_line(line,lined)

            if debug==True:
                print("line count : ",len(lines))
                cv2.imshow("lined",lined)
                cv2.waitKey(0)
            
            # merge similar lines to reduce number of lines
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
            
            adapted_couple1,adapted_couple2=adapt_coordinates(orig.shape[:2],image.shape[:2],couple1,couple2)
            warped = four_point_transform(orig,adapted_couple1,adapted_couple2)
            if debug==True:
                cv2.imshow("warped",warped)
                cv2.waitKey(0)
            
            result = ocr_fr.ocr(warped,rec=False)[0]
            if result==None: 
                cv2.destroyAllWindows()
                continue
            else : 
                if debug:
                    print("good")
                break
        except Exception as e:
            cv2.destroyAllWindows()
            print(f"Exception: {str(e)}")
            continue
    
    rotated=warped.copy()
    result = ocr_ar.ocr(rotated)[0]
    if result == None: 
        rotated=Rectangle.rotate_image(rotated,45)
        result= ocr_ar.ocr(rotated)[0]
    if result == None:
        raise ValueError("Text not readable")
    largest=Rectangle.largest_rect(result)
    if Rectangle.isVerticle(largest) :
        rotated=Rectangle.rotate_90(warped)

    correct_angle=0
    max_conf=0
    for i in range (2):
        rot=Rectangle.rotate_image(rotated,i*180)
        res=ocr_ar.ocr(rot)[0] 
        if res is not None:
            conf=avg_conf(res)
            if max_conf<conf:
                max_conf=conf
                correct_angle=i*180
    
    rotated=Rectangle.rotate_image(rotated,correct_angle)
    if debug:
        cv2.imshow("rotated",rotated)
        cv2.waitKey(0)
    
    return rotated


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="scan_card")
    parser.add_argument('--input', type=str, help='input path to image')
    args = parser.parse_args()

    ocr_fr=PaddleOCR(lang="fr")
    ocr_ar=PaddleOCR(lang="ar")

    image = cv2.imread(args.input)
    final = scan_card(image,ocr_fr,ocr_ar,debug=False)
