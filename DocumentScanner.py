from scanner import scanner
import argparse
from paddleocr import PaddleOCR
from langdetect import detect
import Rectangle

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Example script to demonstrate command line argument parsing.')
    parser.add_argument('--input', type=str, help='Path to input image')
    args = parser.parse_args()

    img_path=args.input
    img=scanner(img_path)

    OCRout=""
    isGood=False

    ocr_ar = PaddleOCR(lang="ar")
    result = ocr_ar.ocr(img)[0]

    # correcting the orientation of a card, because "determine_deskew" can still return a wrong making the card vertical instead of horizantal
    # rotate the card from horizantal to vertical
    largest=Rectangle.largest_rect(result)
    if Rectangle.isVerticle(largest) :
        img=Rectangle.rotate_90(img)
        # cv2.imshow("rot90", img)
        # cv2.waitKey(0)
        result = ocr_ar.ocr(img)[0]
    
    # correct the card if it's upside down
    # the way it works is that we choose the word that has the highest confidence score, then we rotate it by 180 degrees 
    # and read it compare the new confidence score, if the new score is higher it means that the word was upside down and 
    # it should be rotated by 180 degrees
    # i choose to use arabic because it is better at reading text when it's upside down
    points=Rectangle.best_word(result)
    cropped=Rectangle.crop_rectangle(img,points)
    max_conf=0
    deg_correct=0
    for i in range(2):
        rot=Rectangle.rotate_image(cropped,i*180)
        res=ocr_ar.ocr(rot)[0] # reads the text in the image
        if res is not None :
            conf = Rectangle.average_confidence(res)
            if max_conf<conf:
                max_conf=conf
                deg_correct=i*180
    
    img=Rectangle.rotate_image(img,deg_correct)
    result = ocr_ar.ocr(img)[0]

    # extract text from image
    # in arabic
    for line in result:
        if line[1][1]>0.7:
            word = line[1][0]
            try:
                if detect(word[::-1])=="ar":
                    OCRout+=word[::-1] + "\n"
            except:
                continue

    # in french
    ocr_fr = PaddleOCR(lang="fr")
    result = ocr_fr.ocr(img)[0]
    for line in result:
        if line[1][1]>0.8:
            OCRout+= line[1][0] + "\n"

    # writes the output in a txt file
    with open("output.txt", 'w', encoding="utf-8") as file:
        file.write(OCRout)