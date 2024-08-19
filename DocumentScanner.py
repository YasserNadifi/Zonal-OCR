from scan import scan_card
import cv2
import argparse
from paddleocr import PaddleOCR
from langdetect import detect
from llmchat import llmchat
import json
import re


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DocumentScanner.')
    parser.add_argument('--input', type=str, help='Path to input image')
    args = parser.parse_args()

    image=cv2.imread(args.input)

    ocr_ar = PaddleOCR(lang="ar")
    ocr_fr = PaddleOCR(lang="fr")

    image=scan_card(image,ocr_fr,ocr_ar,debug=False)

    result = ocr_ar.ocr(image)[0]

    # extract text from image
    # in arabic
    OCRout_ar=""
    for line in result:
        if line[1][1]>0.7:
            word = line[1][0]
            try:
                if detect(word[::-1])=="ar":
                    OCRout_ar+=word[::-1] + " "
            except:
                continue

    # in french
    OCRout_fr=""
    result = ocr_fr.ocr(image)[0]
    for line in result:
        if line[1][1]>0.8:
            OCRout_fr+= line[1][0] + " "
    
    with open('config.json', 'r') as file:
        config = json.load(file)
        API_URL = config['api_url']
        API_KEY = config['api_key']
        model_id =config['model_id']
        bot=llmchat(API_URL,API_KEY,model_id)
        
        prompt=config['prompt_doc_type']+OCRout_fr
        type=int(bot.send_prompt(prompt))
        if type==1: tags=json.dumps(config['prompt_id_tag']) 
        elif type==2 : tags=json.dumps(config['prompt_permis_tag'])
        prompt=config['prompt_seperate_tags']+tags
        bot.send_prompt(prompt)
        prompt=config['prompt_seperate_tags_ar']+OCRout_ar
        result=bot.send_prompt(prompt)
        
        final_response=re.search(r'\{.*\}', result, re.DOTALL).group()
    
    print(final_response)
    

