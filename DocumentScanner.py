from ImgProcess import scan_card
import cv2
import argparse
from paddleocr import PaddleOCR
from langdetect import detect
from llmchat import llmchat
import json
import re


def last_int(s):
    matches = re.findall(r'\d+', s)
    if matches:
        return int(matches[-1])
    else:
        return None


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='DocumentScanner.')
    parser.add_argument('--input', type=str, help='Path to input image')
    args = parser.parse_args()

    image=cv2.imread(args.input)

    ocr_ar = PaddleOCR(lang="ar")
    ocr_fr = PaddleOCR(lang="fr")

    # scan the card from the input image
    image=scan_card(image,ocr_ar,debug=False)

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
    
    print("ocr ar : ",OCRout_ar)

    # in french
    OCRout_fr=""
    result = ocr_fr.ocr(image)[0]
    for line in result:
        if line[1][1]>0.8:
            OCRout_fr+= line[1][0] + " "
    print("ocr fr : ",OCRout_fr)

    # load the config.json
    with open('config2.json', 'r') as file:
        config = json.load(file)
    
    API_URL = config['api_url']
    API_KEY = config['api_key']
    model_id =config['model_id']

    # create a instance of llmchat
    # this object allows us to communicate with the llm and store both user prompts and llm responses, which allows for conversationnal memory
    bot=llmchat(API_URL,API_KEY,model_id)
    
    # first prompt is to determine the type of the document
    prompt1=config['prompt_doc_type']
    prompt1=prompt1.replace("${ocr_fr}",OCRout_fr)
    response1=bot.send_prompt(prompt1)

    print("\ndoc type response: ",response1)
    type=last_int(response1)

    if type==1: tags=config['prompt_id_tag']
    elif type==2 : tags=config['prompt_permis_tag']

    # the second prompt is to seperate the french ocr text into tags and structure it into a json file
    prompt2=config['prompt_seperate_tags']
    prompt2=prompt2.replace("${tags}",tags)
    prompt2=prompt2.replace("${ocr_fr}",OCRout_fr)
    result=bot.send_prompt(prompt2)

    print(result)

    # the third prompt is to add arabic values to the json file
    prompt3=config['prompt_seperate_tags_ar2']
    prompt3=prompt3.replace("${ocr_ar}",OCRout_ar)
    result=bot.send_prompt(prompt3)
        
    # final_response=re.search(r'\{.*\}', result, re.DOTALL).group()
    
    print(result)
    

