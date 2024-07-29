from fastapi import FastAPI,File
import torch
import numpy as np
import cv2
import os
import json
from PIL import Image
import easyocr
import re
from fastapi.responses import JSONResponse
import io

app=FastAPI()

verso_model=torch.hub.load('yolov5/', 'custom', source='local', path = 'models/verso.onnx', force_reload = True)

recto_model=torch.hub.load('yolov5/', 'custom', source='local', path = 'models/recto.onnx', force_reload = True)
reader = easyocr.Reader(['en','fr'],gpu=False)
sift = cv2.SIFT_create()
def detect(path,model):
    
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    d=img.copy()
    result=model(d)
    result.show()
    return result.pandas().xyxy[0]
def redress(im,ref):
    
    im_ref=cv2.imread(ref)
    
    try:

        # Find keypoints and compute descriptors with SIFT
        kp_ref, des_ref = sift.detectAndCompute(im_ref, None)
        kp, des = sift.detectAndCompute(im, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des, des_ref, k=2)
        good_matches = []
        factor = 0.6
        for m, n in matches:
            if m.distance < factor * n.distance:
                good_matches.append(m)
        # Retrieve your keypoints in `good_matches`
        good_kp = np.array([kp[match.queryIdx].pt for match in good_matches])
        good_kp_ref = np.array([kp_ref[match.trainIdx].pt for match in good_matches])
        # Find transformation
        m, mask = cv2.findHomography(good_kp, good_kp_ref, cv2.RANSAC, 5.0)

        # Apply transformation
        im_adjusted = cv2.warpPerspective(im, m, (im_ref.shape[1], im_ref.shape[0]))
    
    except:
        im_adjusted=im

    return im_adjusted
def get_df(image,model_name):
    
    img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    result=model_name(img)
    df=result.pandas().xyxy[0]
    df=df.sort_values(by=['name','ymax'],ignore_index=True)
    return df
def remove_items(test_list, item): 
    res = list(filter((item).__ne__, test_list)) 
    return res

def card_valid(text):
    keys=["CNI","CARTE D'IDENTITE",'CARTE D’IDENTITE',"CARTE D'IDENTITÉ","CARTE DIDENTITÉ","CARTE DIDENTITE"
    ,'CARTE NATIONALE DIDENTITE',"CARTE NATIONALE DIDENTITÉ","CARTE NATIONALE D'IDENTITE","CARTE NATIONALE D'IDENTITÉ",'DOCUMENT DIDENTITE','IDENTITY CARD','IDENTITY DOCUMENT']
    for k in keys:
        occ=re.findall(k,text,re.IGNORECASE)
        if occ:
            return k
        else:
            continue

def is_image(file_content):
    try:
        img=Image.open(io.BytesIO(file_content)).verify()
        return True
    except (IOError, SyntaxError):
        return False

@app.post('/authentification')
async def authentification(recto_img:bytes = File(...),verso_img:bytes = File(...)): # prends le chemin des images recto et verso dans cet ordre et retourne un dictionnaire


    if is_image(recto_img)==False or is_image(verso_img)==False:
        return JSONResponse(status_code=500, content={"status_code": 500, "message":'The Files can be images.'})
    img=Image.open(io.BytesIO(recto_img))
    img=np.asarray(img)
    img=redress(img,'utils/ref_recto.jpg') # code pour redresser l'image
    
    img1=Image.open(io.BytesIO(verso_img))
    img1=np.asarray(img1)
    img1=redress(img1,'utils/ref_verso.jpg') # code pour redresser l'image
    
    df=get_df(img,recto_model)
    d=["CardNo","GivenNames","Surname","DateOfBirth",
    "PlaceOfBirth","DateOfIssue","IssuingAuthority",
    "Address",'Sex','Height','ExpiryDate','Signature']
    recto={key: '' for key in d}
    for name,y in zip(['A','d','sexe','taille','edate','sign'],['A','d','Sex','Height','ExpiryDate','Signature']):
        
        bbox=df.loc[(df['name']==name) & (df['confidence']>0.5)][['xmin','ymin','xmax','ymax']].values.astype(int)
        if name=='d':
            
            for i,ch in zip(range(len(bbox)),d):
                try:
                    x1,y1,x2,y2=bbox[i]
                    crop=img.copy()[y1:y2,x1:x2]
                    NomCI = reader.readtext(crop,detail=0)
                    l=' '.join(NomCI)
                    recto[ch]=l
                except:
                    recto[ch]=''
            
                
        elif name!='sign':
            try:
                x1,y1,x2,y2=bbox[0]
                crop=img.copy()[y1:y2,x1:x2]
                if name=='A':
                    NomCI = reader.readtext(crop,detail=0)
                    entete=' '.join(NomCI)
                else:
                    NomCI = reader.readtext(crop,detail=0)
                    l=' '.join(NomCI)
                    recto[y]=l
            except:
                if name=='A':
                    entete='non lisible'
                else:    
                    recto[y]=''
            

        else:           
            try:
                x1,y1,x2,y2=bbox[0]
                crop=img.copy()[y1:y2,x1:x2]           
                recto[y]=Image.fromarray(crop)
                
            except:
                recto[y]=''
   
    ###################### Verso Extraction #######################

    
    
    df=get_df(img1,verso_model)
    verso={key: '' for key in ['nin','d']}
    for name in ['nin','d']:
        try:
            bbox=df.loc[(df['name']==name) & (df['confidence']>0.5)][['xmin','ymin','xmax','ymax']].values.astype(int)
            x1,y1,x2,y2=bbox[0]
            crop=img1.copy()[y1:y2,x1:x2]
            NomCI = reader.readtext(crop,detail=0)
            l=' '.join(NomCI)
            verso[name]=l
        except:
            verso[name]=''
    
    ######################## Checking ################

    cpt=0
    check_card=card_valid(entete)
    if check_card:
        #print('entete_valid')
        cpt+=1

    ################# Code Verification ###############
    
    code=verso['d']
    code=code.replace('<',' ')
    code=code.split(' ')
    code=remove_items(code,'')
    numero=''.join(code[1:3])
    try:
        PP=''.join(remove_items(list(recto["CardNo"]),' '))
        check_pp=re.findall(PP,numero,re.IGNORECASE)
    except:
        cpt=cpt
   
    try:
       check_pp=re.findall(PP,numero,re.IGNORECASE)
       if check_pp and (PP!='' and numero!=''):
        # print('NPP_valid')
        cpt+=1 
    except:
        cpt=cpt
    
    text=' '.join(code)
    try:
        check_name=re.findall(recto['Surname'],text,re.IGNORECASE)
        if check_name and (recto['Surname']!='' and text!=''):
        # print('nom_valid')
            cpt+=1
    except:
        cpt=cpt
    
    try:
        check_surname=re.findall(recto["GivenNames"],text,re.IGNORECASE)
        if check_surname and (recto["GivenNames"]!='' and text!=''):
        # print('prenoms_valid')
            cpt+=1
    except:
        cpt=cpt
    
    try:
        CD=recto["ExpiryDate"]
        CD=CD.split('/')
        CD=recto['Sex']+CD[2][-2:]+CD[1]+CD[0]
        check_cd=re.findall(CD,text,re.IGNORECASE)
        if check_cd:
            # print('Cdate_valid')
            cpt+=1
    except:
        cpt=cpt
    prob=int((cpt/5)*100)
    recto.pop('Signature')
    if prob==0:
        recto={key: '' for key in d}
        verso={key: '' for key in ['NIN']}
        recto.update(verso)
        recto.update({'Authenticity':f"{prob}%"})
    else:
        recto.update({"NIN":verso['nin']})
        recto.update({'Authenticity':f"{prob}%"})

    return JSONResponse(status_code=200, content={"status_code": 200, "data": recto})