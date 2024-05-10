import torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import re
import json
from transformers import AutoModel, AutoTokenizer
torch.set_grad_enabled(False)
model = AutoModel.from_pretrained('Shanghai_AI_Laboratory/internlm-xcomposer2-4khd-7b', trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('Shanghai_AI_Laboratory/internlm-xcomposer2-4khd-7b', trust_remote_code=True)
model = model.eval()
def InternLMXComposer(image,query):
# query = '<ImageHere>Illustrate the fine details present in the image'
# image = 'examples/4244.png'
   with torch.cuda.amp.autocast():
     response, his = model.chat(tokenizer, query='<ImageHere>'+query, image=image, hd_num=55, history=[], do_sample=False, num_beams=3)
   # print(response)
   with open('InternLMXComposer_GeoQA.txt', 'a', encoding='utf-8') as file:
     file.write('answer:' + str(response) + '\n')
   print(response)
   print(re.search(r'\d+(\.\d+)?(?=\D*$)', response).group(0))
   return re.search(r'\d+(\.\d+)?(?=\D*$)', response).group(0)

allImages=[]
allQuestions=[]
allAnswers=[]
j=0
flag=0
total=0
correct=0
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
image_files = os.listdir('image')
image_files.sort(key=natural_sort_key)
for files in image_files:
        file_path = os.path.join('image', files)
        allImages.append(file_path)
json_files = os.listdir('json')
json_files.sort(key=natural_sort_key)
for filename in json_files:
    try:
        total=total+1
        if filename.endswith('.json'):
            file_path = os.path.join('json', filename)
            with open(file_path, 'r',encoding='utf-8') as file:
                json_data = json.load(file)
                with open('InternLMXComposer_GeoQA.txt', 'a', encoding='utf-8') as file:
                    file.write('ID:' + str(j) + '\n'+'picturePath:'+allImages[j]+'\n')
                # allAnswers.append(re.search(r'故选：?(\w)', json_data['answer']).group(1))
                res=InternLMXComposer(allImages[j], json_data['subject'])
                ans=re.search(r'\d+(\.\d+)?(?=\D*$)', json_data['choices'][json_data['label']]).group(0)
                if res==ans:
                    correct=correct+1
                with open('InternLMXComposer_GeoQA.txt', 'a', encoding='utf-8') as file:
                    file.write('answer:' + res + '\n'+'correct answer:'+ans+'\n'+"total:"+str(total)+"  correct:"+str(correct)+"  acc:"+str(correct/total)+'\n'+'------------------------------------------------------------------------------------'+'\n')
                print("total:"+str(total)+"  correct:"+str(correct)+"  acc:"+str(correct/total))
                print(allImages[j])
                j=j+1
    except (AttributeError, UnboundLocalError, ConnectionError,KeyError):
        j=j+1
