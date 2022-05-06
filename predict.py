#This is the predict function 
#Written by Savion Ragster
#April - May 2022
#this program loads in an image and can predict what is in it. 

#To run this program:
#python predict.py --device gpu --top_k 3 --img_path flowers/train/13/image_05750.jpg

from train import *
from get_input_args import get_input_args

in_args = get_input_args()
data_dir = in_args.data_dir
device = in_args.device
arch = in_args.arch
img_path = in_args.img_path
top_k = in_args.top_k
cat_to_name = in_args.cat_to_name 
 
if cat_to_name == cat_to_name:
    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
 
#Set Device correctly:
if device == 'gpu':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
else:
    device = torch.device("cpu")
    print(f'device: {device}')

prediction = predict(img_path, new_model, cat_to_name, device, top_k)#This creates the prediction and returns top probabilities and names. 
print(f"Top labels are: {prediction[0]} \n and percentages are: {prediction[1]}" )