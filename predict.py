#This is the prediction for feeding in a single user input image:
from train import *
from get_input_args import get_input_args

in_args = get_input_args()
data_dir = in_args.data_dir
device = in_args.device
arch = in_args.arch
img_path = in_args.img_path
top_k = in_args.top_k
cat_to_name = in_args.cat_to_name #Uncomment to import your own cat to name
  
# import json
# with open('cat_to_name.json', 'r') as f:
#     cat_to_name = json.load(f)
  



prediction = predict(img_path, new_model, cat_to_name, device, top_k)#make sure you run this only once otherwise dic gets flipped twice
print(f"Top labels are: {prediction[0]} \n and percentages are: {prediction[1]}" )



#if you can't get this working - you can always revert it. 