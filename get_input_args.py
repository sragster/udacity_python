#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#     
#This program was largeley copied from 'get_input_args.py' from the first project. 
#

# PROGRAMMER: Savion R
# DATE CREATED: Jan 6, 2022                            
# REVISED DATE: Aprill 30, 2022
# PURPOSE:get in args
# Imports python modules
import argparse
#from train import *
# 

def get_input_args(): 
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
       
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--data_dir', type=str, default='flowers', help='path to folder of images')
    parser.add_argument('--arch', type = str, default = 'vgg', help = '--arch - the CNN model architecture')
    parser.add_argument('--save_dir', type = str, default = 'saves/checkpoint.pth', help = '--save_dir - where to save the .pth file')
    parser.add_argument('--learning_rate', type = float, default = 0.01, help = '--learning rate')
    parser.add_argument('--hidden_units', type = int, default = 0, help = '--numbah hidden units')
    parser.add_argument('--epochs', type = int, default = '3', help = '--numbah epochs')
    parser.add_argument('--device', type = str, default = 'cpu', help = '--device: cpu default, gpu used if you type gpu')
    parser.add_argument('--train', type = bool, default = False, help = '--train: if True, will train network')
    #this is causing a bug where it goes to 'True' even if you type false
    parser.add_argument('--img_path', type = str, default = 'flowers/test/10/image_07090.jpg', help = '--img_path - path to image')
    parser.add_argument('--top_k', type = int, default = 5, help = '--top_k - top k categories to show')
    #parser.add_argument('--cat_to_name', type = str, default = 'cat_to_name', help = '--cat_to_name - file with category names mapping')
    parser.add_argument('--cat_to_name', action = 'store', dest = 'cat_to_name', type = str, help = '--cat_to_name - file with category names mapping')
    with open(passed_args.cat_to_name, 'r') as f: cat_to_name = json.load(f)
    
    
#     prediction_parser.add_argument('--category_names', action="store"
#                         dest="category_names", type=str, help = 'Directory and file containing the category names and mapping.' )
# with open(passed_args.category_names, 'r') as f:
#    cat_to_name = json.load(f)  #dictionary
    
    
   
    in_args = parser.parse_args()
    return in_args