# -*- coding: utf-8 -*-
"""
Created on Sat May 23 01:20:25 2020

@author: Florian Delpech

This program enables to distribute samples between train, valid and test data according to given percentages
"""
import csv
import random as rd
import shutil
import numpy as np

# Inputs: 
# - List: list of samples (training, testing, validation)
# - name: name of the chosen dataset (between training, testing, validation)
# Output:
# - Repart label: a 14-lenght list such as list[i] is the amount of samples for the label i
def repartition(List, name):
    Repart_label = np.zeros(len(label_dct), dtype = int)
    for ind in range(len(List)):
        file = List[ind]
        app = file.split("_")[0]
        label_name = app + "_" + file.split("_")[1]
        label = label_dct.get(label_name)
        Repart_label[label]+=1
    return Repart_label

# A function that enables to get the key from the value that corresponds to it
def recup_key(dictio,val):
    for c,v in dictio.items():
        if v == val:
            return(c)
# Inputs:
# - List: list of samples (training, testing, validation)
# - nb: amount of samples in the chosen dataset (training, testing, validation)
# This function enables to print the amount of samples for each label along with its percentage
def print_repart_label(List, nb):
    for i in range(len(List)):
        if nb!=0:
            print(recup_key(label_dct, i) + " : " + str(List[i]) + " (" + str(round(List[i]*100/nb,2)) + "%)")
        else:
            print(recup_key(label_dct, i) + " : " + str(List[i]) + " (0%)")

#Inputs:
# - List: list of samples (training, testing, validation)
# - name: name of the chosen dataset (between training, testing, validation)
# This function enables to copy samples from the overall dataset to the specitic dataset (training, validation or testing)
def copy_file(List, name):
    with open("Dataset/" + name[0].lower() + name[1:] + "-list.txt", "w", newline='') as name_list :
        for ind in range(len(List)):
            file = List[ind]
            shutil.copyfile("Data/" + file + ".csv", "Dataset/" + name + "/" + file + ".csv")
            name_list.write(name + "/" + file + ".csv\n")
    return None

pct_train = 0.65 # percentage of training data
pct_test = 0.2 # percentage of testing data
pct_valid = 0.15 # percentage of validation data

assert pct_train + pct_test + pct_valid == 1

dataset_name = "Dataset.csv"
dataset = open(dataset_name, "r")
dataset_reader = csv.reader(dataset)

categories_file = open("Dataset/Labels.csv","r")
categories = csv.reader(categories_file)
list_categories = []
for row in categories:
  list_categories += row
categories_file.close()

label_dct = {k:i for i,k in enumerate(list_categories)}


total = 0
List_files = []

for row in dataset_reader:
    total+=1
    List_files+=row

print("Nombre de données : " + str(total))

nb_train = round(pct_train*total)
nb_test = round(pct_test*total)
nb_valid = total - nb_test - nb_train

print("Shuffling list")
for i in range(1000):
    rd.shuffle(List_files)
print("List shuffled")
    
Training_list = List_files[:nb_train]
Testing_list = List_files[nb_train:nb_train+nb_test]
Validation_list = List_files[nb_train+nb_test:]

Training_repart_label = repartition(Training_list, "Training")
Testing_repart_label = repartition(Testing_list, "Testing")
Validation_repart_label = repartition(Validation_list, "Validation")

print("Nombre de données d'entrainement : " + str(nb_train) + " (" + str(int(pct_train*100)) + "%)")
print_repart_label(Training_repart_label, nb_train)
print("\n")
print("Nombre de données de test : " + str(nb_test)+ " (" + str(int(pct_test*100)) + "%)")
print_repart_label(Testing_repart_label, nb_test)
print("\n")
print("Nombre de données de validation : " + str(nb_valid) + " (" + str(int(pct_valid*100)) + "%)")
print_repart_label(Validation_repart_label, nb_valid)
    
copy = int(input("Copy files? (press 1 if yes) ")) #checking that the repartition is convenient before copying
           
if copy == 1 :
    print("Copying files in folder Training")
    copy_file(Training_list, "Training")
    print("Copying files in folder Testing")
    copy_file(Testing_list, "Testing")
    print("Copying files in folder Validation\n")
    copy_file(Validation_list, "Validation")
    print("Copy completed")
    
  
    

