# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 23:55:49 2020

@author: Florian Delpech

This function enables to save histograms classified by the network in a folder 'Analyse'.
The subfolders are the predicted label. Each histogram has a 'check' flag: RIGHT if it has been well-classified, WRONG if not.
"""


import csv
import matplotlib.pyplot as plt

# A function that enables to get the key from the value that corresponds to it
def key(dictio,val):
    for c,v in dictio.items():
        if v == val:
            return(c)

# A function that extracts data from a txt file into a list
def txt2list(filename):
    lines_list = []
    with open(filename, 'r') as txt:
        for line in txt:
            lines_list.append(line.rstrip('\n'))
    return lines_list

# Inputs:
# - name: name of the histogram to build
# - path: destination path to save the histogram
# - check: 'WRONG' or 'RIGHT' to distinguinsh well-classified and bad-classified histograms
# - label: histogram label (class + encryption technique)
def build_histogram(name, path, check, label):
    histo_file = open(name,'r')
    histo_reader = csv.reader(histo_file)
    Packet_size = []
    Time_of_arrival = []
    for row in histo_reader:
        Time_of_arrival.append(float(row[0]))
        Packet_size.append(float(row[1]))
    histo_file.close()
    name=name.split(".csv")[0]
    name=name.split("Dataset/Validation/")[1]
    plt.plot(Time_of_arrival, Packet_size, "k.")
    plt.xlabel("Time of arrival")
    plt.ylabel("Packet size")
    plt.title(name + "\n" + str(len(Packet_size)) + " paquets ; " + check)
    plt.savefig(path + "/" + check + "_" + str(label) + "_" + name + ".png")
    plt.clf()
    

filename = "analyse.csv"
file = open(filename,'r')
file_reader = csv.reader(file)

labels = {0:"Video_nonVPN", 1:"Video_Tor", 2:"Video_VPN", 3:"VOIP_nonVPN", 4:"VOIP_Tor"}
labels.update({5:"VOIP_VPN", 6:"FileTransfer_nonVPN", 7:"FileTransfer_Tor", 8:"FileTransfer_VPN"})
labels.update({9:"Chat_nonVPN", 10:"Chat_Tor", 11:"Chat_VPN", 12:"Browsing_nonVPN", 13:"Browsing_Tor"})
compteur = 0
name_list = []
check = ("WRONG", "RIGHT")

for row in file_reader:
    compteur+=1
    print("Processing image " + str(compteur))
    label1, label2 = int(row[0,0]), int(row[0,1])
    name = row[1]
    name_list.append(name)
    root_path = "Analyse/" + str(label1) + "." + labels.get(label1)
    build_histogram(name, root_path, check[0], label2)

validation_list = txt2list("Dataset/validation-list.txt")
for i in range(len(validation_list)):
    file_name = "Dataset/" + validation_list[i]
    if file_name not in name_list:
        compteur+=1
        print("Processing image " + str(compteur))
        name = file_name.split(".csv")[0].split("Dataset/Validation/")[1]
        name = name.split("_")[0] + "_" + name.split("_")[1]
        label = key(labels, name)
        root_path = "Analyse/" + str(label) + "." + name
        build_histogram(file_name, root_path, check[1], label)

file.close()
   


    
    

    
    