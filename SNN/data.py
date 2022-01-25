import os
import csv
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import utils
import matplotlib.pyplot as plt
import matplotlib


def load_dataset(User_params, root_dir, root_dir_data, np_dtype) :
    if User_params['Project'] == "Image":
        # Here we load the Dataset
        if(User_params['Dataset'] == "MNIST") :
            transform = torchvision.transforms.Compose([
                # you can add other transformations in this list
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            root = os.path.expanduser(root_dir_data + "/datasets/torch/mnist")
            train_dataset = torchvision.datasets.MNIST(root, train=True, transform=transform, target_transform=None, download=True)
            test_dataset = torchvision.datasets.MNIST(root, train=False, transform=transform, target_transform=None, download=True)
        elif(User_params['Dataset'] == "FashionMNIST") :
            transform = torchvision.transforms.Compose([
                # you can add other transformations in this list
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            root = os.path.expanduser(root_dir_data + "/datasets/torch/fashion-mnist")
            train_dataset = torchvision.datasets.FashionMNIST(root, train=True, transform=transform, target_transform=None, download=True)
            test_dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=transform, target_transform=None, download=True)
        elif(User_params['Dataset'] == "CIFAR10") :
            transform = torchvision.transforms.Compose([
                # you can add other transformations in this list
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])
            root = os.path.expanduser(root_dir_data + "/datasets/torch/cifar-10")
            train_dataset = torchvision.datasets.CIFAR10(root, train=True, transform=transform, target_transform=None, download=True)
            test_dataset = torchvision.datasets.CIFAR10(root, train=False, transform=transform, target_transform=None, download=True)
        #=======================================================================================
        # Standardize data
        if (User_params['Dataset']=="MNIST" or User_params['Dataset']=="FashionMNIST"):
            # x_train = torch.tensor(train_dataset.train_data, device=device, dtype=dtype)
            x_train = np.array(train_dataset.train_data, dtype=np_dtype)/255
            # x_test = torch.tensor(test_dataset.test_data, device=device, dtype=dtype)
            x_test = np.array(test_dataset.test_data, dtype=np_dtype)/255

            # y_train = torch.tensor(train_dataset.train_labels, device=device, dtype=dtype)
            y_train = np.array(train_dataset.train_labels, dtype=np.int)
            # y_test  = torch.tensor(test_dataset.test_labels, device=device, dtype=dtype)
            y_test  = np.array(test_dataset.test_labels, dtype=np.int)

            mean_lum = np.mean(x_train)

        # =================================================================================
        if (User_params['Dataset']=="MNIST" or User_params['Dataset']=="FashionMNIST"):
            Dataset_train_size = 60000
            Dataset_test_size = 10000
            User_params['number_of_batches'] = int(Dataset_train_size / User_params['batch_size'])
            # train_dataloader = data_generator(x_train, y_train, User_params['batch_size'], device, dtype)
            # valid_dataloader = data_generator(x_test, y_test, User_params['batch_size'], device, dtype)
            # test_dataloader = data_generator(x_test, y_test, User_params['batch_size'], device, dtype)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=User_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)
            valid_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=User_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=User_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)
        elif (User_params['Dataset']=="CIFAR10"):
            Dataset_train_size = 50000
            Dataset_test_size = 10000
            User_params['number_of_batches'] = int(Dataset_train_size / User_params['batch_size'])
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=User_params['batch_size'],shuffle=True, num_workers=0, drop_last=True)
            valid_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=User_params['batch_size'],shuffle=True, num_workers=0, drop_last=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=User_params['batch_size'],shuffle=True, num_workers=0, drop_last=True)

        return [train_dataloader, valid_dataloader, test_dataloader], [x_train, x_test, y_train, y_test, mean_lum]

    elif User_params['Project'] == "VPN":
        bins_time = User_params['Dataset_params']['nb_bins_time']
        nb_bins_size = User_params['Dataset_params']['nb_bins_size']
        gaps = User_params['Dataset_params']['gaps'] #intervals between graduations
        min_value = User_params['Dataset_params']['min_value']
        bins_size, shift = utils.calibration_bins_size(nb_bins_size, gaps, min_value)

        bins = [bins_time, bins_size]
        ##################################################
        train_data_root = root_dir_data + "/Training/"
        valid_data_root = root_dir_data + "/Validation/"
        test_data_root = root_dir_data + "/Testing/"

        training_hist = os.listdir(train_data_root)
        training_hist = [x for x in training_hist if os.path.isfile(os.path.join(train_data_root,x))]
        print("{} training histograms".format(len(training_hist)))

        valid_hist = os.listdir(valid_data_root)
        valid_hist = [x for x in valid_hist if os.path.isfile(os.path.join(valid_data_root,x))]
        print("{} validation histograms".format(len(valid_hist)))

        test_hist = os.listdir(test_data_root)
        test_hist = [x for x in test_hist if os.path.isfile(os.path.join(test_data_root,x))]
        print("{} testing histograms".format(len(test_hist)))        
        ##################################################
        if User_params['Problem'] == "All" or User_params['Problem']=="All_with_encryption_feature" :
            categories_file = open(root_dir + "/Labels.csv","r")
            categories = csv.reader(categories_file)
            list_categories = []
            for row in categories:
                list_categories += row
            categories_file.close()
        elif User_params['Problem'] == "Encryption" :
            list_categories = ["nonVPN", "VPN", "Tor"]
        elif User_params['Problem']== "Application" or User_params['Problem'] == "Application_with_encryption_feature" :
            list_categories = ["Video", "VOIP", "FileTransfer", "Chat", "Browsing"]

        label_dct = {k:i for i,k in enumerate(list_categories)}
        print("label_dct:")
        print(label_dct)
        #=================================================================================
        try:
            train_dataset = TrafficClassificationDataset(User_params=User_params, data_root=root_dir_data, label_dct=label_dct, bins=bins, shift=shift, mode="train")
            train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.weights,len(train_dataset.weights))
            train_dataloader_tmp = DataLoader(train_dataset, batch_size=User_params['batch_size'], shuffle=False, num_workers=0, sampler=train_sampler, collate_fn=utils.collate_fn)
            train_dataloader_create=True
        except:
            print("Error while trying to create the main train_dataset/train_dataloader, creating another one similar to validation ...")
            train_dataset = TrafficClassificationDataset(User_params=User_params, data_root=root_dir_data, label_dct=label_dct, bins=bins, shift=shift, mode="valid")
            train_dataloader_tmp = DataLoader(train_dataset, batch_size=User_params['batch_size'], shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
            train_dataloader_create=False
        print("Number of train_dataset samples : ", len(train_dataset.labels))
        # print("Video_nonVPN : ", np.sum(np.array(train_dataset.labels) == 0))
        # print("Video_Tor : ", np.sum(np.array(train_dataset.labels) == 1))
        # print("Video_VPN : ", np.sum(np.array(train_dataset.labels) == 2))
        # print("VOIP_nonVPN : ", np.sum(np.array(train_dataset.labels) == 3))
        # print("VOIP_Tor : ", np.sum(np.array(train_dataset.labels) == 4))
        # print("VOIP_VPN : ", np.sum(np.array(train_dataset.labels) == 5))
        # print("FileTransfer_nonVPN : ", np.sum(np.array(train_dataset.labels) == 6))
        # print("FileTransfer_Tor : ", np.sum(np.array(train_dataset.labels) == 7))
        # print("FileTransfer_VPN : ", np.sum(np.array(train_dataset.labels) == 8))
        # print("Chat_nonVPN : ", np.sum(np.array(train_dataset.labels) == 9))
        # print("Chat_Tor : ", np.sum(np.array(train_dataset.labels) == 10))
        # print("Chat_VPN : ", np.sum(np.array(train_dataset.labels) == 11))
        # print("Browsing_nonVPN : ", np.sum(np.array(train_dataset.labels) == 12))
        # print("Browsing_Tor : ", np.sum(np.array(train_dataset.labels) == 13))

        if User_params['Problem']=="All_with_encryption_feature" or User_params['Problem']=="Application_with_encryption_feature":
            train_batch_num = int(len(training_hist) / User_params['batch_size'])
            ratio = train_batch_num / (train_batch_num + 1)
            index = 0
            for x_batch, y_batch in train_dataloader_tmp:
                if index == 0:
                    mean = torch.mean(x_batch).item()
                else:
                    mean = ratio * mean + (1 - ratio) * torch.mean(x_batch).item()
                index += 1
            print("Mean value of train_dataset :", mean)
            train_dataset.mean = mean

        if train_dataloader_create :
            train_dataloader = DataLoader(train_dataset, batch_size=User_params['batch_size'], shuffle=False, num_workers=0, sampler=train_sampler, collate_fn=utils.collate_fn)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=User_params['batch_size'], shuffle=False, num_workers=0, collate_fn=utils.collate_fn)

        valid_dataset = TrafficClassificationDataset(User_params=User_params, data_root=root_dir_data, label_dct=label_dct, bins=bins, shift=shift, mode="valid")
        valid_dataset.mean = train_dataset.mean
        valid_dataloader = DataLoader(valid_dataset, batch_size=User_params['batch_size'], shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
        print("Number of valid_dataset samples : ", len(valid_dataset.labels))

        test_dataset = TrafficClassificationDataset(User_params=User_params, data_root=root_dir_data, label_dct=label_dct, bins=bins, shift=shift, mode="test")
        test_dataset.mean = train_dataset.mean
        test_dataloader = DataLoader(test_dataset, batch_size=User_params['batch_size'], shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
        print("Number of test_dataset samples : ", len(test_dataset.labels))
        
        User_params['number_of_batches'] = int(len(training_hist) / User_params['batch_size'])

        # valid_batch_num = int(len(valid_hist) / User_params['batch_size'])
        # ratio = valid_batch_num / (valid_batch_num + 1)
        # index = 0
        # for x_batch, y_batch in valid_dataloader:
        #     if index == 0:
        #         mean = torch.mean(x_batch).item()
        #     else:
        #         mean = ratio * mean + (1 - ratio) * torch.mean(x_batch).item()
        #     index += 1
        # print("Mean value of valid_dataset :", mean)

        # test_batch_num = int(len(test_hist) / User_params['batch_size'])
        # ratio = test_batch_num / (test_batch_num + 1)
        # index = 0
        # for x_batch, y_batch in test_dataloader:
        #     if index == 0:
        #         mean = torch.mean(x_batch).item()
        #     else:
        #         mean = ratio * mean + (1 - ratio) * torch.mean(x_batch).item()
        #     index += 1
        # print("Mean value of test_dataset :", mean)

        return [train_dataloader, valid_dataloader, test_dataloader], label_dct


class TrafficClassificationDataset(Dataset):

    def __init__(self, User_params, data_root, label_dct, mode, bins, shift):
        
        assert mode in ["train", "valid", "test"], 'mode should be "train", "valid" or "test"'

        self.User_params = User_params
        self.filenames = []
        self.labels = []
        self.mode = mode
        self.bins = bins
        self.shift = shift
        self.mean = 0.1

        if self.mode == "train" or self.mode == "valid" or self.mode == "test":
            testing_list = utils.txt2list(os.path.join(data_root, "testing-list.txt"))
            validation_list = utils.txt2list(os.path.join(data_root, "validation-list.txt"))
        else:
            testing_list = []
            validation_list = []

        for root, dirs, files in os.walk(data_root):
            for filename in files:
                if not filename.endswith(".csv"):
                    continue
                elif self.User_params['Dataset']=="ISCX_nonVPN" and filename.split("_")[1]!="nonVPN":
                    continue
                elif self.User_params['Dataset']=="ISCX_VPN" and filename.split("_")[1]!="VPN":
                    continue
                elif self.User_params['Dataset']=="ISCX_Tor" and filename.split("_")[1]!="Tor":
                    continue

                command = root.split("/")[-1]
                if (self.User_params['Problem']) == "All" or User_params['Problem']=="All_with_encryption_feature":
                    label_name =  filename.split("_")[0] + "_" + filename.split("_")[1]
                elif (self.User_params['Problem']) == "Encryption":
                    label_name = filename.split("_")[1]
                elif (self.User_params['Problem']) == "Application" or User_params['Problem'] == "Application_with_encryption_feature":
                    label_name = filename.split("_")[0]
                label = label_dct.get(label_name)
                if label is None:
                  print("ignored command: %s"%command)
                  break      
                partial_path = '/'.join([command, filename])
                  
                testing_file = (partial_path in testing_list)
                validation_file = (partial_path in validation_list)
                training_file = not testing_file and not validation_file
                  
                if (self.mode == "test" and testing_file) or (self.mode=="train" and training_file) or (self.mode=="valid" and validation_file):
                    full_name = os.path.join(root, filename)
                    self.filenames.append(full_name)
                    self.labels.append(label)
                
        if self.mode == "train":
            label_weights = 1./np.unique(self.labels, return_counts=True)[1]
            label_weights /=  np.sum(label_weights)
            self.weights = torch.DoubleTensor([label_weights[label] for label in self.labels])
            

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        filename = self.filenames[idx]

        histogram_data = open(filename, "r")
        reader_histogram_data = csv.reader(histogram_data)
        ind = 0
        for row in reader_histogram_data:
            if ind == 0:
                Time_of_arrival1 = list(map(float, row))
            elif ind == 1:
                Packet_size1 = list(map(int, row))
            elif ind == 2:
                Time_of_arrival2 = list(map(float, row))
            else:
                Packet_size2 = list(map(int, row))
                for i in range(len(Packet_size2)):
                    Packet_size2[i] += self.shift
            ind += 1
          
        histogram_data.close()
        
        Time, Size = Time_of_arrival1 + Time_of_arrival2, Packet_size1 + Packet_size2
        assert(len(Time)==len(Size))
        item = np.histogram2d(Time, Size, bins = self.bins)[0]
        label = self.labels[idx]

        if self.User_params['Problem'] == "All_with_encryption_feature" or self.User_params['Problem'] == "Application_with_encryption_feature":
            handmade_features = np.zeros((int(np.shape(item)[0]),3))
            item = np.append(item,handmade_features,axis=1)
            labels_encryption = {'nonVPN': [0, 3, 6, 9, 12], 'Tor': [1, 4, 7, 10, 13], 'VPN': [2, 5, 8, 11]}
            item[:,300] = self.mean*(label in labels_encryption['nonVPN'])
            item[:,301] = self.mean*(label in labels_encryption['Tor'])
            item[:,302] = self.mean*(label in labels_encryption['VPN'])



        # Size_mean = int(np.mean(Size))
        # Size_shift = np.roll(Size,shift=1)
        # Size_shift[0]=0
        # Time_shift = np.roll(Time,shift=1)
        # Time_shift[0] = 0
        # Size_diff=Size-Size_shift
        # Time_diff=Time-Time_shift+1e-9
        # Size_slope=Size_diff/Time_diff
        # Size_slope_mean=np.mean(Size_slope)
        # Size_slope_mean=int(Size_slope_mean)

        # with open('./data.csv', 'a', newline='') as csvfile:
            # writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # writer.writerow([label, Size_mean, Size_slope_mean])




        if not self.User_params['Dataset_params']['plot_data_mode'] == "none":
            label_dct = {0:'Video_Unencrypted', 1:'Video_Tor', 2:'Video_VPN', 3:'VOIP_Unencrypted', 4:'VOIP_Tor', 5:'VOIP_VPN',
             6:'FileTransfer_Unencrypted', 7:'FileTransfer_Tor', 8:'FileTransfer_VPN', 9:'Chat_Unencrypted', 10:'Chat_Tor',
             11:'Chat_VPN', 12:'Browsing_Unencrypted', 13:'Browsing_Tor'}

            if label_dct[label] == self.User_params['Dataset_params']['plot_data_type']:

                if self.User_params['Dataset_params']['plot_data_mode'] == "histogram":
                    # item_image = item / np.max(np.abs(item))
                    # item_image *= (255.0 / item_image.max())
                    item_image = np.clip(255*item, 0, 255)

                    plt.imshow(item_image.T, cmap=plt.cm.gray_r, origin="lower", aspect='auto')
                    # X, Y = np.meshgrid(np.arange(item_image.shape[0]), np.arange(item_image.shape[1]))
                    # plt.scatter(X,Y,c=item_image.T, s=1**2, cmap=plt.cm.gray_r, marker="o")
                    plt.xlabel("Normalized Time of Arrival",fontsize=22)
                    plt.ylabel("Packet Size(B)",fontsize=22)

                elif self.User_params['Dataset_params']['plot_data_mode'] == "point_cloud":
                    plt.plot(Time, Size, "k.")
                    plt.xlabel("Time of Arrival(s)",fontsize=22)
                    plt.ylabel("Packet Size(B)",fontsize=22)

                # plt.title(name + "\n " + str(len(Size)) + " paquets ; " + str(round(Time[-1] - Time[0], 1)) + "s")
                plt.title(label_dct[label],fontsize=30)

                plt.show()

        return item, label
