# SNN_TOR_VPN: Encrypted Internet Traffic Classification Using a Supervised Spiking Neural Network

## Overview
Internet traffic recognition is essential for access providers since it helps them define adapted priorities in order to enhance user experience, e.g., a high priority for an audio conference and a low priority for a file transfer. As internet traffic becomes increasingly encrypted, the main classic traffic recognition technique, payload inspection, is rendered ineffective. Hence this paper uses machine learning techniques looking only at packet size and time of arrival. For the first time, Spiking neural networks (SNNs), which are inspired by biological neurons, were used for this task for two reasons. Firstly, they can recognize time-related data packet features. Secondly, they can be implemented efficiently on neuromorphic hardware. Here we used a simple feedforward SNN, with only one fully connected hidden layer, and trained in a supervised manner using the new method known as Surrogate Gradient Learning. Surprisingly, such a simple SNN reached an accuracy of 95.9% on ISCX datasets, outperforming previous approaches. Besides better accuracy, there is also a significant improvement in simplicity: input size, the number of neurons, trainable parameters are all reduced by one to four orders of magnitude. Next, we analyzed the reasons for this good performance. It turns out that, beyond spatial (i.e., packet size) features, the SNN also exploits temporal ones, mainly the nearly synchronous (i.e., within a 200ms range) arrival times of packets with specific sizes. Taken together, these results show that SNNs are an excellent fit for encrypted internet traffic classification: they can be more accurate than conventional artificial neural networks (ANN), and they could be implemented efficiently on low-power embedded systems.

The SNN_TOR_VPN project aims to classify encrypted internet traffic using a supervised spiking neural network (SNN). This project includes data processing, training, and evaluation of the SNN model to differentiate between various types of internet traffic such as VPN, Tor, and non-VPN traffic.


## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project is organized into the following directories and key files:

```
SNN_TOR_VPN/
│
├── Internet_traffic/
│   ├── IP_adresses_dictionnary.csv
│   ├── Processing programs/
│   │   ├── Analyse.py
│   │   ├── Filter_data.py
│   │   ├── Repartition.py
├── SNN/
│   ├── Labels.csv
│   ├── Main_SNN-TOR-VPN.py
│   ├── data.py
│   ├── models.py
│   ├── optim.py
│   ├── utils.py
├── README.md
```

## Installation

To use this project, you need to have Python installed on your machine. Clone this repository and install the required dependencies.

```sh
git clone https://github.com/ali-rasteh/SNN_TOR_VPN.git
cd SNN_TOR_VPN
pip install -r requirements.txt
```

## Usage

### Running the Analysis

The `Analyse.py` script processes the internet traffic data to generate histograms. Each histogram is saved in a folder named `Analyse`.

```sh
python Internet_traffic/Processing\ programs/Analyse.py
```

### Filtering Data

The `Filter_data.py` script processes the dataset containing traffic flows and organizes them into labeled folders.

```sh
python Internet_traffic/Processing\ programs/Filter_data.py
```

### Dataset Repartition

The `Repartition.py` script distributes samples between training, validation, and test datasets according to given percentages.

```sh
python Internet_traffic/Processing\ programs/Repartition.py
```

### Training the SNN

The `Main_SNN-TOR-VPN.py` script is the main entry point for training the supervised spiking neural network.
Run the script with desired parameters to train and test your network.

```sh
python SNN/Main_SNN-TOR-VPN.py
```

## Dataset

The dataset includes various types of internet traffic, labeled as follows:

- Video_nonVPN
- Video_Tor
- Video_VPN
- VOIP_nonVPN
- VOIP_Tor
- VOIP_VPN
- FileTransfer_nonVPN
- FileTransfer_Tor
- FileTransfer_VPN
- Chat_nonVPN
- Chat_Tor
- Chat_VPN
- Browsing_nonVPN
- Browsing_Tor

The labels are stored in the `SNN/Labels.csv` file.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## Citation

If you use this repository or code in your research, please cite it as follows:

### BibTeX
```bibtex
@misc{Rasteh_SNN_TOR_VPN,
  author       = {Rasteh, Ali and Delpech, Florian},
  title        = {SNN_TOR_VPN: Encrypted Internet Traffic Classification Using a Supervised Spiking Neural Network},
  year         = {2022},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/ali-rasteh/SNN_TOR_VPN}},
  doi          = {[![DOI](https://zenodo.org/badge/451463674.svg)](https://doi.org/10.5281/zenodo.14846258)},
  note         = {Accessed: 2022-01-25}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
