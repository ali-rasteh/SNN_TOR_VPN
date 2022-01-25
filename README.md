# PIR-SNN-TOR-VPN: Encrypted Internet traffic classification using a supervised Spiking Neural Network

Internet traffic recognition is an essential tool for access providers since recognizing applications related to different data packets transmitted on a network help them define adapted priorities. That means, for instance, high capacity requirements for an audio conference and, low ones for a file transfer. The intended effect is to enhance users’ experience. Nevertheless, internet traffic is increasingly encrypted which makes classic traffic recognition techniques, such as payload inspection, ineffective. Thus, this research aims at doing the same thing using machine learning techniques for encrypted traffic classification, looking only at packet size and time of arrival. Spiking neural networks (SNN), largely inspired by how biological neurons operate, were used for their ability to recognize time-related data packet features. Furthermore, such SNNs could be implemented efficiently on neuromorphic hardware with a low energy footprint.

Content:
- Bibliography:
  * "bibliography": complete bibliography report
  * "Recap articles": quick sum up of relevant references
  * Folder "Articles": articles for the bibliography

- Internet_traffic:
  * "Dataset_repartition": sample repartition between training, validation and testing datasets
  * "IP_adresses_dictionnary": each histogram name contains 4 numbers. The first two correspond to IP     adresses (the last two to port numbers). 
  * Folder "Processing programs": Python programs used to process original dataset

- SNN:
  * "data, models, optim, utils": Python programs to run the notebook
  * "Labels": list of labels
  * Folder "Dataset": CSV-files containing packet' size and time of arrival for each histogram
  * Folder "Model and results": information about the proposed model (parameters, performances, saved model)
  * Folder "Notebook": the notebook to run 
