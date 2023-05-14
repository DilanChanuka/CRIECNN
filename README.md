# CRIECNN: circRNA-RBP binding site prediction using an ensemble convolutional neural network and advanced feature extraction methods.

CRIECNN is an ensemble deep learning model, and this repository contains all of the experiments' code and datasets.

## Abstract

Circular RNAs (circRNAs) are a class of non-coding RNA that has been recognized as a significant molecule in biology. The identification of interactions between circRNAs and RNA-binding proteins (RBPs) is a crucial objective in circRNA research. However, the limited availability and insufficient accuracy of existing prediction models highlight the need for more advanced approaches. To address this issue, we propose CRIECNN (Circular RNA-RBP interaction predictor using an Ensemble Convolutional Neural Network), an ensemble deep learning model that uses advanced feature extraction methods and a novel approach to improve circRNA-RBP binding site prediction accuracy. We evaluate CRIECNN using four distinct sequence datasets and encoding methods (BERT, Doc2Vec, KNF, and EIIP) to extract circRNA sequence features and improve prediction accuracy. We use an ensemble convolutional neural network as the core of the model, followed by a BiLSTM with a self-attention mechanism to further refine the features. Our results show that CRIECNN outperforms state-of-the-art methods in both accuracy and performance, and effectively predicts circRNA-RBP interactions from either full-length circRNA sequences or sequence fragments. Our novel approach represents a significant improvement in circRNA-RBP interaction prediction.

## Experiment Requirements

To carry out these experiments, we used a Google Colab Pro instance with TPU and GPU backends as needed for each sub-dataset. The experiments were carried out using High-RAM runtimes. TPU backend with 35GB RAM and 16GB GPU backends with 25GB RAM were used. That kind of Hardware performance is required to conduct the experiments when we run the model with large sub-datasets. Keras 2.6.0 was used to compile and fit the CRIECNN model, and PyTorch 1.13.0+cu116 was used to implement the BERT encoder.

By using the following commands (on the Google Colab Notebook), we can run the CRIECNN model and test it. We need to use the path.py and protein list.py files to set the required paths and Protein list before running the CRIECNN.py file.

!python /content/drive/MyDrive/CRIECNN/code/CRIECNN.py

Similarly, all experiments can be tested using existing code files.
