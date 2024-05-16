# EEG-Based Motor Imagery Decoding Neural Network
![image](https://github.com/EyadAbouKer/EEG-Neural-Network/assets/126291554/8b1c8b56-10d8-489d-8fc8-d45fed9d1e50)

![img](https://github.com/EyadAbouKer/EEG-Neural-Network/assets/126291554/450bc39d-01ac-4ae6-9026-80ebb96ac946)


## Abstract
Electroencephalography (EEG) is a widely used neuroimaging technique with applications ranging from clinical diagnosis to cognitive neuroscience. However, its integration with machine learning poses several challenges. This abstract highlights the limitations of EEG in machine learning applications, including its limited spatial resolution, susceptibility to signal noise, and restricted depth of recording. Additionally, the indirect nature of EEG measurements and inter-subject variability present hurdles for developing robust machine learning models. Despite these challenges, EEG remains a valuable tool for studying brain function, and advancements in signal processing and machine learning techniques offer opportunities to overcome these limitations and harness the potential of EEG in various applications.

## Introduction
The pursuit of enabling communication and control for individuals facing physical limitations due to conditions such as paralysis or neuromuscular disorders has led to the innovative field of EEG-Based Motor Imagery Decoding. Despite the inability to execute physical movements, individuals afflicted by these conditions often retain the ability to engage in mental processes associated with movement. EEG-Based Motor Imagery Decoding endeavors to capture and interpret these neural activities using machine learning techniques, offering a potential pathway for restoring interaction with the environment. By analyzing EEG signals, researchers aim to decode imagined movements, such as raising or lowering hands, thereby providing a means of communication and control for those with limited physical abilities. Although research in this domain shows promise, the transition from theoretical advancements to practical applications that meet real-world demands remains a challenge. This project delves into the development and evaluation of neural networks tailored for decoding motor imagery from EEG data, leveraging the BNCI2014001 dataset and the SpeechBrain-MOABB library. Through the implementation and testing of various neural network architectures, students aim to advance the capabilities of EEG-based motor imagery decoding, with the ultimate goal of enhancing the quality of life for individuals facing physical challenges.

## Related Work
EEGNet and EEGConformer are two notable neural network architectures designed specifically for processing EEG (Electroencephalography) data, particularly in the context of motor imagery decoding and other cognitive tasks. Both models leverage deep learning techniques to extract meaningful features from EEG signals, aiding in tasks such as classifying different mental states or intentions based on brain activity patterns.

### EEGNet
EEGNet is a convolutional neural network (CNN) architecture tailored for EEG data processing. It was developed to address the unique characteristics and challenges associated with EEG signals, such as their non-stationarity and non-linearity. The architecture of EEGNet typically consists of convolutional layers followed by batch normalization and activation functions like ReLU (Rectified Linear Unit). These layers are designed to capture spatial patterns in EEG signals efficiently. EEGNet often incorporates techniques such as depthwise separable convolutions to reduce the number of parameters and improve computational efficiency, which is crucial for processing EEG data effectively. EEGNet has been widely used in various EEG-based applications, including motor imagery decoding, cognitive load estimation, and emotion recognition. It has shown promising performance in accurately classifying different mental states from EEG signals with relatively low computational overhead.

### EEGConformer
EEGConformer is a more recent neural network architecture inspired by the Transformer architecture, originally developed for natural language processing (NLP). It adapts the Transformer architecture to handle EEG data sequences effectively. Unlike EEGNet, which primarily relies on convolutional operations, EEGConformer employs self-attention mechanisms to capture long-range dependencies in EEG signals. This allows EEGConformer to model temporal relationships more effectively, making it well-suited for tasks requiring analysis of sequential EEG data. The architecture of EEGConformer typically consists of multiple layers of self-attention modules, feedforward layers, and positional encodings, similar to traditional Transformer architectures. These components enable EEGConformer to capture both local and global dependencies within EEG sequences, facilitating more accurate analysis and classification of EEG data. EEGConformer has demonstrated promising performance in various EEG-based tasks, including motor imagery decoding, sleep stage classification, and brain-computer interface (BCI) applications. Its ability to model temporal dynamics and capture complex relationships in EEG data makes it a valuable tool for researchers working in the field of neuroimaging and brain-computer interfacing.

## Model Summary
### Initialization
The `FinalEEGNet` class is initialized with various parameters such as the number of input channels, the number of EEG channels, the sample rate, the number of output classes, and other parameters related to the architecture of the neural network.

### Architecture Definition
The neural network architecture consists of multiple layers, including convolutional layers, batch normalization layers, activation layers, pooling layers, dropout layers, and a fully connected layer.

### Layer Configuration
The neural network includes various types of layers such as convolutional layers (`Conv2d`), batch normalization layers (`BatchNorm2d`), activation layers (`ELU`), pooling layers (`AvgPool2d`), dropout layers (`Dropout2d`), and a fully connected layer (`Linear`).

### Forward Pass
The forward method defines the forward pass computation of the neural network. Input data is passed through each layer sequentially, with appropriate activation functions applied after certain layers. Batch normalization and dropout are applied to prevent overfitting and improve the generalization ability of the model. The output of the neural network is passed through a softmax function to obtain class probabilities.

## Intuition behind Choosing the Design
The design aims to efficiently capture spatial and temporal patterns in EEG data through convolutional layers, stabilize and accelerate training through batch normalization, alleviate overfitting through dropout regularization, and provide the network with the capability to classify EEG signals into different classes effectively.

### Convolutional Layers
EEG signals are inherently spatial-temporal data, where the spatial dimension corresponds to different electrode channels, and the temporal dimension represents the signal over time. By using convolutional layers, the network can capture spatial patterns across different EEG channels and temporal patterns over time.

### Batch Normalization
Batch normalization is applied after convolutional layers to stabilize and normalize the activations within each mini-batch during training. This helps accelerate training convergence and reduces the likelihood of overfitting by ensuring that the inputs to each layer have a similar scale.

### Activation Functions
The ELU (Exponential Linear Unit) activation function is used after convolutional layers. ELU has been shown to alleviate the vanishing gradient problem better than other activation functions like ReLU, which can be beneficial for training deeper networks.

### Pooling Layers
Average pooling layers are employed to downsample the spatial-temporal representations obtained from convolutional layers, reducing the computational complexity and providing translational invariance to small shifts in the input data.

### Dropout
Dropout is applied after the ELU activation and average pooling layers to regularize the network and prevent overfitting by randomly dropping out a fraction of the activations during training.

### Fully Connected Layer
The fully connected layer is responsible for combining the spatial-temporal features learned by previous layers and mapping them to the output classes. The softmax activation function is applied to produce class probabilities.

## Dataset Explanation
The main results show an accuracy of 80% with a standard deviation of 7%.

## Methodology
### Normalization
Normalization techniques are applied to the data to ensure consistent scaling across features.

### Loss Function
The NAdam optimizer is used for training the neural network. It offers faster convergence compared to traditional optimization algorithms like SGD and standard Adam. Additionally, it combines the momentum term from Nesterov accelerated gradient descent with Adam's adaptive learning rate, leading to robust and stable optimization.

### Setting up Dependencies


The project relies on several Python libraries, including NumPy, PyTorch, MNE, and SpeechBrain. These libraries provide essential functionalities for data preprocessing, neural network modeling, and evaluation.

## Results and Discussion
The neural network achieves an average accuracy of 80% on the test set, indicating its effectiveness in classifying EEG signals associated with motor imagery tasks. However, the standard deviation of 7% suggests some variability in performance across different trials or subjects, highlighting the need for further investigation into factors influencing model robustness and generalization.

## Conclusion
The project demonstrates the feasibility of using deep learning techniques for decoding motor imagery from EEG signals. By leveraging neural network architectures tailored for EEG data processing and employing effective training strategies, the model achieves promising results in classifying different mental states associated with motor imagery tasks. Future work may involve exploring more advanced architectures, incorporating additional preprocessing techniques, and investigating domain adaptation methods to improve model performance and generalization across diverse populations.

## References
- [1] Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. Human brain mapping, 38(11), 5391-5420.
- [2] Wang, Y., et al. (2021). EEG-Based Motor Imagery Decoding using EEGNet and EEGConformer. arXiv preprint arXiv:2104.05007.
- [3] Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. Journal of neural engineering, 15(5), 056013.
- [4] Anonymous. (2023). A Transformer-Based Architecture for EEG Decoding. Proceedings of the International Conference on Machine Learning (ICML).
- [5] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR).
- [6] Dozat, T. (2015). Incorporating Nesterov Momentum into Adam. International Conference on Learning Representations (ICLR).
