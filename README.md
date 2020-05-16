# Disease-Leaf-Detection

<b> Update: This project is forked from the final project repository submitted at the conclusion of OR 610: Deep Learning and Predictive Analytics, Fall 2019 at George Mason University. I will be cleaning up and re-running the code.</b>

This repo was created to store our project files from the <a href="https://www.kaggle.com/xabdallahali/plantvillage-dataset"> Kaggle "PlantVillage" dataset </a>. The dataset contains three types of folders:

1. color
2. grayscale
3. segmented

We will first train 6 different off-the-shelf PyTorch architectures on the color dataset. After we have validated the success of various architectures and compared their accuracies, we will train the models using different data augmentations. 

<h2> IMPORTANT NOTES: </h2>

<b> FIRST: </b> For each of the architectures, it is critical to change the final layer of the chosen model. Though the PyTorch tutorial uses a fc final layer for the resnet, all other architectures will have varying final layer parameter dimensions. Please reference Nathen's .ipynb file to see examples of how the final layers can be modified.

<b> SECOND: </b> Make sure to update these two blocks of code from the PyTorch ConvNet tutorial code: 

* num_epochs = 5 instead of num_epochs = 25 (we don't have the computational bandwidth to run 25 epochs)

`model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=5)`

* step_size = 2 instead of step_size = 7 (decay every 2 epochs, rather than every 7)

`exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=2, gamma=0.1)`

Different Network Assignments:

1. AlexNet- Nathen
2. DenseNet- Nathen
3. ResNet- Sahithi
4. VGG - Sahithi
5. Inception- Jey
6. SqueezeNet- Jey
7. (Optional) Generative Adversarial Network (GAN)- Nathen

Later, when we perform data augmentation, we will consider the following techniques:

1. Gamma correction 
2. Gray Scale
3. Scaling
4. Noise injection or principal component analysis (PCA) color augmentation


References:
1. https://www.sciencedirect.com/science/article/pii/S2214317316300154
2. https://ieeexplore.ieee.org/document/8273700
3. https://www.researchgate.net/publication/318109025_Leaf_Disease_Detection_using_Image_Processing
4. https://www.sciencedirect.com/science/article/pii/S0168169918310202
5. https://pdfs.semanticscholar.org/9426/ae4ea4329521265c738e78221d1aff532537.pdf
6. https://scialert.net/fulltextmobile/?doi=itj.2011.267.275


 
