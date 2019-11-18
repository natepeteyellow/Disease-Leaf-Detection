# Disease-Leaf-Detection

This repo was created to store our project files from the Kaggle "PlantVillage" dataset (https://www.kaggle.com/xabdallahali/plantvillage-dataset). The dataset contains three types of folders:

1. color
2. grayscale
3. segmented

We will first train 6 different architectures on the color dataset. After we have validated the success of various architectures from PyTorch, we will use different data augmentations to train the models on. 

For each of the architectures, it is critical to change the final layer of the chosen model. Though the PyTorch tutorial uses a fc final layer for the resnet, all other architectures will have varying final layer parameters. Please make sure to reference the final_project_data_prep.ipynb to see examples of how the final layers can be modified. 

Different Networks:
1. AlexNet- Nathen
2. DenseNet- Nathen
3. ResNet- Sahithi
4. VGG - Sahithi
5. Inception- Jey
6. SqueezeNet- Jey

(7) Generative Adversarial Network (GAN)- Nathen
 