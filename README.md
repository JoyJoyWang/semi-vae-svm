# VAE + SVM on FashionMNIST

This project implements a combination of a Variational Autoencoder (VAE) and Support Vector Machine (SVM) to classify the FashionMNIST dataset. The VAE is used for feature extraction, and the extracted latent features are then classified using SVM. The notebook `A3 DL Q4 VAE+SVM.ipynb` contains all the code to train, save, and test the models.

## Files in the Project

- **A3 DL Q4 VAE+SVM.ipynb**: Jupyter notebook containing the full code to train and test the VAE and SVM models.
- **svm_fashionmnist_100.pkl**: Pre-trained SVM model for 100 labeled examples.
- **svm_fashionmnist_600.pkl**: Pre-trained SVM model for 600 labeled examples.
- **svm_fashionmnist_1000.pkl**: Pre-trained SVM model for 1000 labeled examples.
- **svm_fashionmnist_3000.pkl**: Pre-trained SVM model for 3000 labeled examples.
- **vae_fashionmnist_100.pth**: Pre-trained VAE model for 100 labeled examples.
- **vae_fashionmnist_600.pth**: Pre-trained VAE model for 600 labeled examples.
- **vae_fashionmnist_1000.pth**: Pre-trained VAE model for 1000 labeled examples.
- **vae_fashionmnist_3000.pth**: Pre-trained VAE model for 3000 labeled examples.

## Training the Models

To train the models using different number of labels, you can use the blocks in the notebook **A3 DL Q4 VAE+SVM.ipynb**. 

There are 4 sections regarding different number of labels (100 labels, 600 labels, 1000 labels, 3000 labels), run all codes at the beginning to load dataset and set up, then run these section to train.

The process is divided into two steps:

### 1. **Train the VAE**

The VAE is trained on the FashionMNIST dataset. The latent features extracted by the encoder are then used to train an SVM.

- Run the corresponding code block to train the VAE.
- Once trained, the VAE weights are saved in `.pth` files.

### 2. **Train the SVM**

Once the latent representations are extracted using the VAE, these features are used to train the SVM classifier. The SVM is trained on the latent features for different numbers of labeled examples (100, 600, 1000, and 3000).

- After training, the SVM models are saved as `.pkl` files using `joblib`.

## Testing the Models

To test the pre-trained models, load the saved VAE and SVM models, extract the latent features from the test data, and use the SVM to make predictions. The test accuracy is then computed.

You can find **a section named "TEST the model"**, then run all blocks under this section to test the model. You can change the paths of checkpoints here:

```
model.load_state_dict(torch.load('vae_fashionmnist_3000.pth'))  # change to the checkpoint you want to test.

model.eval()

import joblib

clf = joblib.load('svm_fashionmnist_3000.pkl') # change to the checkpoint you want to test.
```

