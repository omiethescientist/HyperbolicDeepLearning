# HyperbolicDeepLearning
A set of hyperbolic neural networks for bioinformatic research 

Poincare Varitaional Autoencoder based on Emile Mathieu's work:

https://github.com/emilemathieu/pvae/tree/master/pvae

https://arxiv.org/abs/1901.06033

Current Command Line Implementation:

python3.7 main.py path/to/data --input-dim 24000 --latent-dim 2 --output-model ./model.pth --output-valid ./validation.pdf --labels 1 --hidden-lay 2 --hidden-size 256 --drop-rate 0.2 --curvature 2.0 --lr 1e-3 --epochs 100 --batch-size 32 --workers 1 --valid-split 0.33 --double 1 --cuda 0

Parameters:

Input Path

Input Dimensions: Number of Genes

Latent Dimensions: Number of latent variables you want

Output Model: Model Output Path

Output Valid: Validation Output Path

Labels: Does your data matrix have a column at the end for labels [0,1]

Hidden Layers: Number of hidden layers in the PVAE Encoder and Decoder

Hidden Size: Size of the hidden layers

Drop Rate: Dropout rate for dropout layers

Curvature: Negative curvature of the hyperbolic latent space (Increasing this increases the sparcity of the points on the 
latent space usually)

lr: Learning rate for optimizer (RiemannAdam)

epochs: number of epochs for training (PVAE converges quickly, so consider using a low number for early stopping)

batch size: Data batches for data loader

workers: number of workers for datalaoder

valid split: The ratio of test to train in the data

double: Using float64 (Highly reccomended for numerical stability)

cuda: use gpu

To Do:

Work on using KNNG as input layer to create regular autoencoder based on Klimovskaia et. al 2019 

https://github.com/facebookresearch/PoincareMaps

Work on creating a hyperbolic recurrent neural network in pytorch for TFBS prediction based on Ganea et. al 2018

Contact:
kanaomar@msu.edu
