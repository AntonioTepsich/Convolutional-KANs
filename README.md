# Convolutional Kolmogorov-Arnold Network (CKAN) 
### Introducing Convolutional KAN Networks!
This project extends the idea of the innovative architecture of Kolmogorov-Arnold Networks (KAN) to the Convolutional Layers, changing the classic linear transformation of the convolution to non linear activations in each pixel. AGREGAR ALGUNA LICENCIA? 
### Authors
This repository was made by:
 - Alexander Bodner | abodner@udesa.edu.ar | [Twitter](https://twitter.com/AlexBodner_) | [LinkedIn](https://www.linkedin.com/in/alexanderbodner/)
 - Antonio Tepsich | atepsich@udesa.edu.ar | [Twitter](https://twitter.com/antotepsich) | [LinkedIn](https://www.linkedin.com/in/antonio-tepsich/)
 - Jack Spolski | jspolski@udesa.edu.ar | [LinkedIn](https://www.linkedin.com/in/jack-spolski-9882a3196/)
 - Santiago Pourteau | spourteau@udesa.edu.ar | [Twitter](https://twitter.com/SantiPourteau) | [LinkedIn](https://www.linkedin.com/in/santiago-pourteau-1bba8619a/)

### Credits
This repository uses an efficient implementation of KAN which is available [here](https://github.com/Blealtan/efficient-kan). 
The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan).
The original paper of the KAN is available [here](https://arxiv.org/pdf/2404.19756).

### What is a KAN?
KANs are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations just like MLPs: MLPs are based on the universal approximation theorem, while KANs are based on Kolmogorov-Arnold representation theorem. KANs and MLPs are dual: KANs have activation functions on edges, while MLPs have activation functions on nodes. KAN seems to be more parameter efficient than MLPs, but each KAN Layer has more parameters than a MLP layer. 

<img width="1163" alt="mlp_kan_compare" src="https://github.com/KindXiaoming/pykan/assets/23551623/695adc2d-0d0b-4e4b-bcff-db2c8070f841">

For more information about this novel architecture please visit:
- The official Pytorch implementation of the architecture: https://github.com/KindXiaoming/pykan
- The research paper: https://arxiv.org/abs/2404.19756

### What is a KAN Convolution?
KAN Convoluions are very similar to convolutions, but instead of applying the dot product between the kernel and the corresponding pixels in the image, we apply a Non Linear function to each element, and then add them up. The kernel of the KAN Convolution is equivalent to a KAN Linear Layer of 4 inputs and 1 output neuron. For each input i, we apply a ϕ_i learnable function, and the resulting pixel of that convolution step is the sum of ϕ_i(x_i). This can be visualized in the following two figures.

![image](./images/Convs.png)

## Preliminary Evaluations

### Experiments

The implementation of KAN Convolutions is a promosing idea, although it is still in its early stages. We have conducted some preliminary experiments to evaluate the performance of KAN Convolutions. The reason we say preliminary is because we wanted to publish this idea as soon as possible, so that the community can start working on it. We are aware that there are many hyperparameters to tune, and many experiments to conduct. In the coming days and weeks we will be tuning the hyperparameters of our model and the models we use to compare. We also recognize that we have not used large or complicated datasets. We will be conducting experiments on more complex datasets in the future, this implies that the amount parameters of the KANS will increase since we will need to implement more Kan Convolutional layers. At the moment we aren't seeing a significant improvement in the performance of the KAN Convolutional Networks compared to the traditional Convolutional Networks. We believe that this is due to the fact that we are using simple datasets and simple models. We are confident that as we increase the complexity of the models and the datasets we will see a significant improvement in the performance of the KAN Convolutional Networks.

The different architectures we have tested are:
- KAN Convolutional Layers connected to Kan Linear Layers (KKAN)
- Kan Convolutional Layers connected to a MLP (CKAN)
- CKAN with Batch Normalization between convolutions (CKAN_BN)
- Large ConvNet (Classic Convolutinos connected to a MLP) (ConvNet)
- Small ConvNet (SimpleCNN)
- One Layer MLP (SimpleLinear)

# GRAFICOS, TABLAS Y CONCLUSIONES

---

# CORREGIR ChebyKan y fijarse imagenes y corregir 
**Fixed version:**

**Function Interpolation:** converge faster than MLPs when the function is (mostly) smooth.
### Parameters in a KAN Convolution
Suppose that we have a KxK kernel. In this case, for each element of this matrix we have a ϕ, which its parameter count is: gridsize + 1. For implementation issues, eficcient kan defines:  
![equation](https://github.com/AntonioTepsich/ckan/assets/61150961/074990fb-88c8-4498-93ac-7055f7755535)

This gives more expresability to the activation function b. So the parameter count for a linear layer is gridsize + 2. So in total we have K²(gridsize + 2) parameters for KAN Convolution, vs only K² for a common convolution.

![alt text](img/Interpolation_fix.png)

ChebyKAN: [1, 8, 1] with 8 degree.
MLP: [1, 128, 1] with Tanh.

With decent training, the MLP can achieve similar performance as ChebyKAN. Note that ChebyKAN shows some overfitting.

However ChebyKAN converges much faster than MLP.

![alt text](img/Convergence_Speed.png)

ChebyKAN: Adam, lr=0.01.
MLP: Adam, lr=0.03.

@5000 epoch, ChebyKAN has already converged, while MLP is still far from convergence.

![alt text](img/Early_Stopping.png)



# Installation
```bash
git clone git@github.com/AntonioTepsich/ckan.git
cd ckan
pip install -r requirements.txt
```
# Usage
Just copy `train_model.py` to your project and import it.
```python
from ChebyKANLayer import ChebyKANLayer
```
# Example
Construct a ChebyKAN for MNIST
```python
class KKAN_Convolutional_Network(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size= (3,3),
            device = device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size = (3,3),
            device = device
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 
        
        # self.linear1 = nn.Linear(625, 256)
        # self.linear2 = nn.Linear(256, 10)
        self.kan1 = KANLinear(
            625,
            256,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1],
        )
        self.kan2 = KANLinear(
            256,
            10,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1],
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)

        x = self.flat(x)
        x = self.kan1(x) 
        x = self.kan2(x)
        return x
```
**Note:** Since Chebyshev polynomials are defined on the interval [-1, 1], we need to use tanh to keep the input in that range. We also use LayerNorm to avoid gradient vanishing caused by tanh. Removing LayerNorm will cause the network really hard to train.

Have a look at `Cheby-KAN_MNIST.ipynb`, `Function_Interpolation_Test.ipynb`, and `Multivar_Interpolation_Test.ipynb` for more examples.


## Contributing
We invite the community to join us in advancing this project. There are numerous ways to contribute. You are welcome to contribute by submitting pull requests or opening issues to share ideas and suggest enhancements. Together, we can unlock the full possibilities of KAN and push the boundaries of Computer Vision ❤️.
