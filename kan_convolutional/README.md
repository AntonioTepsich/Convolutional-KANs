# Convolutional Kolmogorov-Arnold Network (CKAN) 
### Introducing Convolutional KAN Networks!
This project extends the idea of the innovative architecture of Kolmogorov-Arnold Networks (KAN) to the Convolutional Layers, changing the classic linear transform of the convolution to non linear activations in each pixel. AGREGAR ALGUNA LICENCIA? 
### Authors
This repository was made by Alexander Bodner, Antonio Tepsich, Jack Spolski and Santiago Pourteau.
AGREGAR ALGUN CONTACTO? MAIL? REDES? GITHUB DE TODOS?
### Credits
This repository uses an efficient implementation of KAN [here](https://github.com/Blealtan/efficient-kan). 
The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan).
The original paper of the KAN is available [here](https://arxiv.org/pdf/2404.19756).

### What is a KAN? ESCRIBIR BIEN, PONER ALGUNA ECUACION BASICA
KANs are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations just like MLPs: MLPs are based on the universal approximation theorem, while KANs are based on Kolmogorov-Arnold representation theorem. KANs and MLPs are dual: KANs have activation functions on edges, while MLPs have activation functions on nodes. KAN seems to be more parameter efficient than MLPs, but each KAN Layer has more parameters than a MLP layer. 
### What is a KAN Convolution?
KAN Convoluions are very similar to convolutions, but instead of applying the dot product between the kernel and the corresponding pixels in the image, we apply a Non Linear function to each element, and then add them up. The kernel of the KAN Convolution is equivalent to a KAN Linear Layer of 4 inputs and 1 output neuron. For each input i, we apply a ϕ_i learnable function, and the resulting pixel of that convolution step is the sum of ϕ_i(x_i). This can be visualized in the following figure.
![image](https://github.com/AntonioTepsich/ckan/assets/61150961/df79e546-a343-4396-8d1d-f04a8d4d62d3)


### --------------------------------ACA IMG LO DE LAS KAN--------------------------------
<img width="1163" alt="mlp_kan_compare" src="https://github.com/KindXiaoming/pykan/assets/23551623/695adc2d-0d0b-4e4b-bcff-db2c8070f841">

For more information about this novel architecture please visit:
- The official Pytorch implementation of the architecture: https://github.com/KindXiaoming/pykan
- The research paper: https://arxiv.org/abs/2404.19756

## Preliminary Evaluations

### --------------------------------ACA EXPLICAR EXPERIMENTOS--------------------------------
DECIR QUE LO HICIMOS RAPIDO BASICAMENTE Y QUE SEGUIMOS EVALUANDO ARQUITECTURAS Y ETC, POR ESO ES PRELIMINAR. TAMBIEN PONDRIA RESULTADOS LLAMATIVOS ARRIBA DE TODO
The implementation of Kolmogorov-Arnold Q-Network (KAQN) offers a promising avenue in reinforcement learning. In this project, we replace the Multi-Layer Perceptron (MLP) component of Deep Q-Networks (DQN) with the Kolmogorov-Arnold Network. Furthermore, we employ the Double Deep Q-Network (DDQN) update rule to enhance stability and learning efficiency.

The following plot compare DDQN implementation with KAN (width=8) and the classical MLP (width=32) on the `CartPole-v1` environment for 500 episodes on 32 seeds (with 50 warm-ups episodes).

<img alt="Epsisode length evolution during training on CartPole-v1" src="https://raw.githubusercontent.com/riiswa/kanrl/main/cartpole_results.png">

The following plot displays the interpretable policy learned by KAQN during a successful training session.

<img alt="Interpretable policy for CartPole" src="https://raw.githubusercontent.com/riiswa/kanrl/main/policy.png">

- **Observation**: KAQN exhibits unstable learning and struggles to solve `CartPole-v1` across multiple seeds with the current hyperparameters (refer to `config.yaml`).
- **Next Steps**: Further investigation is warranted to select more suitable hyperparameters. It's possible that KAQN encounters challenges with the non-stationary nature of value function approximation. Consider exploring alternative configurations or adapting KAQN for policy learning.
- **Performance Comparison**: It's noteworthy that KAQN operates notably slower than DQN, with over a 10x difference in speed, despite having fewer parameters. This applies to both inference and training phases.
- **Interpretable Policy**: The learned policy with KANs is more interpretable than MLP, I'm currently working on extraction on interpretable policy...


**MNIST:** ~97% accuracy after about 20 epochs. 
```
Epoch 1, Train Loss: 1.1218, Test Loss: 0.4689, Test Acc: 0.91
Epoch 2, Train Loss: 0.3302, Test Loss: 0.2599, Test Acc: 0.93
Epoch 3, Train Loss: 0.2170, Test Loss: 0.2359, Test Acc: 0.94
Epoch 4, Train Loss: 0.1696, Test Loss: 0.1857, Test Acc: 0.95
Epoch 5, Train Loss: 0.1422, Test Loss: 0.1574, Test Acc: 0.96
Epoch 6, Train Loss: 0.1241, Test Loss: 0.1597, Test Acc: 0.95
Epoch 7, Train Loss: 0.1052, Test Loss: 0.1475, Test Acc: 0.96
Epoch 8, Train Loss: 0.0932, Test Loss: 0.1321, Test Acc: 0.96
Epoch 9, Train Loss: 0.0879, Test Loss: 0.1553, Test Acc: 0.95
Epoch 10, Train Loss: 0.0780, Test Loss: 0.1239, Test Acc: 0.96
Epoch 11, Train Loss: 0.0722, Test Loss: 0.1283, Test Acc: 0.96
Epoch 12, Train Loss: 0.0629, Test Loss: 0.1236, Test Acc: 0.96
Epoch 13, Train Loss: 0.0612, Test Loss: 0.1271, Test Acc: 0.96
Epoch 14, Train Loss: 0.0521, Test Loss: 0.1390, Test Acc: 0.96
Epoch 15, Train Loss: 0.0488, Test Loss: 0.1374, Test Acc: 0.96
Epoch 16, Train Loss: 0.0487, Test Loss: 0.1309, Test Acc: 0.96
Epoch 17, Train Loss: 0.0416, Test Loss: 0.1253, Test Acc: 0.96
Epoch 18, Train Loss: 0.0402, Test Loss: 0.1346, Test Acc: 0.96
Epoch 19, Train Loss: 0.0373, Test Loss: 0.1199, Test Acc: 0.97
Epoch 20, Train Loss: 0.0346, Test Loss: 0.1434, Test Acc: 0.96
Epoch 21, Train Loss: 0.0314, Test Loss: 0.1142, Test Acc: 0.97
Epoch 22, Train Loss: 0.0285, Test Loss: 0.1258, Test Acc: 0.97
Epoch 23, Train Loss: 0.0289, Test Loss: 0.1192, Test Acc: 0.97
```
![MNIST](img/MNIST.png)
The network parameters are [28*28, 32, 16, 10] with 4 degree Chebyshev polynomials.

It needs a low learning rate (2e-4) to train. The network is very sensitive to the learning rate.

Note that it's still not as good as MLPs. Detailed comparison is on the way.

---

~~**Function Interpolation:** much better than MLPs when the function is (mostly) smooth, very effective in discovering mathematical laws.~~

![alt text](img/Interpolation.png)
~~ChebyKAN: [1, 8, 1] with 8 degree.~~
~~MLP: [1, 1024, 512, 1] with ReLU~~

**Edit: The comparison above is not fair.**
**Thanks @usamec for pointing out the mistake that the MLP was too big and not trained properly.**

<!-- ~~Edit: Adding noise to the data does not affect the ChebyKAN's performance.~~ -->

<!-- ![alt text](img/Interpolation_noise.png) -->

---

**Fixed version:**

**Function Interpolation:** converge faster than MLPs when the function is (mostly) smooth.


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



# Instalation
```bash
git clone git@github.com/AntonioTepsich/ckan.git
cd ckan
pip install -r requirements.txt
```
# Usage
### --------------------------------ACA EXPLICAR USO Y EJEMPLO--------------------------------
Just copy `ChebyKANLayer.py` to your project and import it.
```python
from ChebyKANLayer import ChebyKANLayer
```
# Example
Construct a ChebyKAN for MNIST
```python
class MNISTChebyKAN(nn.Module):
    def __init__(self):
        super(MNISTChebyKAN, self).__init__()
        self.chebykan1 = ChebyKANLayer(28*28, 32, 4)
        self.ln1 = nn.LayerNorm(32) # To avoid gradient vanishing caused by tanh
        self.chebykan2 = ChebyKANLayer(32, 16, 4)
        self.ln2 = nn.LayerNorm(16)
        self.chebykan3 = ChebyKANLayer(16, 10, 4)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        x = self.chebykan1(x)
        x = self.ln1(x)
        x = self.chebykan2(x)
        x = self.ln2(x)
        x = self.chebykan3(x)
        return x
```
**Note:** Since Chebyshev polynomials are defined on the interval [-1, 1], we need to use tanh to keep the input in that range. We also use LayerNorm to avoid gradient vanishing caused by tanh. Removing LayerNorm will cause the network really hard to train.

Have a look at `Cheby-KAN_MNIST.ipynb`, `Function_Interpolation_Test.ipynb`, and `Multivar_Interpolation_Test.ipynb` for more examples.


## Contributing
We invite the community to join us in advancing this project. There are numerous ways to contribute. You are welcome to contribute by submitting pull requests or opening issues to share ideas and suggest enhancements. Together, we can unlock the full possibilities of KAN and push the boundaries of Computer Vision ❤️.
