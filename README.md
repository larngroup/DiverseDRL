# End-to-end deep reinforcement learning 

In this work, we explore the potential of deep learning to streamline the process of identifying new potential drugs through the computational generation of molecules with interesting biological properties. Two deep neural networks compose our targeted generation framework: \emph{the Generator}, which is trained to learn the building rules of valid molecules employing SMILES strings notation, and \emph{the Predictor} which evaluates the newly generated compounds by predicting their affinity for the desired target. Then, the Generator is optimized through Reinforcement Learning to produce molecules with bespoken properties.

The innovation of this approach is the exploratory strategy applied during the reinforcement training process that seeks to add novelty to the generated compounds. This training strategy employs two Generators interchangeably to sample new SMILES: the initially trained model that will remain fixed and a copy of the previous one that will be updated during the training to uncover the most promising molecules. The evolution of the reward assigned by the Predictor determines how often each one is employed to select the next token of the molecule. This strategy establishes a compromise between the need to acquire more information about the chemical space and the need to sample new molecules, with the experience gained so far.

To demonstrate the effectiveness of the method, the \emph{Generator} is trained to design molecules with high inhibitory power for the adenosine $A_{2A}$ and $\kappa$ opioid receptors. The results reveal that the model can effectively modify the biological affinity of the newly generated molecules towards the craved direction. More importantly, it was possible to find auspicious sets of unique and diverse molecules, which was the main purpose of the newly implemented strategy. 

## Getting Started

The entire implementation of this molecule generation framework was implemented in Keras with Tensorflow backend.

### Prerequisites

 In order to get started you will need: 

- CUDA 9.0
- NVIDIA GPU
- Keras
- Tensorflow 1.13
- RDKit
- Scikit-learn
- Bunch
- Numpy
- tqdm

## General methodology

The general methodology for single-objective optimization, referring to individual properties, is described in the following image.



### ![pipeline](https://github.com/larngroup/DiverseDRL/blob/master/Figures/pipline.png?raw=true)



### Generator

It is a model with RNN architecture initially trained through supervised learning to discover, through SMILES strings, the required rules to generate valid molecules. After the first training phase, 95% of the generated molecules were synthesizable, and the diversity of these molecules was also greater than 90%, with a low percentage of repeated molecules.

### Predictor

This is a QSAR model whose function is to make a mapping between the generated molecules and the property that we want to optimize. This model was implemented using RNNs and SMILES strings for the architecture and the molecule descriptor, respectively. However, different alternatives to QSAR were tested - such as Fully connected neural network, SVR, RF, and KNN - and all of them having the ECFP as molecules descriptor. The comparison with these alternatives allowed us to conclude that the SMILES-based approach provided a more robust and reliable regressive model.

### Reinforcement Learning

At this point, the objective was to develop a versatile method that would allow the Generator to adapt to the requested property and generate molecules with that property, modified as desired. This has been done successfully for properties such as affinity for the K-opioid receptor and quantitative estimate of drug-likeness. The RL method used was policy-gradient-based, that is, the search for the optimal policy was made directly. Therefore, it was possible to bias the Generator so that it generates compounds with greater or lesser affinity for KOR and with higher QED values, independently. Nevertheless, with our exploratory strategy it was possible to achieve this objective and maintain the variability and synthesizability of the compounds at very acceptable rates.

![kor_max](https://github.com/larngroup/DiverseDRL/blob/master/Figures/kor_max.png?raw=true)

![qed_max](https://github.com/larngroup/DiverseDRL/blob/master/Figures/qed_max.png?raw=true)



The implemented strategy allowed to intercalate the used Generator to predict the next token between the most recently updated Generator (which favored exploitation) and the unbiased Generator (which favored exploration). The choice between the two was made by generating a random number and checking if it was greater or less than a threshold. This threshold varies depending on the evolution of the reward up to that point. If it is increasing, the threshold remains low, however, if the reward did not evolve as expected, the threshold was increased to ensure more exploration.



​				![generation](https://github.com/larngroup/DiverseDRL/blob/master/Figures/generation.png?raw=true)





### Multi-Objective Optimization

To address the optimization of two objectives simultaneously, three different strategies were implemented. However, all of them had as their basic principle to transform the problem from multi-objective to single-objective. In other words, the objective was to scale the vector of rewards, in a single numerical reward. The strategies followed were:

- Linear scalarization
- Weight search - linear scalarization
- Chebyshev scalarization

In the first case, weights were uniformly assigned between 0 and 1 to each objective with a step of 0.1

In the second case, a method was used to more precisely determine the weights of each objective in order to more effectively approach the Pareto front.

In the third case,  we used a non-linear scalarization method allowing the identification of Pareto front policies that are not in the convex part of these areas. Weights were also uniformly assigned between 0 and 1.

## Future Work

The next steps are to test the framework with different datasets of interest and, on the other hand, to optimize other properties that guarantee the specificity of the drugs for a given receptor.

Also, it may be important to implement a more robust QSAR model, with more robust descriptors and indicating whether or not a potential drug will be able to cross the blood-brain barrier.

