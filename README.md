# End-to-end deep reinforcement learning 

The long period of time, the enormous financial cost of bringing a new drug onto the market, and the uncertainty about whether or not it will be accepted by the responsible authorities are clear obstacles to the development of new drugs.

Deep Learning techniques at the early stages of the drug discovery process can help to identify candidate drugs with biological properties of interest, reduce the vast research space of drug-like compounds, and minimize all the inherent issues. However, many of the works that use these techniques focus on optimizing a specific property, which is scarce for drug development since this problem requires a solution that satisfies more than one objective.
Our approach aims to implement a framework for the targeted generation of molecules designed to optimize different properties. These properties are biological and psychochemical-based. We aim to design candidate drugs that can perform the desired function and have harmless effects for the organism. As a generator model, we implemented a recurrent neural network that was trained to learn the building rules of production valid molecules in the form of SMILES strings. The generator model was then re-trained through Reinforcement Learning to produce molecules with bespoke properties. To evaluate the newly generated molecules we implemented a second recurrent neural network model that works like a QSAR model, mapping the molecular structure to the desired biological property. The novelty of this approach is the exploratory strategy that ensures, throughout the training process, a compromise between the need to acquire information about the environment and the need to make the correct decisions taking into account the goal. 

To demonstrate the effectiveness of the method, we started by biasing the generator model to address single-objectives such as the affinity for the k-opioid receptor (KOR) and maximizing the Quantitative Estimate of Druglikeness  (QED) . In addition, we explored different techniques to implement multi-objective optimization methods to generate molecules that satisfy both biological and physicochemical properties.
Regarding the single-objective experiments, the optimized model was able to generate molecules with the previously mentioned properties biased towards the right direction, maintaining the percentage of valid molecules. 

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



### ![pipeline](C:\Users\Tiago\Pictures\Saved Pictures\article\pipeline.jpg)



### Generator

It is a model with RNN architecture initially trained through supervised learning to discover, through SMILES strings, the required rules to generate valid molecules. After the first training phase, 95% of the generated molecules were synthesizable, and the diversity of these molecules was also greater than 90%, with a low percentage of repeated molecules.

### Predictor

This is a QSAR model whose function is to make a mapping between the generated molecules and the property that we want to optimize. This model was implemented using RNNs and SMILES strings for the architecture and the molecule descriptor, respectively. However, different alternatives to QSAR were tested - such as Fully connected neural network, SVR, RF, and KNN - and all of them having the ECFP as molecules descriptor. The comparison with these alternatives allowed us to conclude that the SMILES-based approach provided a more robust and reliable regressive model.

### Reinforcement Learning

At this point, the objective was to develop a versatile method that would allow the Generator to adapt to the requested property and generate molecules with that property, modified as desired. This has been done successfully for properties such as affinity for the K-opioid receptor and quantitative estimate of drug-likeness. The RL method used was policy-gradient-based, that is, the search for the optimal policy was made directly. Therefore, it was possible to bias the Generator so that it generates compounds with greater or lesser affinity for KOR and with higher QED values, independently. Nevertheless, with our exploratory strategy it was possible to achieve this objective and maintain the variability and synthesizability of the compounds at very acceptable rates.

![kor_max](https://github.com/larngroup/DiverseDRL/blob/master/Figures/kor_max.png?raw=true)

![qed_max](https://github.com/larngroup/DiverseDRL/blob/master/Figures/qed_max.png?raw=true)



The implemented strategy allowed to intercalate the used Generator to predict the next token between the most recently updated Generator (which favored exploitation) and the unbiased Generator (which favored exploration). The choice between the two was made by generating a random number and checking if it was greater or less than a threshold. This threshold varies depending on the evolution of the reward up to that point. If it is increasing, the threshold remains low, however, if the reward did not evolve as expected, the threshold was increased to ensure more exploration.



​				![generation](C:\Users\Tiago\Pictures\Saved Pictures\dissertation\generation.png)





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

