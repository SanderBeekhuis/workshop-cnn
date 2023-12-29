# Programma middag:  

6.	Intro Neurale netwerken (30 min) (Remco) (bestaande presentatie)
7.	Praktisch neurale netwerken (met hele slechte performance op ‘ons dataset’ + mnist (45m) (Sander)
    a.	Recap
    b.	Input: Dataset die al opgedeeld is
    c.	DOEL: Neuraal netwerk getraind op ons dataset (truck/car bijvoobeeld),
    d.	Plaatje kunnen opsturen naar model
8.	CNN theorie (30 min) (Remco) (bestaande presentatie) 
    a.	Eventueel transfer learning
9.	CNN Praktisch (1u) (Sander)	
    a.	Input: Dataset uit 4
    b.	Doel: CNN getraind op ons dataset, waarbij we geen pretrained base gebruiken
10.	Afronding, discussie, recap


## Cnn theorie
- Base (convolution) + head (regular "dense" neural network)
  - Q is a regular NN the same as a NN with dense layers
- Base exists of 
  - convulutional layer with Relu activation
  - Maximum pooling layer

- How to go from base to head? 
  - The base yields X feature maps (i.e. 3D input)
  - The head wants/takes 1D input
  - Flatten? or GlobalAvgPool!
  - Advantage of GlobalAvgPool is reducing parameter space while, usually retaining enough information. 

- Stacking small kernels creates a large "receptive field" for a neuron.  Without exploding the parameters (i.e. the learned weights)
  - E.g 3 layers of 3x3 have 3\*9 paramerters. While a single 7\*7 has 49 parameters. For the same 7\*7 receptive field.

- Number of parameters is slightly largers then you would expect since we (at least in tf) feed in a trainable bias into every node.
  - https://stats.stackexchange.com/questions/459929/understanding-how-many-biases-are-there-in-a-neural-network#:~:text=There%27s%20one%20bias%20unit%20per,one%20layer%20has%20a%20bias.

- keras uses aliases sometimes.  So `MaxPool2D` is `MaxPooling2D`  This is documented in the keras docs under a collapsed header "aliases"

## Bronnen
https://www.kaggle.com/learn/computer-vision
github link: https://github.com/Kaggle/learntools/tree/master/learntools/computer_vision


Accompanying dataset https://www.kaggle.com/datasets/ryanholbrook/car-or-truck

## Afko's
vgg - visual geometry group.  Made vggnet, esp. vvgnet16. Of which we can use the pretrained base freely. 
- Vggnet 16  uses 16 convultional blocks

## TODO
- Better dataset?
- Run model on a sample?
- Turn work into 1/2 practical excerices.
  - Guide with e.g. number of parameters. 

## Te bespreken.  
-  Training times to long? 
-  What is a sound NN to start with?