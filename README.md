# AI Programming with Python Project. Image classification (102 flower categories) using Pytorch models.

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, code was developed for an image classifier built with PyTorch, then converted into command line applications: train.py, predict.py.

The image classifier recognizes different species of flowers. The dataset contains 102 flower categories.

In Image Classifier Project.ipynb Densenet121 from torchvision.models pre-trained models were used. It was loaded as a pre-trained network, based on which defined a new, untrained feed-forward network as a classifier, using ReLU activations and dropout. Trained the classifier layers using backpropagation using the pre-trained network to get the features. The loss and accuracy of the validation set were tracked to determine the best hyperparameters. 

## Dataset used :     
102 Category Flower Dataset     

http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html     
[Maria-Elena Nilsback](http://www.robots.ox.ac.uk/~men/) and [Andrew Zisserman](http://www.robots.ox.ac.uk/~az/)

## Neural Network used : 
* [Densenet121](https://resources.wolframcloud.com/NeuralNetRepository/resources/DenseNet-121-Trained-on-ImageNet-Competition-Data/)
* [VGG19](https://resources.wolframcloud.com/NeuralNetRepository/resources/VGG-19-Trained-on-ImageNet-Competition-Data/)
       
You can dowload Densenet121 [here](https://www.kaggle.com/code/leighplt/densenet121-pytorch)

You can dowload VGG19 [here](https://www.kaggle.com/code/madz2000/flowers-classification-using-vgg19-88-accuracy)

## Refresher on Neural Network :
[Gradient Descent](https://medium.com/secure-and-private-ai-writing-challenge/playing-with-gradient-descent-intuition-e5bde385078)   
[Backpropogation](https://medium.com/secure-and-private-ai-writing-challenge/playing-with-backpropagation-algorithm-intuition-10c42578a8e8)        


## Command line applications train.py and predict.py:

For command-line applications, there is an option to select either Alexnet or VGG13 models. 

Following arguments mandatory or optional for train.py 

1.	'data_dir'. 'Provide data directory. Mandatory argument', type = str
2.	'--save_dir'. 'Provide a saving directory. The optional argument', type = str
3.	'--arch'. 'Vgg13 can be used if this argument is specified, otherwise, Alexnet will be used', type = str
4.	'--learning_rate'. 'Learning rate, default value 0.001', type = float
5.	'--hidden_units'. 'Hidden units in Classifier. The default value is 2048', type = int
6.	'--epochs'. 'Number of epochs', type = int
7.	'--gpu'. "Option to use GPU", action = store_true

Following arguments mandatory or optional for predict.py

1.	'image_path'. 'Provide the path to the image. Mandatory argument', type = str
2.	'load_checkpoint'. 'Provide the path to the checkpoint. Mandatory argument', type = str
3.	'--top_k'. 'Top K most likely classes. Optional', type = int
4.	'--category_names'. 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str
5.	'--cpu'. "Option to use GPU. Optional", action = store_true

## Run on windows - 
Make sure you have installed Python, Pytorch and Jupyter Notebook.

* _Download all the folders and files_     
`git clone https://github.com/GauravG-20/Flower-Image-Classifier.git`              
* _Then open Visual Studio Code (or powershell) and change the directory to the path where all the files are located._       
`cd Flower-Image-Classifier`      
* _Now run the following commands_ -        
`python train.py [Dir of Flower dataset]`
This will first download the models and then will train the model with default parameters.       
`python predict.py [Dir of Image for prediction] [directory of the saved checkpoint]`

