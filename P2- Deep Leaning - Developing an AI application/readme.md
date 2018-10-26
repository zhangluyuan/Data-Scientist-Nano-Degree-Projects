## P2- Deep Leaning - Developing an AI application
Task: Develop a classification APP to identify flower classes based on images using PyTorch. <br>

The project contains 3 main parts. 
### 1. a jupyter notebook that performs training and prediction; 
#### load and transform data
-	load and transform data using torchvision’s dataset and transforms module
-	create dataloader suing torch.utils.data.DataLoader method
#### build and train a pre-trained model
 - load pretrained torchvision model. I chose ‘vgg19’. 
- Redefine the last unit of vgg19 the new linear hidden layer of 4096 units with dropout.
- Output layer has 102 units (102 types of flowers in the data set). Output is LogSoftmax
- Define criterion (nn.NLLLoss() here) and optimizer (torch.optim.Adam() here). 
- Train the model with learning_rate=0.001 and epochs=3.
- Training loss, validation loss, and validation accuracy is printed every 40 steps.
- Test the trained model with test data set. Test accuracy is 0.875.
- Save the checkpoint. Checkpoint dictionary includes: architecture name, class_to_idx, epochs, learning_rate, hidden_units, state_dict, optimizer, classifier.
#### Re-build a model from the saved checkpoint
- load checkpoint with torch.load,
- load pre-trained model with the model name from checkpoint and torchvision.models
- change last unit of the model to the classifier defined in checkpoint
- load state_dict from checkpoint to the model
- load class_to_idx from checkpoint to model.class_to_indx
- return the model for prediction
#### Predict a flower image with the rebuilt model
- Pre-process image using PIL’s Image model:  resize to 256 pixel,crop to 224x224, convert image to numpy array, normalized pixel values to 1, then standard scale pixel values using torchvision models’ expected mean and std, and finally reorder dimensions by moving color channel to the first dimension (color channel is the last dimension in Image object)
- Run the model for the processed image, and extract the top 5 categories. Convert the categories by mapping model index to classes, then classes to flower names. 
- Print out the flower image with its name, together with the top 5 predictions with the category names and probabilities in bars. 
### 2. a train.py python program. It performs similar tasks with the jupyter notebook, but with a lot of more flexibilities. 
- This program allows users to load and modify pre-trained torchvision models of their choice, and define hyper-parameters such as hidden units, number of labels, and learning rate. 
- It also asks users to input the image folder that the model will train on.
- Training loss, validation loss and accuracy (if validation data set is available) are printed during training. Test accuracy is also printed. 
- After training, the model’s checkpoint is saved to a folder that user designate. 
- The checkpoint can be used to rebuild the model for future use.
### 3. a predict.py python program. 
- This program will predict the type of flower in an image using the checkpoint that user provides.
- User also needs to provide category_to_name dictionary, so that predicted category/class can be mapped to the name of the flower. 
- A classification model is rebuilt from the checkpoint that user provides. The checkpoint should contain architecture name, structure of the last unit of the architecture, model’s state_dict, and model’s class_to_indx dictionary. 
- The image to predict is pre-processed by resizing to 256 pixels (the short side), and then cropped to 224x224 using PIL Image module. Then image is normalized to 1, and standard scaled. Color dimension is moved to the first dimension. 
- Top predictions (e.g. top 5) are printed with class name and probabilities.

