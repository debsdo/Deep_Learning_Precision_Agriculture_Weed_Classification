# Deep_Learning_Precision_Agriculture_Weed_Classification

This is a Deep learning based project in Precision Agriculture to distinguish weeds from crops using state-of-the-art classification models like VGG16, VGG19, ResNet50, MobileNet etc.

Keras Transfer Learning Suite has been used to train the state-of-the-art classification models.

# The codes can be used by anyone who wishes to use Keras pre-trained models for fine tuning purposes. 
Just replace the directory names, training samples, validation samples and other parameters.

The Top layer of any model has not been included. Rather a GlobalMaxPooling2D layer, Dense layer, Dropout layer and a final Dense layer(parameters depend upon the number of classes) have been added to the top of the base model.
