Observations:
-When adding the number of convolution layers + maxpooling, makes the image smaller and performs better 
 giving an above 90% accuracy of classification.
-When having just a single convolutional layer + single maxpooling, returns a bad accuracy of under 50%
-When applying too many convolutional filters seems to return the program a greater loss and poor accuracy
-When applying 4 convolutional layers + maxpooling, returns an error, since the image is too compressed that 
 we get a negative dimension.
-Applying specifically 2 convolutional layers + maxpooling, with 10 and 130 filters respectivly returns the
 program an accuracy of 97% and above. When using the same number of convolutions BUT only 16 filters in each
 one, we get a poor accuracy. 
-When I change the dropout to 0.9, we get a poor accuracy, since important nodes can get eliminated throughout
 the process.
-When applying a dropout of 0.4 - 0.5, we get an accuracy of 96%..
