# Computational_intelligence_final_project_NLP
the final project for computational intelligence course.the goal of this project was to train a model to do classification on text from diffrent persian newspaper articles.
this model was trained on around 150,000 samples and reached a test accuracy of around (85-84) percent on a completely seperate test set that was around 16.7 thousand articles.
due to the sheer size and the ownership rights of this data i sadly cannot add it to this project but i have prepared a sample of data that should help you understand the data better.
many diffrent methods were tested on this project and eventually the linearSVC model using TF-IDF vectorization was chosen as it showed the highest accuracy.
there were two other candidate models that are as following : 

-RNN (LSTM)

-NN (Global Average pooling on embeded vectors of words)

both of these models will also be placed in this repository and can be found in the candidate_models directory.

using the HAZM python library for persian NLP i normalized the spacings and tokenized using this library by passing these objects to the SKlearn TF-IDF library and rewriting the defaults values.


