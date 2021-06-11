
## 目录
1. [Dataset Declaration](#Dataset Declaration)
2. [Environment](#Environment)
3. [How2train](#How2train)
4. [How2eval](#How2eval)


## Dataset Declaration
The marked name in the dataset is chinese phonetic, 
->Qinggao corresponds to Morally_Lofty
->Xianshi to leisure
->Yinyi to Seclusion

## Environment
tensorflow-gpu==1.13.1   
keras==2.1.5   

## How2train
1. The pictures stored in "DataSets" folder are divided into two parts: "Train" is the training picture, and "Test" is the test picture. 
2. Before training, you need to first prepare the data set, and create different folders in the train or test file. The name of each folder is the corresponding category name, and the picture under the folder is the picture of this class.The file format is as follows:
```
|-datasets
    |-train
        |-cat
            |-123.jpg
            |-234.jpg
        |-dog
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |-cat
            |-567.jpg
            |-678.jpg
        |-dog
            |-789.jpg
            |-890.jpg
        |-...
```
3. After preparing the data set, it is necessary to run txt_annotation.py in the root directory to generate the cls_train.txt required for training. Before running, it is necessary to modify the classes in it and change them into the classes it needs to be divided into.   
4. Then modify the cls_classes.txt in the model_data folder so that it also corresponds to the classes you need to be divided into. 
5. After adjusting the network and weight you want to choose in train.py, you can start training!

## How2eval
1. The pictures stored in "DataSets" folder are divided into two parts: "Train" is the training picture, and "Test" is the test picture. When evaluating, we use the pictures in "Test" folder.  
2. Before the evaluation, you need to prepare the data set first. Create different folders in the train or test file. The name of each folder is the corresponding category name, and the image under the folder is the image of the class.The file format is as follows:
```
|-datasets
    |-train
        |-cat
            |-123.jpg
            |-234.jpg
        |-dog
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |-cat
            |-567.jpg
            |-678.jpg
        |-dog
            |-789.jpg
            |-890.jpg
        |-...
```
3. After the data set is prepared, you need to run txt_annotation.py in the root directory to generate the cls_test.txt needed for evaluation. Before running, you need to modify the classes in it and change them into the classes you need to divide into.   
4. Then modify the following parts of model_path, classes_path, backbone and alpha in classifying.py file to make them correspond to the trained files;**model_path corresponds to the weight file under logs folder, classes_path is the corresponding class of model_path, the backbone feature extraction network used by backbone, alpha is the alpha value ** when using mobilenet.
5. Run eval_top1.py and eval_top5.py to evaluate the model accuracy.
