# Architecture

## Dataset architecture

To retrieve and process the dataset used for this experiment, we will try to replicate medallion architecture. This pattern is used to be able to organize, track and analyze data easily. The architecture will be consist of bronze (raw), silver(interim), and gold layer(train/test).

### Bronze Layer
In this layer data is downloaded and stored in the data repository. Focus only on extracting data from multiple source and stored it raw.

### Silver Layer
This layer will store transformed data, unifiying (hopefully) into single format, which later can be transformed into the format need on the gold layer.

### Gold Layer
From silver layer, we transform the data into a usable training data for machine learning model. For example for ocr task, we will adjust what is existing on silver layer into the text training data format or the YOLO format.