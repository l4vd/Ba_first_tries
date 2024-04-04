# Ba_first_tries

## Revelations:

As it appears the paper uses f1-macro instead of f1-score this is due to the imbalanced nature of the dataset.
Therefore, it would only seem reasonable to use macro Precision as well as macro Recall as well.
Furthermore as it turns out my results for this case aren't off by as much as before but still way off:


| Metric    | Currently | Goal  |
|-----------|-----------|-------|
| precision | 0.51      | 0.438 |
| recall    | 0.55      | 0.508 |
| f1-macro  | 0.49      | 0.619 |
| auc-roc   | 0.5975    | 0.699 |

these results can also be found for within the output_collab.txt file bellow "10iter:"

the goals are from Collaboration Aware Hit Song Prediction

## Instruction:

- please start by reading the revelations


- for the environment initialisation you can use the included yaml file (Ba_first_tries.yml)
- if you want to skip the data preprocessing you can download the dataset I used for running the network using the following sharelink: https://1drv.ms/u/s!AjbyAXSURZ1bkJpuxpxWyLvrfxXuBA?e=uunVdA
- download MusicOSet through the following link: https://zenodo.org/records/4904639
  - the link can also be found in the paper "Collaboration-Aware Hit Song Prediction"
- make sure that you have the following dir Structure:

![Image of Structure](Dir_struct.PNG)

- now run Network_Creation.ipynb
- next run the MetricCalc.java class in the java dir
  - run said class twice, once for each previously created gexf file (change the names of the outputfiles accordingly)
  - to nodes_real_train.csv and to edges_real_test.csv
- if you don't want to bother with the java dependencies use the project version after the commit "without java" 
  - in that case just skip running the java class and go on with the next step
- next run the Network_Clustering.ipynb
- next run the Preprocess_Data...ipynb in the models/Original_MLP dir
- lastely run the MLP_collap.py file for running the network
  - the pytorch alternative can be found within the First_MLP.py file
