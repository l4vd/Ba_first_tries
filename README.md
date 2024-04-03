# Ba_first_tries

## Instruction:

- make sure that you have the following dir Structure:

![Image of Structure](Dir_struct.PNG)

- now run Network_Creation.ipynb
- next run the MetricCalc.java class in the java dir
  - run said class twice, once for each previously created gexf file (change the names of the outputfiles accordingly)
  - to nodes_real_train.csv and to edges_real_test.csv
- if you don't want to bother with the java dependencies include the metric calc in the Network_Clustering.ipynb file
  - for this endevour use the code included in the metric_calc_python.txt file
- next run the Network_Clustering.ipynb
- next run the Preprocess_Data...ipynb in the models/Original_MLP dir
- lastely run the MLP_collap.py file for running the network
  - the pytorch alternative can be found within the First_MLP.py file