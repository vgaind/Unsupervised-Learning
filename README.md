# Unsupervised Learning
Use the process flag to run the code

process_flag: 1=clustering 2=dim red 3=dim red+cluster 4=dim red+ nn 5=cluster + nn

Combination algorithms (process flag >2) require pipelining one algorithms output to another
When running process_flag=3:
 edit pipeline_estimator_name (line 755) to [‘PCA’,’ICA’,’Random_Projection’,’Dictionary_Learning’]

When running process_flag=4:
 edit pipeline_estimator_name (line 755) to [‘PCA’,’ICA’,’Random_Projection’,’Dictionary_Learning’]

When running process_flag=5:
 edit pipeline_estimator_name (line 755) to [‘Kmeans’,’EM’]

