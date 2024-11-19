# aws-abid-count-task
Project to Explore different model architectures for the count task proposed in the ABID challenge using AWS 

## Project Proposal
### Domain background
Facilities like Amazon Fulfillment centers handle shipping for a variety of products in massive quantities, thus there is a need to apply verification systems and control systems to product outflows. 
The Amazon Bin Challenge provided a dataset and 3 tasks for people to contribute to get better solutions to this problem.
Project is an experiment to explore deeper into vision systems applied to logistic problems, a particular area of interest and importance.

### Problem Statement
The problem to be solve is improving accuracy and reduce error in counting object instances in the bin. 

### Solution Statement
The project will attempt to improve results from the baseline solution proposed by the [Challenge Github Page](https://github.com/awslabs/open-data-docs/tree/main/docs/aft-vbi-pds).
In the project we'll explore changing the pretrained model head with other architecture like attention head, convolutional layer, and different size kernel convolutional layers.
The hypothesis is that these types of model that have been used to solved more complex problem in other domain have the potential to provided better results in this domain.
The progress will be measure on the different model variations by accuracy and rmse error.

### Dataset and Inputs
The dataset used is the [Amazon Bin Image Dataset](https://github.com/awslabs/open-data-docs/tree/main/docs/aft-vbi-pds) (around 1% of it) due to time and resource constraints
The features are images and the target values are the number of object per image (the script to processed target are shown in [Challenge Github Page](https://github.com/awslabs/open-data-docs/tree/main/docs/aft-vbi-pds)[Challenge Github Page](https://github.com/awslabs/open-data-docs/tree/main/docs/aft-vbi-pds)

### Benchmarks
The project will compare the baseline model which uses dense network head to other model variation with different heads, and collect evaluation metric to draw conclusions

### Evaluation Metrics
$$
frac 1 N \sum_{\substack{i=1}}1[p_{i} == g_{i}]
$$



# References
- [Amazon Bin Image Dataset Github](https://github.com/awslabs/open-data-docs/tree/main/docs/aft-vbi-pds)
- [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/)
- [Amazon Inventory Reconciliation using AI](https://github.com/pablo-tech/Image-Inventory-Reconciliation-with-SVM-and-CNN/tree/master)
- [Github Repo Explaning the challenge with Base Model](https://github.com/silverbottlep/abid_challenge/tree/master?tab=readme-ov-file)
