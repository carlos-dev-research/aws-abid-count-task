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

In the project we will use Amazon SageMaker notebooks in the AWS environment. The Sagemaker Notebook allow us to easily deploy and train models, with estimator constructs (multi-instance training and hyperparameter finetunning) we can train models with different architectures or hyperparamters and together with profiling tools, we can collect relevant information on model outputs.

### Dataset and Inputs
The dataset used is the [Amazon Bin Image Dataset](https://github.com/awslabs/open-data-docs/tree/main/docs/aft-vbi-pds)  provided by Amazon.
- The features are images which are resized to 3x224x224 (RGB Channels x height x width) to keep consistency.
- The target values are the number of object per image in other words a single positive int number (the script to processed target are shown in [Challenge Github Page](https://github.com/awslabs/open-data-docs/tree/main/docs/aft-vbi-pds)[Challenge Github Page](https://github.com/awslabs/open-data-docs/tree/main/docs/aft-vbi-pds).
- For resource constraints we will only use 1% of the dataset which is equivalent to 5000 samples.


### Benchmarks
The project will compare the baseline model which uses dense network head to other model variation with different heads, and collect evaluation metric to draw conclusions

### Evaluation Metrics
$$
Accuracy: \frac{1}{N}\sum_{i=1}^N 1[p_{i} == g_{i}]
$$
$$
RMSE: \sqrt{\frac{1}{N}\sum_{i=1}^N(p_{i}-g_{i})^2}
$$
Where:
$$p = prediction$$
$$g = ground truth$$

### Project Design
1. Download baseline model and [helper scripts](https://github.com/awslabs/open-data-docs/tree/main/docs/aft-vbi-pds)
2. Build and design pytorch model variants
3. Setup enviroment and create config scripts and notebooks
4. Run different models and collect result for later comparisons
5. Collect and structure results
6. Make conclusions on results,  Did the new architectures got better results? Would it worth it to explore with the complete dataset?

### Notes
- Study is showcased in aws-abid-count-task-Study.ipynb
- Results are shown in results.csv

### Final Conclusion from the Study
#### Model Improvements and Observations
- Additional Epochs: Extending training epochs yielded only marginal improvements. This could be attributed to limitations in the optimization algorithm or the model’s inherent inability to generalize to the data effectively. Notably, results showcased in [Abid Challenge Repo](https://github.com/silverbottlep/abid_challenge) under section 4.3 demonstrate that using a different optimization algorithm led to significantly better performance. Therefore, future work should prioritize experimenting with alternative optimization algorithms and re-evaluating previously tested models.

- Model Selection: Among the explored models, the DNN_HEAD with a regression head emerged as the preferred choice over the baseline DNN_HEAD with a classification head. The regression head enables continuous estimation, which proved advantageous in tasks where we can leverage continuity of the solution, number of objects

#### Performance Relative to Existing Benchmarks
- Limitations of Current Approach: Despite achieving some advancements, our approach did not outperform publicly available models tailored to this task. For instance, the baseline solution trained using the script from [Abid Challenge Repo](https://github.com/silverbottlep/abid_challenge) achieved accuracies up to 40% and rmse as low as 1.3 but focused exclusively on six classes (0 to 5 objects).

#### Recommendations for Future Work
- Optimization Algorithms: Investigate and integrate alternative optimization algorithms to enhance model convergence and performance.
- Model Generalization: Conduct further research into improving the generalization capabilities of the model, possibly through regularization techniques or additional data preprocessing.
- Comparison with Public Models: Perform an in-depth comparison of our methods with state-of-the-art publicly available models to identify gaps and areas for improvement.
- Focus on Continuous Estimation: Continue refining models with regression heads, as these have shown promise in delivering better approximations for quantity estimation tasks.

While the results demonstrate potential, they also highlight the need for further experimentation and optimization to get a more in depth view of the problem at hand.


# References
- [Amazon Bin Image Dataset Github](https://github.com/awslabs/open-data-docs/tree/main/docs/aft-vbi-pds)
- [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/)
- [Amazon Inventory Reconciliation using AI](https://github.com/pablo-tech/Image-Inventory-Reconciliation-with-SVM-and-CNN/tree/master)
- [Github Repo Explaning the challenge with Base Model](https://github.com/silverbottlep/abid_challenge/tree/master?tab=readme-ov-file)
