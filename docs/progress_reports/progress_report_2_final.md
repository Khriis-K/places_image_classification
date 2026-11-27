# **TAC-450 Progress Report 2: Places365 Scene Classification**

**Team:** Ashley Zapata Minero, Naaz Vemmerath Kulangara, Halas Graden, Christopher Kuizon

# 

# **Accomplished this week**

- **Training and tuning (Task 4).** We tried five learning rates (0.005 to 0.08) and 0.04 did best. The final model is ResNet-50 trained for 10 epochs on one V100, about 5 hours. It gets **56.7% top-1 / 86.7% top-5** on our validation set. Training accuracy ended only \~3.6 points above validation, so it isn't overfitting.  
- **TensorBoard (Task 5).** The following are logged: sample images (with and without augmentation), per-epoch loss and accuracy curves for train and validation, and the model graph. We also exported the charts as images for the final report.  
- **Evaluation (Task 6).** On our held-out test set (36.5 k images never used for training or tuning), the model gets **56.5% top-1 / 86.6% top-5**, within 0.2 points of validation, so our val set was a good guide. With 10 crops per image (how the Places paper reports results) we get **57.2% / 87.1%**, **2.4 points higher than the paper's ResNet-152** with a smaller model. We plotted the full confusion matrix plus a zoomed-in view of the most-confused classes.  
- **Distributed training (Task 7, started).** We started on the multi-GPU version of the training job, with the goal of running across 2 nodes.

# **Contributions**

- **Christopher Kuizon:** ran the learning-rate sweep and final training runs on CARC; made the training job survive storage crashes (below); started the multi-node training job.  
- **Ashley Zapata Minero:** compared our group’s test results against the Places paper  
- **Naaz Vemmerath Kulangara:** set up TensorBoard logging and exported the training curves and sample images for the report.  
- **Halas Graden:** built confusion matrices, checked classes the model gets wrong and why.

# **Challenges**

- **CARC storage kept dropping out.** During the sweep, the shared /scratch1 filesystem lost its connection on three different nodes and crashed four jobs. We changed the training script to retry checkpoint saves and keep TensorBoard logs on node-local storage, copying them over once per epoch. A storage error now just prints a warning instead of killing the run.  
- **Long runs.** One training run is \~5 hr; every learning rate costs queue time. Jobs checkpoint every epoch and resubmit near the time limit, so we don’t lose more than one epoch of work.  
- **Relatively harder classes.** The worst are broad ones like *river* (2%), *mountain* (8%) and *building facade* (2%). They overlap with more specific classes (*mountain* vs. *mountain snowy*, *river* vs. *canal*) and the model almost always picks the specific one. That said, the right answer is usually still in its top 5\. These aren't the classes with the fewest training images, so we don’t think weighing small classes more would fix it.

**Plan for next week**

- Finish distributed training on at least 4 GPUs across 2 nodes and compare training time to one GPU (Task 7).  
- Run the scaling study on 1, 2, 4 and 8 GPUs: training time, throughput, speedup and efficiency (Task 8).  
- Work out how to scale batch size and learning rate as we add GPUs without losing accuracy.