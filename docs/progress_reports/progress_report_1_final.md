# **TAC-450 Progress Report 1: Places365 Scene Classification**

**Team:** Ashley Zapata Minero, Naaz Vemmerath Kulangara, Halas Graden, Christopher Kuizon

# 

# **Accomplished this week**

- **Problem and dataset (Task 1).** We're doing scene recognition: given a photo, figure out what kind of place it is (kitchen, beach, airport terminal, etc.). We went with **Places365-Standard**; 365 classes and about 1.8 M images. It's a lot bigger than CIFAR-100. Scenes are harder than single objects because the model has to look at the whole picture. The size should make multi-GPU and profiling tasks later on more interesting too.  
- **Data pipeline (Task 2).** The datasett (\~26 GB) is in the shared CARC space, checked against published checksums. We made a stratified, seeded 95/5 train/validation split (1.71 M per 90 k images). The official validation set (36.5 k images, 100 per class) is held out as our test set. We use random crops, flips, and color jitter in training. Validation/test use a fixed center crop, normalized with measured per-channel statistics.  
- **Model choice (Task 3).** We chose **ResNet-50 pretrained on ImageNet** and fine-tuned the whole network. It has a strong published baseline on Places365 to compare against, runs fast on CARC's V100 GPUs, and is small enough for the multi-GPU scaling study.  
- **Training infrastructure (Task 4, started).** We have a PyTorch training script and a Slurm job for a single GPU. If a job runs out of time it saves a checkpoint, resubmits itself and picks up where it left off. We ran a short E2E test on CARC and it worked.

**Contributions**

- **Christopher Kuizon:** set up GitHub repo, CARC environment and the Slurm jobs; wrote the data loading and training code; tracked down the data-loading slowdown described below.  
- **Ashley Zapata Minero:** wrote up the problem statement and why we picked this dataset; went through the Places paper for the baseline numbers we'll compare to.  
- **Naaz Vemmerath Kulangara:** explored the dataset (class balance, sample images, color stats) and helped decide on the augmentation and the train/val/test split.  
- **Halas Graden:** compared candidate architectures (ResNet, VGG, EfficientNet, training from scratch vs. transfer learning) and wrote up the reasoning for ResNet-50.

# 

# **Challenges**

- **Slow data loading.** Reading 1.8 M small image files off CARC's shared filesystem was about 400 images/sec, slower than the GPU can train. It wasn't the CPU; each file just takes a while to open over network. We fixed it by copying the dataset into the node's RAM when the job starts. That takes a minute, and loading is 5.5x faster (around 2,200 images/sec).  
- **No public test labels.** Places365 does not publish its test labels; we cut out our own val set from the train data and kept the official val set as an untouched test set.  
- **Getting started on CARC.** We reviewed lectures on how to set up a GPU environment.

**Plan for next week**

- Run full single-GPU training and a learning-rate sweep to pick hyperparameters (Task 4).  
- Set up TensorBoard visualizations: sample images, loss/accuracy curves, model graph (Task 5).  
- Evaluate on test set with a confusion matrix and compare to the paper's baseline (Task 6).