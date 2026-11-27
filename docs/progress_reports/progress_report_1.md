# Progress Report 1 — Places365 Scene Classification

**Team:** Ashley Zapata Minero, Naaz Vemmerath Kulangara, Halas Graden, Christopher Kuizon
**Week of:** ____

## What we got done

This week was mostly about picking our problem and getting everything set up so we can actually train.

- **Problem and dataset (Task 1).** We're doing scene recognition: given a photo, figure out what kind of place it is (kitchen, beach, airport terminal, etc.). We went with **Places365-Standard**, which has 365 classes and about 1.8 million images. It's a lot bigger than CIFAR-100. Scenes are also harder than single objects because the model has to look at the whole picture. The size should make the multi-GPU and profiling tasks later on more interesting too.
- **Data pipeline (Task 2).** The dataset (~26 GB) is in our shared CARC project space, and we checked it against the official checksums. The real test labels aren't public, so we used the official validation set (36,500 images) as our test set. We split off 5% of the training data as our own validation set. For augmentation we went with random crops, flips and some color jitter.
- **Model (Task 3).** We picked **ResNet-50 pretrained on ImageNet** and are fine-tuning the whole thing. It has a published Places365 baseline we can compare against, it trains quickly on CARC's V100s, and it's small enough to scale across multiple GPUs later.
- **Training setup (Task 4, started).** We have a PyTorch training script and a Slurm job for a single GPU. If a job runs out of time it saves a checkpoint, resubmits itself and picks up where it left off. We ran a short test on CARC and it worked.

## Who did what

- **Christopher:** set up the GitHub repo, the CARC environment and the Slurm jobs; wrote the data loading and training code; tracked down the data-loading slowdown described below.
- **Ashley:** wrote up the problem statement and why we picked this dataset; went through the Places paper for the baseline numbers we'll compare to.
- **Naaz:** explored the dataset (class balance, sample images, color stats) and helped decide on the augmentation and the train/val/test split.
- **Halas:** looked at different model options (ResNet, VGG, EfficientNet, training from scratch vs. pretrained) and wrote up why we went with ResNet-50.

## Challenges

- **Data loading was really slow at first.** Reading 1.8 million small image files off CARC's shared filesystem only got us about 400 images/sec, which is slower than the GPU can train. It wasn't the CPU; each file just takes a while to open over the network. We fixed it by copying the dataset into the node's RAM when the job starts. That takes about a minute, and loading got about **5.5× faster** (around 2,200 images/sec).
- **No public test labels**, so we had to build our own validation/test split (above).
- **Learning CARC.** Figuring out the right course account, the storage limits and a working GPU environment took some trial and error.

## Next week

- Do a full training run and try a few learning rates to find the best one (Task 4).
- Get TensorBoard going with sample images, loss/accuracy curves and the model graph (Task 5).
- Evaluate on the test set, make a confusion matrix, and see how we compare to the paper (Task 6).
