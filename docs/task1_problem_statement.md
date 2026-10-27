# Task 1 — Problem Statement and Dataset Selection

## Problem Statement

We want to answer the question *"What kind of place is this?"* from a single photograph. **Scene recognition** is a different task from object recognition (CIFAR-100, ImageNet). An object classifier labels the most prominent *thing* in an image, but a scene classifier has to use the whole image: its spatial layout, surfaces, lighting, and the set of objects that appear together. For example, a *kitchen* and a *restaurant kitchen* contain almost the same objects (stoves, counters, cabinets), and an *airport terminal* and a *train station platform* look very similar. Telling them apart takes global context rather than one object.

Our goal is to train a deep convolutional neural network in PyTorch that classifies a 256×256 RGB photo into one of **365 scene categories**, such as indoor (*bedroom, library, operating room*), natural (*forest path, glacier, coast*), and urban (*street, skyscraper, gas station*). We will report Top-1 and Top-5 accuracy. Top-5 matters because scene labels are often ambiguous: one image can reasonably be called a *park*, a *botanical garden*, or a *picnic area*.

Accurate scene recognition is useful in many applications:

- **Robotics and autonomous navigation:** a robot or vehicle needs to know whether it is in a corridor, a parking garage, or on a highway before it can act appropriately.
- **Photo organization and search:** automatically tagging personal and stock photo libraries by location type.
- **Context for other vision tasks:** knowing the scene is a strong prior for object detection ("a boat is more likely at a harbor than in a bedroom").
- **Content moderation and geo-analysis:** recognizing settings in large image collections.

The task also suits this course's focus on high-performance computing. At 1.8 million images (~26 GB), the dataset is large enough that single-GPU training takes a long time and data loading becomes a real bottleneck. That makes it a meaningful workload for the distributed-training, scaling, and profiling experiments in Tasks 7–9.

## Dataset Selection: Places365-Standard

We selected **Places365-Standard**, a subset of the MIT CSAIL **Places** database (Zhou, Lapedriza, Khosla, Oliva & Torralba, *"Places: A 10 Million Image Database for Scene Recognition,"* IEEE TPAMI, 2017). We use the `train_256_places365standard.tar` release, in which every image has been resized to 256×256 pixels.

### Dataset summary (verified against our local copy)

| Property | Value |
|---|---|
| Number of classes | **365** scene categories |
| Training images | **1,803,460** |
| Images per class | **3,068** (min, *dressing_room*) to **5,000** (max) |
| Classes with the full 5,000 images | 339 of 365 |
| Mean images per class | ~4,941 |
| Image format | RGB JPEG, 256×256 |
| Archive size | ~26 GB |
| Official validation set | 100 images per class (36,500 total) — `val_256.tar`\* |
| Official test set | 900 images per class (labels withheld) |

We counted the images in each class folder of the archive. The totals match the numbers published in the paper exactly (1,803,460 training images, 3,068–5,000 per class), which confirms that our copy is complete.

### Meets the project requirements

The assignment requires **at least 100 categories with at least 500 images each**. Places365-Standard has **3.65×** the required number of categories, and even its smallest class (3,068 images) has more than **6×** the per-class minimum. Like CIFAR-100, it is a fixed, labeled, multi-class image classification benchmark, but its images are higher resolution (256×256 vs. 32×32) and it has more than 30× as many training images.

### Label quality

The labels were collected through a multi-stage human annotation process, not by web scraping alone:

1. Candidate images were downloaded from search engines by combining each scene name with common adjectives (e.g., "messy kitchen", "wooden kitchen"), and duplicates were removed.
2. **Amazon Mechanical Turk** workers verified every image in two rounds of yes/no annotation. Each task included control images with known ground-truth labels from the SUN database to filter out unreliable annotators.
3. A third round used a trained classifier to find likely mislabeled or borderline images and sent them back to human annotators.
4. A fourth round targeted pairs of categories that are easy to confuse (e.g., *swimming pool indoor* vs. *outdoor*, *art school* vs. *art studio*): workers chose between the two categories directly.

The dataset has a well-documented annotation process and is widely used as a benchmark. Published baselines (e.g., ResNet-152 at ~54.7% Top-1 / ~85.1% Top-5 on the validation set) give us a reference point for judging our model. The relatively low Top-1 accuracy even for strong models shows that the task is hard and that some categories overlap semantically. We expect this to show up in our confusion matrix (Task 6).

### Sufficiency for deep learning

With ~4,900 labeled examples per class, the dataset is large enough to train a CNN from scratch, not just fine-tune one. It also leaves plenty of data for a held-out split. Because labels for the official test set are not public, our evaluation plan is:

- **Training / validation:** split the 1.8 M training images 95% / 5%, stratified by class: 1,713,290 training and 90,170 validation images (~247 per class). The dataset is large enough that 5% still gives a precise accuracy estimate (±0.3 points) while keeping as much data as possible for training.
- **Test:** use the official `val_256` set (36,500 images, exactly 100 per class) as our final held-out test set, never used for tuning.

\* The paper and the dataset's download page both describe the validation set as 50 images per class. The official `val_256.tar` we downloaded (MD5 verified) and its label file `places365_val.txt` actually contain 100 per class, so we use the full 36,500.

The training set is only mildly imbalanced (the largest class is about 1.6× the smallest, and 93% of classes have exactly 5,000 images). We don't expect imbalance to be a significant issue, but we will report per-class accuracy so that we can check.
