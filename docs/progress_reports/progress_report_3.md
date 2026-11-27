# **TAC-450 Progress Report 3: Places365 Scene Classification**

**Team:** Ashley Zapata Minero, Naaz Vemmerath Kulangara, Halas Graden, Christopher Kuizon

# 

# **Accomplished this week**

- **Distributed training (Task 7).** We trained the same ResNet-50 with PyTorch DDP on 4 V100s across 2 nodes. 10 epochs took **72 min vs. \~4 hr 25 min** on one GPU, a **3.7x speedup (93% efficiency)**, with the same accuracy (56.7% top-1).  
- **Scaling study (Task 8).** We ran the same training on 1, 2, 4 and 8 GPUs. 8 GPUs finished in **37 min, a 7.2x speedup (90% efficiency)**. We kept 256 images per GPU and scaled the learning rate with the total batch (0.04 on 1 GPU up to 0.32 on 8). All four runs ended within 0.11 points of each other. Scaling isn't perfectly linear because the GPUs exchange \~97 MB of gradients every step, much of it over the network between nodes.  
- **Profiling (Task 9, started).** We're running the PyTorch Profiler and nvidia-smi on the training job now; results expected tomorrow.

# **Contributions**

- **Christopher Kuizon:** built the multi-node DDP job; ran the 2, 4 and 8 GPU runs; made the scaling plots.  
- **Ashley Zapata Minero:** worked out how to scale batch size and learning rate with GPU count; compared accuracy across runs.  
- **Naaz Vemmerath Kulangara:** analyzed communication overhead to explain why scaling isn't linear.  
- **Halas Graden:** running the Task 9 profiling; proposed spatial pyramid pooling (below).

# **Challenges**

- **Fair 1-GPU baseline.** Our original 1-GPU run was interrupted twice, which added \~40 min. We rebuilt the uninterrupted time from per-epoch logs and added exact timing to the multi-GPU runs.  
- **Large batches lag mid-training.** Accuracy dips around epochs 3–5 for 4 and 8 GPUs before catching up as the learning rate decays. Fine at 10 epochs, but we don't think it'd hold for a shorter schedule.  
- **One run per config.** Each run costs a lot of GPU time, so we can't measure run-to-run variation.

**Plan for next week**

- Finish Task 9: apply at least two optimizations and report throughput and accuracy before and after.  
- Try **spatial pyramid pooling** ([He et al., 2014](https://arxiv.org/abs/1406.4729)): pool the final feature map over a 2x2 grid as well as the whole image, so the classifier gets some spatial context. Two variants: 1x1 + 2x2, and 1x1 + a 2x1 top/bottom split (horizontal flips swap the left/right quadrants, but not top/bottom). Same baseline settings (LR 0.04, same epochs and seed).  
- Try flip averaging at test time as a cheap add-on.  
- Start the conclusion (Task 10), final report and presentation.
