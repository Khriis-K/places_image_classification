# Task 3 — Deep Learning Model Selection

## Choice

We use **ResNet-50**, a 50-layer **convolutional neural network (CNN)**, and train it with **transfer learning**: training starts from weights that torchvision has already trained on ImageNet (`ResNet50_Weights.IMAGENET1K_V2`), not from random numbers. We replace the final 1,000-class ImageNet layer with a new 365-class layer and then **fine-tune** the whole network, meaning every layer keeps learning on Places365, not only the new one.

We use this one model in every later task: single-GPU training (Tasks 4–6), the distributed and scaling runs (Tasks 7–8), and profiling (Task 9).

Key terms:
- **Convolutional neural network (CNN):** a network built from small learned filters that slide across the image. Early layers detect edges and textures, later layers detect parts and whole objects. Because the same filter is applied everywhere, a CNN needs far fewer parameters than a fully connected network on 224×224 images.
- **Residual connection:** a shortcut that adds a block's input to its output (`y = x + F(x)`). Gradients can flow through these shortcuts directly, which is what makes networks with 50+ layers trainable. "ResNet" stands for residual network.
- **Transfer learning:** reusing what a model learned on one task (ImageNet objects) as the starting point for another (Places365 scenes).
- **Fine-tuning:** continuing to train those pretrained weights, usually with a smaller learning rate than training from scratch.

## Why a CNN

The assignment suggests CNNs, and they suit this problem well:
- Scenes are recognized from local patterns (textures, surfaces, objects) combined into a global layout. A CNN's stack of layers does exactly this: each layer sees a larger region of the image than the one before.
- CNNs are the architecture the Places paper used for every baseline, so our results compare directly with published numbers.
- They train efficiently on our hardware. Their convolutions map onto the V100's **Tensor Cores** (hardware units for fast matrix math in fp16), which matters for Task 9.

## Why ResNet-50 among CNNs

| Model | Params | Compute per image | Why not (or why) |
|---|---|---|---|
| ResNet-18 | 11.7 M | ~1.8 GFLOPs | Fast, but less accurate. With mixed precision it would probably outrun our data loader (see below), so Tasks 8–9 would mostly measure the CPU, not the GPUs. |
| **ResNet-50** | **25.6 M** | **~4.1 GFLOPs** | **Chosen.** Strong accuracy on Places365, GPU-bound on our setup, affordable. |
| ResNet-152 | 60.2 M | ~11.5 GFLOPs | About 2.5× slower per epoch than ResNet-50, and in the paper it is *not* more accurate than ResNet-50 (table below). |
| VGG-16 | 138 M | ~15.5 GFLOPs | Slowest, and its 138 M parameters mean ~5× more gradient data to exchange between GPUs every step than ResNet-50. |
| Vision Transformer (ViT-B/16) | 86 M | ~17.6 GFLOPs | Needs more data or heavier augmentation to beat CNNs, and is 4× the compute of ResNet-50. |

(**GFLOPs** = billions of arithmetic operations per image in the forward pass, a rough measure of training cost.)

**Published accuracy on Places365.** The official Places365 validation set is our test set (Task 2), so these numbers are directly comparable to what we will report:

| Model (Places paper / official repo) | Top-1 | Top-5 | Note |
|---|---|---|---|
| AlexNet, from scratch | 53.17% | 82.89% | 10-crop, paper Table 2 |
| VGG-16, from scratch | 55.24% | 84.91% | 10-crop, paper Table 2 |
| ResNet-152, fine-tuned from ImageNet | 54.74% | 85.08% | 10-crop, paper Table 2 |
| ResNet-50, from scratch | 55.18% | 85.29% | official repo README (crop setting not stated) |

Two points stand out. First, ResNet-50 is as accurate as models that are 2–5× more expensive. Second, every model lands around 53–55% Top-1, because many Places365 categories overlap (Task 1). A bigger network buys almost nothing here, so a cheaper one that leaves budget for the scaling study is the better trade.

**10-crop** means the published model averaged its predictions over 10 crops of each test image (four corners, center, and their mirror images). This typically adds 1–2 points over a single center crop. Our Task 6 evaluation will report single-crop accuracy and say so, so a result slightly below the table is expected.

## Why transfer learning instead of training from scratch

- **Much less compute.** NVIDIA's reference recipe trains ResNet-50 from scratch on ImageNet for 90 epochs. Starting from ImageNet weights, the network already knows edges, textures and objects, so it reaches good accuracy in far fewer epochs. We plan about **10 epochs** (to be confirmed in Task 4). That is what makes repeating the training run at 1, 2, 4 and 8 GPUs in Task 8 affordable.
- **ImageNet features carry over to scenes, but not completely.** In the Places paper, a classifier on *frozen* ImageNet features reached 40.8% Top-1 on Places205, while a network trained on Places reached 50.0% (paper Table 1). Generic features help, but scenes need their own. That is why we fine-tune every layer rather than only training the new last layer. Fine-tuning all layers also keeps the full 25.6 M-parameter gradient exchange between GPUs, which makes the Task 7–8 communication analysis meaningful.
- **Normalization stays as it is.** Pretrained weights expect inputs normalized with ImageNet's mean and standard deviation, which is what `src/data.py` already uses (Task 2 §4). No change needed.

**We do not use the official Places365-pretrained weights** from the Places authors. They were trained on this exact dataset, so starting from them would skip the learning we are supposed to do. They were also trained on all 1.8 M training images, including the 5% we hold out for validation, so our validation accuracy would be inflated.

**ImageNet V2 weights:** torchvision offers two ImageNet checkpoints for ResNet-50. `IMAGENET1K_V2` (80.9% ImageNet Top-1) was trained with a stronger recipe than `IMAGENET1K_V1` (76.1%), so it is the better starting point. The architecture is identical.

## Compute budget

Reference throughput on one **V100 32 GB**, from NVIDIA's ResNet-50 training benchmark (DeepLearningExamples, DGX-1 32 GB, batch 256 per GPU):

| Precision | Images/s (1 GPU) | One epoch of our 1,713,290 training images |
|---|---|---|
| fp32 | 356 | ~80 min |
| Mixed precision (AMP, fp16) | 1,156 | ~25 min |

**Mixed precision (AMP)** runs most of the math in 16-bit floats on the Tensor Cores and keeps a 32-bit copy of the weights for accuracy. On the V100 it gives about 3.2× the throughput. The V100 has no bf16 support, so we use fp16 with a **gradient scaler** (multiplies the loss to keep small fp16 gradients from rounding to zero).

NVIDIA's numbers use a GPU-accelerated data loader; ours decodes JPEGs on the CPU. **Our loader delivers ~2,190 images/s with 8 CPU cores** (Task 2 §5), so it can feed one ResNet-50 even with AMP, with almost 2× headroom. The GPU stays the bottleneck, which is the situation Tasks 8 and 9 should study. ResNet-18 is roughly 2–3× faster than ResNet-50 (not measured here), which would put it at or above the loader's limit with AMP.

Estimated cost of the plan, at ~10 epochs per training run:

| Use | Runs | GPU-hours (estimate) |
|---|---|---|
| Task 4: hyperparameter checks (short runs) | several, 1–2 epochs each | ~5 |
| Task 4–6: main single-GPU run, AMP | 1 × 10 epochs | ~4.5 |
| Task 7–8: scaling runs on 1/2/4/8 GPUs, AMP | 4 × 10 epochs | ~20–25 (each run costs ~4.5 GPU-h if scaling were perfect; communication adds more) |
| Task 9: profiling, fp32 vs. AMP, loader settings | short runs | ~3 |
| **Total** | | **~35–40 GPU-hours** |

**Measured on CARC** (`src/train_bench.py`, Tesla V100-PCIE-32GB, 8 CPUs, batch 256):

| Configuration | GPU alone (synthetic data) | Full pipeline (real DataLoader) | One epoch |
|---|---|---|---|
| fp32 | 354 | — | ~81 min |
| AMP, default (NCHW) memory layout | 772 | 743 | ~38 min |
| AMP + channels_last | 1,186 | **1,108** | **~26 min** |

fp32 matches NVIDIA's number, but AMP alone reached only two-thirds of it. The missing step is the **memory layout**. Tensor Cores run fp16 convolutions on **channels-last** data (NHWC: the channel values of each pixel stored next to each other). PyTorch's default is NCHW (one whole channel after another), so cuDNN converts the layout around every convolution. Storing the model and inputs as channels-last removes those conversions and gives 1.5× the speed, so `train.py` uses it by default (`--no-channels-last` restores the old behavior for the Task 9 comparison). With it, the GPU is still the bottleneck: the loader alone supplies 1,763 images/s on this node.

The table of estimates above assumed ~25 min per epoch, which the measured 26 min confirms. The wall-clock time of the scaling runs adds up to about 8–10 hours. The plan fits comfortably in the course allocation. In Task 9, we will compare fp32 and AMP on short runs (tens of minutes); a full 10-epoch run in fp32 would take ~13 hours.

## What this means for Tasks 4–9

Starting point for Task 4, to be tuned there:
- `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`, then `model.fc = nn.Linear(2048, 365)`.
- **SGD with momentum** 0.9, weight decay 1e-4, batch 256 per GPU (fits on a 32 GB V100 with AMP).
- Learning rate around 0.01 per 256 images (lower than the ~0.1 used from scratch, so the pretrained weights aren't destroyed early), with a short **warmup** (learning rate ramps up over the first few hundred steps) and **cosine decay** (learning rate falls smoothly to zero by the last epoch).
- **Label smoothing** 0.1: the training targets become 0.9 for the true class instead of 1.0. This discourages overconfidence, which helps when classes overlap as much as these do.

For the scaling study (Task 8), each GPU keeps a batch of 256, so the **global batch** (all GPUs combined) grows from 256 to 2,048. We will scale the learning rate in proportion to the global batch (the **linear scaling rule**, Goyal et al., 2017) and keep the warmup, then report the effect on final accuracy.

**Communication cost (for Task 8):** with **distributed data parallel (DDP)**, every GPU computes gradients on its own batch, and then all GPUs average their gradients before each update (an **all-reduce**). For ResNet-50 that is 25.6 M values (~100 MB in fp32) per step. At ~1,150 images/s, one step of 256 images takes ~0.22 s of compute. Whether the exchange fits inside that time depends on the network between nodes, which is what Task 8 will measure.

Practical notes for the Slurm jobs:
- **Download the pretrained weights once** before the first job (`python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"` from the JupyterLab terminal; it saves to `~/.cache/torch/hub`). Otherwise all DDP processes would try to download the same file at once when training starts.
- **CPUs per GPU:** each GPU needs about 4–5 cores of JPEG decoding to keep up with AMP (1,156 img/s ÷ ~270 img/s per loader worker, from Task 2's 2,190 img/s with 8 workers). Request at least 6 CPUs per GPU.
- **Data per node:** each node unpacks its own copy of the dataset into `$TMPDIR` (~25 GB of RAM), shared by all GPUs on that node, so memory requests are per node, not per GPU.
