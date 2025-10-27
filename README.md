# Contrastive Knowledge Distillation with CLIP

This project was developed for an M.S. Machine Learning course to demonstrate knowledge distillation from a large, powerful teacher model (OpenAI's CLIP) into a smaller, multilingual student model (a variant of LaBSE). The knowledge transfer is achieved using a contrastive learning objective (CLIP loss) on a paired English-Persian dataset, aligning the student's Persian embeddings with the teacher's English embeddings in a shared latent space.

## Features

* **Knowledge Distillation:** Implements distillation from a frozen `EVA02-E-14-plus` teacher to a fine-tuned `smaller-LaBSE` student.
* **Contrastive Loss:** Utilizes a symmetric contrastive loss to align multilingual embeddings.
* **Alternative Loss:** Includes an optional, simplified (asymmetric) contrastive loss for comparison.
* **Modular & Scalable:** Code is refactored from a notebook into a clean, modular Python package with argument parsing.
* **Logging:** Logs all training progress, metrics, and configurations to both the console and a file (`logs/distillation.log`).
* **Text Normalization:** Includes preprocessing pipelines for both English and Persian text.

## Core Concepts & Techniques

* **Knowledge Distillation:** Training a smaller "student" model to mimic the behavior (in this case, the embedding space) of a larger "teacher" model.
* **Contrastive Learning (CLIP):** A self-supervised learning technique that learns by contrasting positive pairs (which should be similar) against negative pairs (which should be dissimilar).
* **Multilingual Embeddings:** Using models (like LaBSE) that can map text from different languages into a shared embedding space.
* **Symmetric Cross-Entropy Loss:** The core loss function of CLIP, which computes the loss in both directions (e.g., text-to-image and image-to-text) and averages them.
* **PyTorch & Hugging Face:** Utilizes PyTorch for model building/training and the Hugging Face `transformers` library for the student model.

---

## How It Works

The goal of this project is to train a small multilingual model (Student) to produce embeddings for Persian text that are "understandable" by a large, frozen English model (Teacher). We force the Student's Persian embeddings to align with the Teacher's English embeddings for the same concept.

### 1. Core Architecture

* **Teacher Model (`src/teacher.py`):** We use the text encoder from `open_clip`'s **`EVA02-E-14-plus`** model. This model is completely frozen (`requires_grad=False`). It takes English sentences and produces 1024-dimensional embeddings.
* **Student Model (`src/model.py`):** We use **`setu4993/smaller-LaBSE`** as the student. This model produces 768-dimensional embeddings. To make it compatible with the teacher, we add a `LinearProjection` head. This head is a small neural network (Linear -> Swish -> BatchNorm -> Linear -> Dropout -> Residual) that projects the 768-dim student embedding to the 1024-dim teacher space. **Only this projection head and the student model are trained.**
* **Data Pipeline (`src/data_loader.py`):** The `train.csv` and `val.csv` files, containing parallel English-Persian sentences, are loaded. A `Normalizer` class cleans the text for both languages (e.g., removing punctuation, standardizing characters) before tokenization.

### 2. Contrastive Loss Functions (`src/loss.py`)

This is the core of the project. We implement two versions of the CLIP loss. In a batch of size $N$, we have $N$ English sentences and $N$ corresponding Persian sentences.

1.  **Embedding Generation:**
    * Teacher produces `reference_embeds` (N, 1024) from English text.
    * Student produces `candidate_embeds` (N, 1024) from Persian text.
2.  **Normalization:** Both embedding matrices are L2-normalized.
3.  **Similarity Matrix:** We compute the cosine similarity between every reference embedding and every candidate embedding, scaled by a learnable temperature $\tau$.

    $$logits = \frac{(ref\_embeds \cdot cand\_embeds^T)}{\tau}$$
    
5.  **Targets:** The correct pairings are on the diagonal. We create a target vector `targets = [0, 1, 2, ..., N-1]`.

#### A. Symmetric CLIP Loss (Default)

This is the standard CLIP loss, which is symmetric.

* **Loss 1 (Ref -> Cand):** We compute the cross-entropy loss across the rows of the `logits` matrix. This pushes the correct *candidate* (Persian) embedding to be the most similar for each *reference* (English) embedding.

  $$Loss_1 = CrossEntropy(logits, targets)$$
  
* **Loss 2 (Cand -> Ref):** We compute the cross-entropy loss across the columns (by transposing the `logits` matrix). This pushes the correct *reference* (English) embedding to be the most similar for each *candidate* (Persian) embedding.

  $$Loss_2 = CrossEntropy(logits.T, targets)$$
  
* **Final Loss:** The final loss is the average of the two.

  $$Loss_{total} = \frac{(Loss_1 + Loss_2)}{2}$$

#### B. Alternative CLIP Loss (Optional)

This is a simplified, asymmetric version discussed in the notebook's analysis. It only computes the loss in one direction (Reference -> Candidate).

* **Final Loss:**

  $$Loss_{total} = CrossEntropy(logits, targets)$$
  
This is computationally simpler but may result in a weaker alignment as it doesn't enforce the symmetric relationship.

### 3. Training and Analysis

The model was trained for 5 epochs using the **Symmetric CLIP Loss**. The validation accuracy measures the model's ability to correctly match the Persian embedding to its corresponding English embedding within a batch.

**Results:**
| Epoch | Avg-train-loss | Avg-train-accuracy | Avg-val-loss | Avg-val-accuracy | Temperature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0378 | 45.37% | 0.0379 | 69.75% | 19.95 |
| 2 | 0.0378 | 73.92% | 0.0379 | 79.92% | 19.90 |
| 3 | 0.0378 | 82.71% | 0.0379 | 85.24% | 19.84 |
| 4 | 0.0378 | 87.49% | 0.0379 | 87.05% | 19.79 |
| 5 | 0.0378 | 90.78% | 0.0379 | **87.46%** | 19.74 |

**Conclusion:**
The training was highly successful. The validation accuracy quickly climbed from ~70% to **87.46%**, demonstrating that the student model's projection head effectively learned to map its Persian embeddings into the teacher's latent space. The learnable temperature $\tau$, which controls the sharpness of the probability distribution, slowly decreased, helping the model fine-tune its alignments. These results confirm that contrastive distillation is a powerful technique for transferring knowledge between models of different sizes and modalities (or in this case, languages).

---

## Project Structure

```
pytorch-contrastive-knowledge-distillation/
├── .gitignore               # Ignores pycache, venv, logs, and data files
├── LICENSE                  # MIT License
├── README.md                # This file
├── data/
│   ├── train.csv            # Training data
│   └── val.csv              # Validation data
├── logs/
│   └── .gitkeep             # Placeholder for log files (e.g., distillation.log)
├── main.py                  # Main script to run training and evaluation
├── requirements.txt         # Project dependencies
├── run_distillation.ipynb   # Jupyter notebook to run the main script
└── src/
    ├── __init__.py          # Makes 'src' a Python package
    ├── config.py            # Contains the main configuration dictionary
    ├── data_loader.py       # Handles loading and normalizing data
    ├── engine.py            # Contains the train and evaluate loops
    ├── loss.py              # Defines the contrastive loss functions
    ├── model.py             # Defines the student model and projection head
    ├── teacher.py           # Defines the teacher model loader
    └── utils.py             # Utility functions (logging, device, etc.)
````

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/pytorch-contrastive-knowledge-distillation.git
    cd pytorch-contrastive-knowledge-distillation
    ```

2.  **Setup the Environment and Data:**
    ```bash
    # Install dependencies
    pip install -r requirements.txt

    # Create the data directory
    mkdir data
    ```
    **You must manually download the data files (`train.csv` and `val.csv`) and add them to the `data/` directory.**

3.  **Run the Training:**
    You can run the project either from the command line or the provided notebook.

    **Option A: Command Line**

    ```bash
    # Run training with default (symmetric) loss for 5 epochs
    python main.py --epochs 5 --batch_size 128
    
    # Run training with the alternative loss function
    python main.py --epochs 5 --loss_type alternative
    
    # See all options
    python main.py --help
    ```

    **Option B: Jupyter Notebook**

    Open and run the cells in `run_distillation.ipynb`.

4.  **Check Results:**
    Training progress, metrics, and configurations will be saved to `logs/distillation.log`. The best-performing student model weights will be saved to `best_student_model.pth`.

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
