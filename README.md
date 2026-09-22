# Graph Neural Network (GCN) — Node Classification on Cora

A beginner-friendly, end-to-end project that trains a **Graph Convolutional Network (GCN)** to classify academic papers into topic categories using both their content and citation links. Built with **PyTorch Geometric**, runnable entirely in **Google Colab**.

---

## 🎯 Project Overview

This project answers a simple question: *does knowing who cites whom help you classify a paper's topic, beyond just reading its text?*

We train two models on the same data and compare them:
- A **GCN** that uses both paper features *and* the citation graph
- A plain **MLP** that only uses paper features (no graph)

**Result:** the GCN outperforms the MLP by ~23 percentage points — proof that graph structure carries real, learnable signal.

---

## 📊 Dataset: Cora

| Property | Value |
|---|---|
| Nodes (papers) | 2,708 |
| Edges (citations) | 10,556 |
| Node features | 1,433 (bag-of-words) |
| Classes (topics) | 7 |
| Labeled training nodes | 140 (only ~5% of the graph) |

Cora is a citation network: each node is a scientific paper, each edge is a citation link, and each paper is labeled with one of 7 research topics (e.g., Neural Networks, Reinforcement Learning, Genetic Algorithms).

The dataset is downloaded automatically via `torch_geometric.datasets.Planetoid` — no manual download needed.

---

## 🧠 Model Architecture

A 2-layer Graph Convolutional Network:

```
Input (1433 features)
    ↓
GCNConv(1433 → 128)  →  ReLU  →  Dropout(0.5)
    ↓
GCNConv(128 → 7)
    ↓
LogSoftmax → predicted class probabilities
```

Each `GCNConv` layer aggregates a node's own features with its neighbors' features (normalized by node degree) before applying a learnable linear transformation. Stacking two layers gives every node a **2-hop receptive field** — it "sees" its neighbors' neighbors.

---

## 📁 What's in This Notebook

| Cell | Purpose |
|---|---|
| 1–2 | Install PyTorch Geometric, import libraries, check GPU |
| 3 | Load Cora and inspect its structure |
| 4 | Visualize class distribution |
| 5 | Define the GCN model |
| 6 | Define training/evaluation functions |
| 7 | Train the GCN for 200 epochs |
| 8 | Plot loss and accuracy curves |
| 9 | Visualize learned embeddings with t-SNE |
| 10 | Train an MLP baseline (no graph) for comparison |
| 11 | Inspect individual node predictions |
| 12 | Experiment with GCN depth (demonstrates over-smoothing) |

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Go to **Runtime → Change runtime type → GPU (T4)** for faster training
3. Run **Cell 1** (installs dependencies), then **Runtime → Restart Runtime**
4. Run all remaining cells in order (**Runtime → Run all**)
5. Total runtime: ~3–5 minutes on a T4 GPU

### Requirements
```
torch >= 2.3.0
torch-geometric
torch-scatter
torch-sparse
matplotlib
scikit-learn
numpy
```
All installs are handled in Cell 1 — no local setup required.

---

## 📈 Results

| Model | Test Accuracy |
|---|---|
| MLP (features only, no graph) | ~58% |
| **GCN (features + graph)** | **~81%** |
| Improvement from graph structure | **+23%** |

**Key findings:**
- The GCN reaches ~81% test accuracy using labels from only 140 of 2,708 nodes (semi-supervised learning).
- t-SNE visualization of learned embeddings shows 7 well-separated topic clusters, confirming the model learns meaningful structure.
- Accuracy **peaks at 2 layers and degrades beyond 3–4** — a direct demonstration of the **over-smoothing** problem in deep GNNs.

---

## 🔑 Core Concepts Demonstrated

- **Message passing**: nodes aggregate information from neighbors, then update their own representation
- **Semi-supervised learning on graphs**: training with very few labels by leveraging graph connectivity
- **Symmetric normalization**: preventing high-degree nodes from dominating aggregation
- **Over-smoothing**: why GNNs can't simply be made deeper like CNNs
- **Inductive vs. transductive learning**: this project is transductive — the whole graph is visible during training, only labels are masked

---

## 📚 References

- Kipf & Welling (2017), *Semi-Supervised Classification with Graph Convolutional Networks*
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io)
- [Planetoid / Cora Dataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html)

---

## 📄 License

This project is for educational purposes. The Cora dataset is publicly available for research use.
