# ♟️ Master-of-Science-Degree-Utils

Auxiliary scripts and notebooks for a Master's degree research project focused on **chess board digitalization** — the automatic recognition of chess positions from images using computer vision and deep learning. This repository contains all the supporting utilities: dataset generation, preprocessing, model training, clustering, and inference testing.

---

## 🎯 Research Overview

The core goal of this project is to digitalize a chess board from an image and reconstruct its position in [FEN notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation). The pipeline covers:

1. **Data collection** — fetching real games from chess.com and generating synthetic board images via Blender
2. **Board segmentation** — detecting and cropping individual squares using Hough line transforms
3. **Piece classification** — identifying which piece (if any) is on each square using a MobileNetV2-based CNN
4. **FEN reconstruction** — assembling the 64-square predictions into a full board FEN string
5. **Game clustering** — grouping games by similarity using move-vector distances and Affinity Propagation

---

## 📁 Repository Structure

```
.
├── Models/
│   ├── imageClassifier.pth              # Trained piece classifier weights
│   └── imageClassifierReal.pth          # Piece classifier trained on real images
│
├── chess_img.py                         # Generate random board images via Blender (bpy)
├── chess_img_fen.py                     # Generate board images from FEN positions via Blender
│
├── separar_casas.py                     # Board square segmentation using Hough lines
│
├── read_imgs.py                         # Load and preprocess cropped square images
├── balancear_dados.py                   # Balance pawn classes in the dataset CSV
├── fix_csv.py                           # Strip whitespace from ChessMovesTable.csv
├── filterToNet.py                       # Split dataset into black/white move CSVs
├── npy_csv.py                           # Convert .npy distance matrix to CSV
│
├── model.py                             # MobileNetV2 piece classifier (6 classes)
├── modeloFEN.py                         # MobileNetV2 full-board FEN predictor (64 × 13 outputs)
│
├── testeModel.py                        # Inference on synthetic board images
├── testereal.py                         # Inference on real board photographs
├── teste.py                             # Batch resize real images to 1920×1080
│
├── csv_to_dist.py                       # Compute pairwise game distances as move vectors
├── conc_matrix.py                       # Symmetrize the upper-triangle distance matrix
├── cluster.py                           # Affinity Propagation clustering on game distances
│
├── baixarjogos.py                       # Download all games from a chess.com profile
├── get_fen.py                           # Dataset balancing experiments (over/undersampling)
│
├── digitalizationBoard.ipynb            # Full board digitalization pipeline (notebook)
├── digitalizationBoard_Piece.ipynb      # Piece-only classification experiments
├── digitalizationBoard_PieceColor.ipynb # Piece + color classification experiments
├── digitalizationBoard_PieceColor_Real.ipynb         # Same, trained on real images
├── digitalizationBoard_PieceColor_Real_PreTrain.ipynb # With MobileNetV2 pretraining
├── digitalizationBoardTest.ipynb        # Digitalization testing on synthetic images
├── digitalizationBoardTestReal.ipynb    # Digitalization testing on real photographs
├── predictmove.ipynb                    # Move prediction experiments
└── cluster.ipynb                        # Interactive clustering exploration
```

---

## 🧩 Module Details

### 🖼️ Synthetic Image Generation (`chess_img.py`, `chess_img_fen.py`)
Uses **Blender** (`bpy`) and `python-chess` to render 3D chess board images programmatically. `chess_img.py` generates random positions while `chess_img_fen.py` renders positions loaded from a FEN CSV file, saving both the rendered PNG and the corresponding FEN label.

### 🔲 Board Segmentation (`separar_casas.py`)
Applies Canny edge detection and Hough line transform (`cv2.HoughLines`) to locate the board grid lines, computes line intersections to define the 64 squares, and crops each square into an individual image file, labelling it with the piece symbol from the FEN.

### 🧠 Models (`model.py`, `modeloFEN.py`)
Two MobileNetV2-based architectures fine-tuned for chess:
- **`model.py`** — classifies a cropped square into one of 6 piece types (P, N, B, R, Q, K) regardless of colour, using a single linear head
- **`modeloFEN.py`** — predicts the full board FEN at once using 64 parallel softmax classifiers (one per square, 13 classes each: 12 pieces + empty)

### 🔍 Inference (`testeModel.py`, `testereal.py`)
End-to-end inference scripts: detect the board, segment squares, run the classifier per square, and reconstruct the predicted FEN string. `testeModel.py` targets synthetic images; `testereal.py` handles real photographs.

### 📊 Game Clustering (`csv_to_dist.py`, `conc_matrix.py`, `cluster.py`)
Encodes each chess move as a 9-dimensional vector (source square, destination square, capture square, piece weights) and computes pairwise Euclidean distances between games. A symmetrized distance matrix is then clustered using **Affinity Propagation** (`sklearn`).

### 🌐 Data Collection (`baixarjogos.py`)
Fetches all games from a chess.com player profile via the public REST API and saves them to a JSON file.

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Deep Learning | PyTorch, torchvision (MobileNetV2) |
| Computer Vision | OpenCV (`cv2`), Pillow |
| Chess Logic | `python-chess` |
| 3D Rendering | Blender (`bpy`) |
| Data & Clustering | NumPy, Pandas, scikit-learn (Affinity Propagation) |
| Data Collection | `requests` (chess.com API) |
| Notebooks | Jupyter |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- NVIDIA GPU with CUDA (CPU fallback supported)
- [Blender](https://www.blender.org/) (required only for image generation scripts)

### Install dependencies

```bash
pip install torch torchvision opencv-python pillow python-chess pandas numpy scikit-learn requests jupyter
```

> Blender scripts (`chess_img.py`, `chess_img_fen.py`) must be run from within Blender's Python environment, not from a standard terminal.

### Expected dataset structure

```
Dataset/
├── img_fen.csv                  # image name ↔ FEN mapping
├── img_piece_square.csv         # image name ↔ piece ↔ square mapping
├── JogoPosFenNextClustered.csv  # clustered game positions
└── Pecas/
    ├── imagens_casa_tensor*.pt  # preloaded image tensors
    └── pecas_casa_tensor*.npy   # one-hot piece labels
```

---

## 📄 License

This repository is intended for academic research purposes as part of a Master's degree project.
