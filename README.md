# fil3d

**fil3d** is a Python package for identifying and analyzing coherent three-dimensional (3D) filamentary or cloud-like structures in 3D datasets (e.g., position–position–velocity gas emission cubes).  
It provides an efficient and modular framework to extract 2D masks from individual slices, connect them across a third dimension, and analyze their morphological, kinematic, and statistical properties.

Originally developed for radio astronomical data, **fil3d** can be applied to any volumetric dataset representing diffuse structures — from interstellar gas to simulated scalar fields.

---

🔍 Key Features

- **3D structure detection** — identifies coherent 3D features from 2D slices  
- **Flexible linking algorithm** — connects masks across slices based on overlap  
- **Modular architecture** — `MaskObjNode` and `MaskObjNodeTree` as reusable data structures  
- **CLI tools** — command-line workflow for FITS cubes (`fil3d-find-trees`)  
- **Data-driven** — designed to integrate with ML-based inference and visualization tools  
- **Lightweight dependencies** — built on NumPy, Astropy, and FilFinder  

---

## 📦 Installation

### From source
```bash
pip install fil3d
```

Command-Line Interface

The package includes a command-line tool to detect and link structures directly from 3D FITS cubes.

fil3d-find-trees \
  --fits cube.fits \
  --save-nodes nodes.pkl \
  --save-trees trees.pkl \
  --thr 0.85 \
  --v-start 500 \
  --v-end 1400

 | Flag                   | Description                                       |
| ---------------------- | ------------------------------------------------- |
| `--fits`               | Path to input FITS cube (`nv × ny × nx`)          |
| `--save-nodes`         | Output path to save per-slice node dictionary     |
| `--save-trees`         | Output path to save linked tree structures        |
| `--thr`                | Overlap threshold between slices (default = 0.85) |
| `--v-start`, `--v-end` | Velocity channel range                            |
| `--log-level`          | Logging verbosity (`DEBUG`, `INFO`, etc.)         |

Example Run: 
fil3d-find-trees --fits data/GASKAP.fits --thr 0.8 --save-trees trees.pkl


## 🧪 Example Workflow
1. Preprocess and extract masks
```bash
from fil3d.cli.find_trees import noderun_for_multichannel
from astropy.io import fits
import numpy as np

data, header = fits.getdata("cube.fits", header=True)
vchannels = np.arange(data.shape[0])

nodes = noderun_for_multichannel(data, header, vchannels)
```

2. Link masks across channels
```bash
from fil3d.cli.find_trees import run_and_save_trees

run_and_save_trees(nodes, save_path="trees.pkl", overlap_thresh=0.85)
```
📘 Citation

If you use fil3d in your research, please cite:
Kim, D. A. (2023), The kinematic structure of magnetically aligned H I filaments.
DOI: 10.1093/mnras/stad2792

BibTeX:
@ARTICLE{2023MNRAS.526.4345K,
       author = {{Kim}, Doyeon A. and {Clark}, S.~E. and {Putman}, M.~E. and {Li}, Larry},
        title = "{The kinematic structure of magnetically aligned H I filaments}",
      journal = {\mnras},
     keywords = {ISM: clouds, ISM: kinematics and dynamics, ISM: magnetic fields, ISM: structure, Astrophysics - Astrophysics of Galaxies, Astrophysics - Solar and Stellar Astrophysics},
         year = 2023,
        month = dec,
       volume = {526},
       number = {3},
        pages = {4345-4358},
          doi = {10.1093/mnras/stad2792},
archivePrefix = {arXiv},
       eprint = {2309.10777},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.4345K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

