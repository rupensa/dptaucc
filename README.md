# DP-tauCC
Source code for the paper "Differentially Private Associative Co-clustering", SIAM SDM 2025, by Elena Battaglia and Ruggero G. Pensa.

[[paper]](SDM25_dptaucc_paper.pdf) [[supplemental material]](SDM25_dptaucc_appendix.pdf)

## How to reproduce the experiments:

Git-clone the repository. Then, install the required packages:

```
pip install -r requirements.txt
```

Run the experiments on synthetic data:

```
python src/DPCoClust_synthetic_NMI.py
```

Run the experiments on real-world datasets:

```
python src/DPCoClust_real_data_NMI.py
```

Run the experiments for getting the runtime curves:

```
python src/DPCoClust_synthetic_time.py
```

Run the experiments to get the iterations/epsilon heatmaps:

```
python src/DPCoClust_real_data_iter.py
```

Run the experiments to get the h/epsilon heatmaps:

```
python src/DPCoClust_real_data_h.py
```

The results are in the <code>output/\<algorithm\></code> directories, one file per dataset. Plots can be generated using the notebook files.

## How to cite
```
@inproceedings{SDM25,
  author       = {Elena Battaglia and Ruggero G. Pensa},
  title        = {Differentially Private Associative Co-Clustering},
  booktitle    = {Proceedings of the 2025 {SIAM} International Conference on Data Mining,
                  {SDM} 2025, Alexandria, VA, USA, May 1-3, 2025},
  pages        = {1--9},
  publisher    = {{SIAM}},
  year         = {2025}
}
```
