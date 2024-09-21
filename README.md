# DP-tauCC
Source code for the paper "Differentially Private Associative Co-clustering", submitted to SIAM SDM 2024.

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

The results are in the <code>output/\<algorithm\></code> directories, one file per dataset. Plots can be generated using the notebook files.