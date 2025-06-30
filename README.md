# Markovian Traffic Equilibrium: Reproduction of Numerical Experiments with MSA

This repository provides a numerical reproduction of the experiments in the **Markovian Traffic Equilibrium (MTE)** paper.

### Original Paper

This project reproduces the numerical experiments using the Method of Successive Averages (MSA) from the paper:  
*“Markovian Traffic Equilibrium”* by J.-B. Baillon and R. Cominetti, published in _**Mathematical Programming**_.
DOI: [10.1007/s10107-006-0076-2](https://doi.org/10.1007/s10107-006-0076-2)

### Data Source

This project uses data from the [TransportationNetworks](https://github.com/bstabler/TransportationNetworks) repository, which provides open-access transportation datasets for academic research purposes.  
The data used include _SiouxFalls_net.tntp_, _SiouxFalls_trips.tntp_, and _SiouxFalls_flow.tntp_.

All data are originally donated and intended **only for academic research**.  
Users are responsible for any conclusions or results derived from the data.

**Per the original license**, users must cite the data source in any publication relying on these datasets.  
The Transportation Networks for Research team and contributing institutions are **not responsible for the correctness or content** of the datasets.

⚠️ **Note:** This repository does **not** redistribute the original datasets. Please download them from the official source linked above.

### Code Description

This project includes two Python scripts, implementing:  
- a stochastic model based on the logit choice model;  
- a deterministic model that chooses the minimum cost action at each node.
