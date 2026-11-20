# Maximum inner product search with frozen L2 codes

The main tables compare **radially calibrated STLQ++ and LSQ++** with **ScaNN AVQ** on maximum inner product search. STLQ++ and LSQ++ reuse their L2-trained codes. At 88 and 152 bits/vector, calibrated STLQ++ leads in **10 of 14** first-neighbor comparisons and **13 of 14** top-10 overlap comparisons.

We evaluate top-1 R@1, R@10, R@100; top-10 R@10; top-50 R@50.

[Before and after calibration comparation](#effect-of-radial-calibration)

[All measurements by dataset](#all-measurements-by-dataset)

## First-neighbor recall (1@1) with calibration

Percentage of queries for which the highest-scoring vector is the exact maximum-inner-product neighbor. Bold marks the highest recall for each dataset and budget. ScaNN uses floating-point query lookup tables.

<table>
<thead>
<tr><th rowspan="2">Dataset</th><th colspan="3">88 bits/vector</th><th colspan="3">152 bits/vector</th></tr>
<tr><th>STLQ++</th><th>LSQ++</th><th>ScaNN</th><th>STLQ++</th><th>LSQ++</th><th>ScaNN</th></tr>
</thead>
<tbody>
<tr><th scope="row">SIFT</th><td align="right"><strong>42.25</strong></td><td align="right">37.50</td><td align="right">10.75</td><td align="right"><strong>58.47</strong></td><td align="right">51.30</td><td align="right">25.58</td></tr>
<tr><th scope="row">DEEP</th><td align="right"><strong>35.43</strong></td><td align="right">32.45</td><td align="right">12.30</td><td align="right"><strong>54.21</strong></td><td align="right">47.21</td><td align="right">26.73</td></tr>
<tr><th scope="row">GIST</th><td align="right"><strong>80.00</strong></td><td align="right">46.60</td><td align="right">0.00</td><td align="right"><strong>72.50</strong></td><td align="right">43.60</td><td align="right">0.10</td></tr>
<tr><th scope="row">GloVe</th><td align="right">28.53</td><td align="right"><strong>28.83</strong></td><td align="right">19.02</td><td align="right"><strong>46.30</strong></td><td align="right">45.06</td><td align="right">36.91</td></tr>
<tr><th scope="row">MSong</th><td align="right">84.10</td><td align="right"><strong>93.10</strong></td><td align="right">18.40</td><td align="right"><strong>99.20</strong></td><td align="right">96.10</td><td align="right">0.00</td></tr>
<tr><th scope="row">Contriever</th><td align="right"><strong>20.43</strong></td><td align="right">20.11</td><td align="right">7.76</td><td align="right"><strong>31.73</strong></td><td align="right">31.24</td><td align="right">16.25</td></tr>
<tr><th scope="row">OpenAI-ArXiv</th><td align="right">28.79</td><td align="right"><strong>29.43</strong></td><td align="right">3.71</td><td align="right">42.27</td><td align="right"><strong>43.54</strong></td><td align="right">12.43</td></tr>
</tbody>
</table>

## Top-10 overlap (10@10) with calibration

Percentage of the exact top ten recovered, averaged over queries. On OpenAI-ArXiv, LSQ++ finds the first neighbor more often at both rates, while STLQ++ recovers slightly more of the top ten. On GIST and MSong, a single vector is the first neighbor for 72.2% and 50.9% of queries, respectively. Top-10 overlap gives a broader view on those datasets.

<table>
<thead>
<tr><th rowspan="2">Dataset</th><th colspan="3">88 bits/vector</th><th colspan="3">152 bits/vector</th></tr>
<tr><th>STLQ++</th><th>LSQ++</th><th>ScaNN</th><th>STLQ++</th><th>LSQ++</th><th>ScaNN</th></tr>
</thead>
<tbody>
<tr><th scope="row">SIFT</th><td align="right"><strong>53.17</strong></td><td align="right">48.08</td><td align="right">16.73</td><td align="right"><strong>68.76</strong></td><td align="right">61.77</td><td align="right">34.05</td></tr>
<tr><th scope="row">DEEP</th><td align="right"><strong>49.82</strong></td><td align="right">46.13</td><td align="right">21.91</td><td align="right"><strong>66.54</strong></td><td align="right">59.97</td><td align="right">40.88</td></tr>
<tr><th scope="row">GIST</th><td align="right">63.53</td><td align="right"><strong>70.19</strong></td><td align="right">16.18</td><td align="right"><strong>82.27</strong></td><td align="right">73.60</td><td align="right">36.87</td></tr>
<tr><th scope="row">GloVe</th><td align="right"><strong>33.82</strong></td><td align="right">31.96</td><td align="right">21.95</td><td align="right"><strong>51.18</strong></td><td align="right">48.91</td><td align="right">41.67</td></tr>
<tr><th scope="row">MSong</th><td align="right"><strong>92.10</strong></td><td align="right">89.27</td><td align="right">34.92</td><td align="right"><strong>95.56</strong></td><td align="right">88.97</td><td align="right">32.28</td></tr>
<tr><th scope="row">Contriever</th><td align="right"><strong>26.43</strong></td><td align="right">23.73</td><td align="right">6.28</td><td align="right"><strong>35.70</strong></td><td align="right">33.71</td><td align="right">13.02</td></tr>
<tr><th scope="row">OpenAI-ArXiv</th><td align="right"><strong>29.34</strong></td><td align="right">28.51</td><td align="right">2.46</td><td align="right"><strong>41.03</strong></td><td align="right">40.81</td><td align="right">7.39</td></tr>
</tbody>
</table>

## Effect of radial calibration

For a query $q$ and reconstruction $h_i$, calibration changes the score from $q^\top h_i$ to $g_{\ell(i)} q^\top h_i$. The vector’s existing 8-bit norm label selects one of 256 gains fitted from base-vector and reconstruction lengths. On SIFT at 152 bits/vector, this raises STLQ++ 1@1 from **26.05% to 58.47%** and LSQ++ from **22.57% to 51.30%**. On GIST and MSong, STLQ++ 1@1 stays the same at both budgets.

**Uncalibrated → calibrated 1@1 (%).** Each pair uses the same codes.

| Dataset | STLQ++ · 88 bit | LSQ++ · 88 bit | STLQ++ · 152 bit | LSQ++ · 152 bit |
|---|---:|---:|---:|---:|
| SIFT | 13.20 → 42.25 | 12.15 → 37.50 | 26.05 → 58.47 | 22.57 → 51.30 |
| DEEP | 17.80 → 35.43 | 16.69 → 32.45 | 35.12 → 54.21 | 28.24 → 47.21 |
| GIST | 80.00 → 80.00 | 46.60 → 46.60 | 72.50 → 72.50 | 43.60 → 43.60 |
| GloVe | 15.80 → 28.53 | 16.90 → 28.83 | 32.93 → 46.30 | 32.85 → 45.06 |
| MSong | 84.10 → 84.10 | 92.30 → 93.10 | 99.20 → 99.20 | 95.80 → 96.10 |
| Contriever | 15.76 → 20.43 | 16.61 → 20.11 | 28.51 → 31.73 | 29.32 → 31.24 |
| OpenAI-ArXiv | 12.77 → 28.79 | 19.50 → 29.43 | 24.70 → 42.27 | 33.44 → 43.54 |

<details>
<summary>Calibrated comparison · 100 and 200 bits/vector on five datasets  (ScaNN's paper reports at 100 and 200)</summary>

These configurations draw on the existing code-length sweeps. STLQ++ uses the configuration with the highest quantized L2 recall among those up to 0.5 bits above budget. LSQ++ uses its closest available rate at or below budget.

**First-neighbor recall (1@1), %**

<table>
<thead>
<tr><th rowspan="2">Dataset</th><th colspan="3">100 bits/vector</th><th colspan="3">200 bits/vector</th></tr>
<tr><th>STLQ++</th><th>LSQ++</th><th>ScaNN</th><th>STLQ++</th><th>LSQ++</th><th>ScaNN</th></tr>
</thead>
<tbody>
<tr><th scope="row">SIFT</th><td align="right"><strong>45.42</strong></td><td align="right">39.54</td><td align="right">13.73</td><td align="right"><strong>65.85</strong></td><td align="right">55.25</td><td align="right">35.22</td></tr>
<tr><th scope="row">DEEP</th><td align="right"><strong>39.00</strong></td><td align="right">33.42</td><td align="right">14.41</td><td align="right"><strong>59.58</strong></td><td align="right">51.13</td><td align="right">41.43</td></tr>
<tr><th scope="row">GloVe</th><td align="right"><strong>31.49</strong></td><td align="right">30.22</td><td align="right">22.63</td><td align="right"><strong>58.57</strong></td><td align="right">52.29</td><td align="right">52.53</td></tr>
<tr><th scope="row">MSong</th><td align="right"><strong>93.30</strong></td><td align="right">91.90</td><td align="right">0.00</td><td align="right">94.30</td><td align="right"><strong>95.20</strong></td><td align="right">0.00</td></tr>
<tr><th scope="row">OpenAI-ArXiv</th><td align="right">29.48</td><td align="right"><strong>32.27</strong></td><td align="right">5.53</td><td align="right">47.57</td><td align="right"><strong>48.05</strong></td><td align="right">19.16</td></tr>
</tbody>
</table>

**Top-10 overlap (10@10), %**

<table>
<thead>
<tr><th rowspan="2">Dataset</th><th colspan="3">100 bits/vector</th><th colspan="3">200 bits/vector</th></tr>
<tr><th>STLQ++</th><th>LSQ++</th><th>ScaNN</th><th>STLQ++</th><th>LSQ++</th><th>ScaNN</th></tr>
</thead>
<tbody>
<tr><th scope="row">SIFT</th><td align="right"><strong>56.66</strong></td><td align="right">50.85</td><td align="right">20.32</td><td align="right"><strong>75.37</strong></td><td align="right">65.42</td><td align="right">45.68</td></tr>
<tr><th scope="row">DEEP</th><td align="right"><strong>53.82</strong></td><td align="right">48.87</td><td align="right">25.50</td><td align="right"><strong>71.57</strong></td><td align="right">64.51</td><td align="right">55.56</td></tr>
<tr><th scope="row">GloVe</th><td align="right"><strong>38.01</strong></td><td align="right">34.74</td><td align="right">25.84</td><td align="right"><strong>63.33</strong></td><td align="right">55.98</td><td align="right">56.67</td></tr>
<tr><th scope="row">MSong</th><td align="right"><strong>94.32</strong></td><td align="right">91.58</td><td align="right">27.63</td><td align="right"><strong>95.34</strong></td><td align="right">89.61</td><td align="right">38.81</td></tr>
<tr><th scope="row">OpenAI-ArXiv</th><td align="right">30.38</td><td align="right"><strong>30.94</strong></td><td align="right">3.54</td><td align="right"><strong>47.98</strong></td><td align="right">45.43</td><td align="right">11.96</td></tr>
</tbody>
</table>

</details>

## Implementation details

The datasets configuration follows README.md and STLQ++'s paper.

Ground truth comes from FP64 inner products on the original vectors in each base-set split. For the 88- and 152-bit comparisons, STLQ++ and LSQ++ each reuse one index per dataset and rate selected by quantized L2 recall. ScaNN 1.4.2 trains AVQ on the same training split with 4-bit subspaces and an anisotropy threshold of 0.2.

Budgets are logical bits per vector. STLQ++ and LSQ++ count the IVF assignment and norm label, and STLQ++ also counts the forest topology and virtual nodes. ScaNN counts its subspace codes. Shared codebooks are excluded throughout.

## All measurements by dataset

Each table compares scoring modes on the same codes. For STLQ++ and LSQ++, **plain** uses the inner product and **cal.** applies radial calibration. For ScaNN, **float** and **INT8** give the query lookup table precision. **1@k** counts queries with the exact first neighbor in the first *k* results. **k@k** measures the fraction of the exact top *k* recovered. All recalls are percentages.

<details>
<summary>SIFT</summary>

**88-bit budget.** STLQ++ M6, RSD3 (88.196 bits/vector), LSQ++ M9 (88), ScaNN (88).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 13.20 | 42.25 | 12.15 | 37.50 | 10.75 | 10.90 |
| 1@10 | 43.79 | 86.56 | 39.01 | 81.95 | 35.40 | 35.23 |
| 1@100 | 81.65 | 99.43 | 76.16 | 99.10 | 71.15 | 71.22 |
| 10@10 | 23.93 | 53.17 | 21.72 | 48.08 | 16.73 | 16.70 |
| 50@50 | 32.89 | 59.40 | 30.27 | 54.56 | 22.72 | 22.69 |

**100-bit budget.** STLQ++ M6 (100.194 bits/vector), LSQ++ M10 (96), ScaNN (100).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 14.96 | 45.42 | 13.76 | 39.54 | 13.73 | 13.63 |
| 1@10 | 48.22 | 89.32 | 43.55 | 84.84 | 41.19 | 41.14 |
| 1@100 | 84.85 | 99.68 | 79.75 | 99.53 | 78.49 | 78.24 |
| 10@10 | 26.56 | 56.66 | 23.65 | 50.85 | 20.32 | 20.25 |
| 50@50 | 35.77 | 62.73 | 32.25 | 56.87 | 27.01 | 26.94 |

**152-bit budget.** STLQ++ M10, RSD2 (152.931 bits/vector), LSQ++ M17 (152), ScaNN (152).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 26.05 | 58.47 | 22.57 | 51.30 | 25.58 | 25.50 |
| 1@10 | 69.60 | 96.93 | 60.10 | 94.27 | 63.65 | 63.57 |
| 1@100 | 95.72 | 99.98 | 91.33 | 99.96 | 92.27 | 92.05 |
| 10@10 | 40.09 | 68.76 | 34.10 | 61.77 | 34.05 | 33.92 |
| 50@50 | 49.05 | 73.62 | 42.71 | 66.81 | 40.89 | 40.73 |

**200-bit budget.** STLQ++ M14 (200.204 bits/vector), LSQ++ M22 (192), ScaNN (200).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 35.55 | 65.85 | 26.29 | 55.25 | 35.22 | 34.94 |
| 1@10 | 81.83 | 98.97 | 66.34 | 95.98 | 78.38 | 77.97 |
| 1@100 | 98.86 | 100.00 | 94.33 | 99.99 | 97.34 | 97.24 |
| 10@10 | 50.06 | 75.37 | 38.15 | 65.42 | 45.68 | 45.48 |
| 50@50 | 58.44 | 79.67 | 46.80 | 70.38 | 52.31 | 52.06 |

</details>

<details>
<summary>DEEP</summary>

**88-bit budget.** STLQ++ M6, RSD2 (87.580 bits/vector), LSQ++ M9 (88), ScaNN (88).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 17.80 | 35.43 | 16.69 | 32.45 | 12.30 | 12.23 |
| 1@10 | 57.86 | 83.21 | 53.00 | 79.66 | 42.01 | 42.08 |
| 1@100 | 92.52 | 99.18 | 88.85 | 98.86 | 80.09 | 80.14 |
| 10@10 | 32.80 | 49.82 | 30.38 | 46.13 | 21.91 | 21.89 |
| 50@50 | 43.77 | 57.76 | 40.67 | 54.18 | 29.51 | 29.50 |

**100-bit budget.** STLQ++ M7 (100.457 bits/vector), LSQ++ M10 (96), ScaNN (100).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 20.03 | 39.00 | 18.00 | 33.42 | 14.41 | 14.42 |
| 1@10 | 63.95 | 86.74 | 56.27 | 82.59 | 47.85 | 48.00 |
| 1@100 | 95.17 | 99.55 | 91.46 | 99.34 | 85.92 | 85.86 |
| 10@10 | 36.76 | 53.82 | 32.74 | 48.87 | 25.50 | 25.52 |
| 50@50 | 47.75 | 61.24 | 43.11 | 56.53 | 33.58 | 33.55 |

**152-bit budget.** STLQ++ M10, RSD3 (152.761 bits/vector), LSQ++ M17 (152), ScaNN (152).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 35.12 | 54.21 | 28.24 | 47.21 | 26.73 | 26.68 |
| 1@10 | 84.15 | 95.89 | 74.00 | 92.48 | 71.90 | 71.93 |
| 1@100 | 99.08 | 99.99 | 96.78 | 99.87 | 97.36 | 97.29 |
| 10@10 | 51.77 | 66.54 | 44.41 | 59.97 | 40.88 | 40.89 |
| 50@50 | 61.39 | 72.53 | 54.27 | 66.61 | 49.26 | 49.24 |

**200-bit budget.** STLQ++ M12 (192.734 bits/vector), LSQ++ M22 (192), ScaNN (200).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 42.02 | 59.58 | 33.13 | 51.13 | 41.43 | 41.24 |
| 1@10 | 91.14 | 97.96 | 80.63 | 95.29 | 89.00 | 88.85 |
| 1@100 | 99.69 | 99.99 | 97.81 | 99.91 | 99.57 | 99.58 |
| 10@10 | 59.04 | 71.57 | 49.65 | 64.51 | 55.56 | 55.57 |
| 50@50 | 67.74 | 76.83 | 59.06 | 70.55 | 62.93 | 62.90 |

</details>

<details>
<summary>GIST</summary>

**88-bit budget.** STLQ++ M5, RSD3 (87.746 bits/vector), LSQ++ M9 (88), ScaNN (88).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 80.00 | 80.00 | 46.60 | 46.60 | 0.00 | 0.00 |
| 1@10 | 92.90 | 92.90 | 97.40 | 97.40 | 12.80 | 12.70 |
| 1@100 | 99.20 | 99.20 | 100.00 | 100.00 | 93.70 | 93.70 |
| 10@10 | 63.57 | 63.53 | 70.21 | 70.19 | 16.18 | 16.04 |
| 50@50 | 81.17 | 81.15 | 80.47 | 80.42 | 53.45 | 53.45 |

**152-bit budget.** STLQ++ M10, RSD2 (151.903 bits/vector), LSQ++ M17 (152), ScaNN (152).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 72.50 | 72.50 | 43.60 | 43.60 | 0.10 | 0.10 |
| 1@10 | 99.70 | 99.70 | 98.20 | 98.40 | 15.90 | 15.90 |
| 1@100 | 100.00 | 100.00 | 100.00 | 100.00 | 95.70 | 95.70 |
| 10@10 | 82.28 | 82.27 | 73.61 | 73.60 | 36.87 | 36.88 |
| 50@50 | 88.29 | 88.34 | 86.23 | 86.31 | 59.98 | 60.03 |

</details>

<details>
<summary>GloVe</summary>

**88-bit budget.** STLQ++ M6, RSD3 (88.050 bits/vector), LSQ++ M9 (88), ScaNN (88).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 15.80 | 28.53 | 16.90 | 28.83 | 19.02 | 19.01 |
| 1@10 | 45.85 | 64.27 | 46.07 | 63.48 | 47.49 | 47.31 |
| 1@100 | 77.23 | 88.49 | 77.10 | 88.16 | 75.43 | 75.35 |
| 10@10 | 23.85 | 33.82 | 23.48 | 31.96 | 21.95 | 21.95 |
| 50@50 | 29.50 | 37.65 | 28.74 | 35.04 | 24.41 | 24.40 |

**100-bit budget.** STLQ++ M7 (100.063 bits/vector), LSQ++ M10 (96), ScaNN (100).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 18.65 | 31.49 | 18.84 | 30.22 | 22.63 | 22.64 |
| 1@10 | 52.18 | 68.70 | 49.95 | 66.63 | 54.28 | 54.38 |
| 1@100 | 82.61 | 91.81 | 80.70 | 90.79 | 81.43 | 81.39 |
| 10@10 | 27.72 | 38.01 | 26.21 | 34.74 | 25.84 | 25.84 |
| 50@50 | 33.21 | 41.47 | 31.38 | 37.78 | 28.63 | 28.62 |

**152-bit budget.** STLQ++ M12, RSD3 (151.739 bits/vector), LSQ++ M17 (152), ScaNN (152).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 32.93 | 46.30 | 32.85 | 45.06 | 36.91 | 36.78 |
| 1@10 | 73.85 | 85.75 | 72.73 | 84.65 | 76.04 | 76.12 |
| 1@100 | 95.75 | 98.42 | 95.06 | 98.28 | 96.04 | 95.98 |
| 10@10 | 41.22 | 51.18 | 40.06 | 48.91 | 41.67 | 41.67 |
| 50@50 | 46.45 | 54.04 | 44.71 | 51.44 | 44.41 | 44.43 |

**200-bit budget.** STLQ++ M15 (199.006 bits/vector), LSQ++ M23 (200), ScaNN (200).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 47.97 | 58.57 | 41.09 | 52.29 | 52.53 | 52.32 |
| 1@10 | 89.09 | 94.44 | 81.78 | 89.81 | 91.38 | 91.25 |
| 1@100 | 99.42 | 99.74 | 98.26 | 99.46 | 99.66 | 99.66 |
| 10@10 | 55.61 | 63.33 | 48.02 | 55.98 | 56.67 | 56.62 |
| 50@50 | 59.66 | 65.76 | 52.37 | 58.71 | 59.35 | 59.30 |

</details>

<details>
<summary>MSong</summary>

**88-bit budget.** STLQ++ M5, RSD2 (87.547 bits/vector), LSQ++ M9 (88), ScaNN (88).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 84.10 | 84.10 | 92.30 | 93.10 | 18.40 | 9.70 |
| 1@10 | 99.90 | 99.90 | 99.30 | 99.30 | 45.20 | 45.20 |
| 1@100 | 100.00 | 100.00 | 100.00 | 100.00 | 46.20 | 46.20 |
| 10@10 | 92.10 | 92.10 | 89.24 | 89.27 | 34.92 | 35.32 |
| 50@50 | 95.38 | 95.38 | 93.34 | 93.39 | 41.52 | 41.13 |

**100-bit budget.** STLQ++ M6 (100.490 bits/vector), LSQ++ M10 (96), ScaNN (100).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 93.30 | 93.30 | 91.10 | 91.90 | 0.00 | 0.00 |
| 1@10 | 100.00 | 100.00 | 98.00 | 98.00 | 37.00 | 37.20 |
| 1@100 | 100.00 | 100.00 | 100.00 | 100.00 | 44.10 | 44.50 |
| 10@10 | 94.33 | 94.32 | 91.44 | 91.58 | 27.63 | 28.17 |
| 50@50 | 97.42 | 97.38 | 93.70 | 93.59 | 35.26 | 35.27 |

**152-bit budget.** STLQ++ M10, RSD2 (152.198 bits/vector), LSQ++ M17 (152), ScaNN (152).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 99.20 | 99.20 | 95.80 | 96.10 | 0.00 | 0.00 |
| 1@10 | 100.00 | 100.00 | 98.70 | 98.70 | 35.50 | 35.20 |
| 1@100 | 100.00 | 100.00 | 100.00 | 100.00 | 96.50 | 96.50 |
| 10@10 | 95.56 | 95.56 | 89.07 | 88.97 | 32.28 | 33.42 |
| 50@50 | 96.95 | 96.94 | 93.63 | 93.64 | 59.10 | 58.40 |

**200-bit budget.** STLQ++ M13 (198.708 bits/vector), LSQ++ M23 (200), ScaNN (200).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 94.30 | 94.30 | 95.10 | 95.20 | 0.00 | 0.00 |
| 1@10 | 100.00 | 100.00 | 98.70 | 98.80 | 29.60 | 29.90 |
| 1@100 | 100.00 | 100.00 | 100.00 | 100.00 | 66.70 | 66.20 |
| 10@10 | 95.33 | 95.34 | 89.65 | 89.61 | 38.81 | 39.00 |
| 50@50 | 96.93 | 96.89 | 93.95 | 93.94 | 46.20 | 46.13 |

</details>

<details>
<summary>Contriever</summary>

**88-bit budget.** STLQ++ M6, RSD3 (87.800 bits/vector), LSQ++ M9 (88), ScaNN (88).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 15.76 | 20.43 | 16.61 | 20.11 | 7.76 | 7.77 |
| 1@10 | 44.86 | 51.54 | 43.93 | 48.87 | 19.30 | 19.12 |
| 1@100 | 75.90 | 80.47 | 75.85 | 78.57 | 37.87 | 37.82 |
| 10@10 | 22.88 | 26.43 | 21.45 | 23.73 | 6.28 | 6.25 |
| 50@50 | 28.41 | 31.33 | 26.19 | 27.84 | 6.70 | 6.68 |

**152-bit budget.** STLQ++ M12, RSD3 (151.927 bits/vector), LSQ++ M17 (152), ScaNN (152).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 28.51 | 31.73 | 29.32 | 31.24 | 16.25 | 16.36 |
| 1@10 | 63.83 | 67.73 | 64.95 | 66.70 | 36.04 | 35.94 |
| 1@100 | 89.95 | 91.34 | 89.92 | 90.68 | 58.19 | 58.14 |
| 10@10 | 32.96 | 35.70 | 32.52 | 33.71 | 13.02 | 12.96 |
| 50@50 | 37.46 | 39.52 | 36.00 | 36.87 | 12.48 | 12.46 |

</details>

<details>
<summary>OpenAI-ArXiv</summary>

**88-bit budget.** STLQ++ M6, RSD3 (88.526 bits/vector), LSQ++ M9 (88), ScaNN (88).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 12.77 | 28.79 | 19.50 | 29.43 | 3.71 | 0.14 |
| 1@10 | 33.01 | 58.89 | 45.84 | 59.35 | 9.35 | 0.62 |
| 1@100 | 59.95 | 84.89 | 75.41 | 84.86 | 20.87 | 3.09 |
| 10@10 | 15.04 | 29.34 | 21.80 | 28.51 | 2.46 | 0.24 |
| 50@50 | 19.01 | 32.68 | 26.66 | 30.91 | 3.18 | 0.57 |

**100-bit budget.** STLQ++ M6 (97.216 bits/vector), LSQ++ M10 (98), ScaNN (100).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 14.72 | 29.48 | 23.23 | 32.27 | 5.53 | 0.18 |
| 1@10 | 36.32 | 60.05 | 51.91 | 63.41 | 13.25 | 0.95 |
| 1@100 | 64.83 | 85.42 | 80.06 | 87.61 | 27.34 | 4.21 |
| 10@10 | 17.01 | 30.38 | 24.79 | 30.94 | 3.54 | 0.33 |
| 50@50 | 21.07 | 33.70 | 29.47 | 33.11 | 4.27 | 0.69 |

**152-bit budget.** STLQ++ M12, RSD3 (152.230 bits/vector), LSQ++ M17 (152), ScaNN (152).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 24.70 | 42.27 | 33.44 | 43.54 | 12.43 | 0.48 |
| 1@10 | 53.80 | 76.49 | 66.47 | 77.26 | 24.93 | 2.09 |
| 1@100 | 81.17 | 95.12 | 90.37 | 95.39 | 43.66 | 7.36 |
| 10@10 | 25.86 | 41.03 | 33.44 | 40.81 | 7.39 | 0.66 |
| 50@50 | 30.03 | 43.61 | 38.06 | 42.87 | 8.07 | 1.12 |

**200-bit budget.** STLQ++ M14 (193.232 bits/vector), LSQ++ M22 (194), ScaNN (200).

| Recall (%) | STLQ++ plain | STLQ++ cal. | LSQ++ plain | LSQ++ cal. | ScaNN float | ScaNN INT8 |
|---|---:|---:|---:|---:|---:|---:|
| 1@1 | 31.44 | 47.57 | 38.87 | 48.05 | 19.16 | 1.02 |
| 1@10 | 63.22 | 83.34 | 73.02 | 82.59 | 36.10 | 4.00 |
| 1@100 | 88.20 | 97.59 | 93.79 | 97.42 | 57.94 | 11.98 |
| 10@10 | 31.51 | 47.98 | 38.59 | 45.43 | 11.96 | 1.15 |
| 50@50 | 35.69 | 50.51 | 42.80 | 47.03 | 12.71 | 1.74 |

</details>
