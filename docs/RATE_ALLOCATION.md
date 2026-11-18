# Choosing code length and coefficient precision

STLQ++ spends bits on both codeword indices and their coefficients. Increasing the number of codebooks, **M**, gives the encoder more codeword combinations. Increasing coefficient precision adjusts the weights on the codewords already selected. At a fixed storage budget, these two choices compete.

## Reading the rate–recall curves

![Rate versus recall across all measured code lengths on five datasets, alongside LSQ++, with an additional panel showing recall versus M at fixed 8-bit precision](figures/rate-recall.png)

**Figure 1.** Each colored curve follows one STLQ++ model and forest through a sweep of linkage widths, with root coefficients fixed at 6 bits. The gray dashed line shows LSQ++ across code lengths. Markers show measured configurations. The horizontal axis gives the reported total bits per base vector, including the norm label and IVF assignment. The final panel fixes both STLQ++ coefficient widths at 8 bits and varies M.

At coarse precision, increasing M generally shifts the rising part of the curve to the right. The curves look roughly like horizontal translations on SIFT and DEEP. On GloVe and OpenAI-ArXiv the flat sections appear earlier, and the rising sections also shift upward. Once a curve flattens, finer coefficients buy little additional recall on that forest. A larger M can reach a higher recall if the budget leaves room for its coefficients. As root reconstructions improve, fewer nodes can lower their reconstruction MSE by taking a parent, and the resulting drop in the linked-node ratio can temper the relative gain in peak recall.

**Read vertically at the intended budget.** On DEEP near 145 bits/vector, M8 with 8-bit linkage reaches 46.77% R@1 at 141.41 bits. M10 with 5-bit linkage reaches **52.36% at 143.24 bits**, while M12 with 3-bit linkage reaches 41.50% at 147.14 bits. Here M10 spends enough bits on both parts of the representation.

## Root and linkage precision

![Recall gains from increasing linkage precision at M8 and from increasing root precision across measured code lengths](figures/coefficient-widths.png)

**Figure 2.** The left panel varies linkage width at M8 with 6-bit roots and uses 3-bit linkage as its baseline. The right panel holds linkage at 4 bits. Each faint point shows the gain over 6-bit roots on the same forest at one M. Lines connect the median gains across the available M values for each dataset. The panels use different vertical scales. OpenAI-ArXiv has only a linkage sweep at these settings.

The first extra linkage bit (3 → 4) gains **10.65 percentage points on SIFT** and **1.87 on OpenAI-ArXiv**. The 5 → 6 step gains only 1.29 and 0.07 points, respectively.

**Prioritize root precision when its cost is small.** A root reconstruction contributes to every descendant on its tree, while its coefficients are stored once. At SIFT M8, starting from 8 bits for both types, reducing either type alone to 6 bits costs almost the same recall (0.37 points for root, 0.36 for linkage). Keeping those two extra root bits costs only **0.80 bits/vector**, compared with **15.71** for linkage. On MSong the difference also shows up in recall itself. Across M6–18, the same reduction costs **2.95 points for root** and **0.56 for linkage** on average. Check the actual rate as M changes (influenced by final Linked-node Ratio): forests with many roots spend more bits on root precision.

## Rate allocation as BIGANN grows

![BIGANN rate versus R@1 from 1M to 1B base vectors, shown without calibration on the left and with calibration on the right](figures/bigann-rate-by-size.png)

**Figure 3.** BIGANN R@1 versus bits per vector, before and after calibration. Each color denotes one database size. At 1M, 10M and 100M, STLQ++ varies coefficient precision at M6 and LSQ++ varies code length from M8 to M10.

At all three sizes, the first few extra bits of coefficient precision give the largest recall gains. The curves then flatten. With 8-bit root and 4-bit linkage coefficients, STLQ++ R@1 falls from **40.06% at 1M** to **28.43% at 1B** as the database grows.

At 1B, calibration raises R@1 from **28.43% to 29.03%** for STLQ++ M6 with RSD-2 linkage (96.48 bits/vector) and from **20.57% to 20.87%** for LSQ++ M9 (96.00 bits/vector).

## Choosing settings on a new dataset

1. Start with a representative subset of base vectors and held-out calibration queries. Pick a target in total bits/vector and a nearby M from the supplied configurations. Keep the scoring mode and IVF settings fixed across the width sweep.
2. Build one forest and retain its float coefficients. Hold root width at 6 bits and try linkage widths of 3, 4, 5 and 6 bits. Around the best affordable point, vary root width separately. Each width change reuses the same model and forest through eval-only publication. See [coefficient codec settings](CONFIGURATION.md).
3. If the curve is already flat within budget, try a larger M. If only the steep, coarse end fits, try a smaller M with finer coefficients. Compare the measured points below budget and confirm the selected setting on the full base.

*Measurement conditions.* Figures 1 and 2 show full-probe quantized R@1 with calibration. In those figures, OpenAI-ArXiv uses IVF1024 and the other datasets use IVF256. Figure 3 uses the official BIGANN ground truth for each base size and all 10,000 queries. Each underlying configuration was evaluated in a single run. The SIFT series combines archived and current runs with different seeds or builds across M. Within each width sweep the forest is fixed.
