
### 1 synthetic + written (rule v2)

| batch | scene | n | compliance | lock rate | compl. possible | compl. impossible | % impossible |
|---|---|---|---|---|---|---|---|
| 1 | 4 circles/nobg/mono | 10 | 100.0 ±  0.0 | 100.0 ±  0.0 | 100.0 ±  0.0 | – |   0.0 ±  0.0 |
| 2 | 4 circles/bg/mono | 10 |  28.0 ± 25.6 |  36.0 ± 19.6 |  28.0 ± 25.6 | – |   0.0 ±  0.0 |
| 3 | 6 circles/nobg/mono | 10 | 100.0 ±  0.0 | 100.0 ±  0.0 | 100.0 ±  0.0 | – |   0.0 ±  0.0 |
| 4 | 6 circles/bg/mono | 10 |   4.0 ±  8.0 |  32.0 ± 16.0 |   4.0 ±  8.0 | – |   0.0 ±  0.0 |
| 5 | 6 objects/nobg/mono | 10 | 100.0 ±  0.0 | 100.0 ±  0.0 | 100.0 ±  0.0 | – |   0.0 ±  0.0 |
| 6 | 6 objects/bg/mono | 10 |  34.0 ± 26.9 |  46.0 ± 25.4 |  34.0 ± 26.9 | – |   0.0 ±  0.0 |
| 7 | 6 objects/nobg/color | 10 | 100.0 ±  0.0 | 100.0 ±  0.0 | 100.0 ±  0.0 | – |   0.0 ±  0.0 |
| 8 | 6 objects/bg/color | 10 |  48.0 ± 28.6 |  54.0 ± 29.7 |  48.0 ± 28.6 | – |   0.0 ±  0.0 |
| 9 | 9 objects/nobg/color | 10 |  96.0 ± 12.0 |  98.0 ±  6.0 |  96.0 ± 12.0 | – |   0.0 ±  0.0 |
| 10 | 9 objects/bg/color | 10 |  18.0 ± 18.9 |  54.0 ± 18.0 |  18.0 ± 18.9 | – |   0.0 ±  0.0 |
| **no clutter** | | 50 |  99.2 ±  5.6 |  99.6 ±  2.8 |  99.2 ±  5.6 | – |   0.0 ±  0.0 |
| **clutter** | | 50 |  26.4 ± 27.3 |  44.4 ± 24.1 |  26.4 ± 27.3 | – |   0.0 ±  0.0 |
| **all** | | 100 |  62.8 ± 41.4 |  72.0 ± 32.5 |  62.8 ± 41.4 | – |   0.0 ±  0.0 |

### 2 synthetic + GSCD-HW: no results in results_possible5/sim_gscd_hw

### 3 DAVIS + written (rule v2)

| batch | scene | n | compliance | lock rate | compl. possible | compl. impossible | % impossible |
|---|---|---|---|---|---|---|---|
| 1 | 4 circles/nobg/mono | 10 |  98.0 ±  6.0 | 100.0 ±  0.0 |  98.0 ±  6.0 | – |   0.0 ±  0.0 |
| 2 | 4 circles/bg/mono | 10 |  44.0 ± 25.0 |  60.0 ± 34.6 |  44.0 ± 25.0 | – |   0.0 ±  0.0 |
| 3 | 6 circles/nobg/mono | 10 |  96.0 ±  8.0 | 100.0 ±  0.0 |  96.0 ±  8.0 | – |   0.0 ±  0.0 |
| 4 | 6 circles/bg/mono | 10 |  58.0 ± 16.6 |  78.0 ± 10.8 |  58.0 ± 16.6 | – |   0.0 ±  0.0 |
| 5 | 6 objects/nobg/mono | 10 | 100.0 ±  0.0 | 100.0 ±  0.0 | 100.0 ±  0.0 | – |   0.0 ±  0.0 |
| 6 | 6 objects/bg/mono | 10 |  92.0 ± 13.3 |  96.0 ±  8.0 |  92.0 ± 13.3 | – |   0.0 ±  0.0 |
| 7 | 6 objects/nobg/color | 10 | 100.0 ±  0.0 | 100.0 ±  0.0 | 100.0 ±  0.0 | – |   0.0 ±  0.0 |
| 8 | 6 objects/bg/color | 10 |  70.0 ± 18.4 |  82.0 ± 14.0 |  70.0 ± 18.4 | – |   0.0 ±  0.0 |
| 9 | 9 objects/nobg/color | 10 | 100.0 ±  0.0 | 100.0 ±  0.0 | 100.0 ±  0.0 | – |   0.0 ±  0.0 |
| 10 | 9 objects/bg/color | 10 |  82.0 ± 14.0 |  90.0 ± 10.0 |  82.0 ± 14.0 | – |   0.0 ±  0.0 |
| **no clutter** | | 50 |  98.8 ±  4.7 | 100.0 ±  0.0 |  98.8 ±  4.7 | – |   0.0 ±  0.0 |
| **clutter** | | 50 |  69.2 ± 24.7 |  81.2 ± 22.1 |  69.2 ± 24.7 | – |   0.0 ±  0.0 |
| **all** | | 100 |  84.0 ± 23.2 |  90.6 ± 18.2 |  84.0 ± 23.2 | – |   0.0 ±  0.0 |

### 4 DAVIS + GSCD-HW: no results in results_possible5/davis_gscd_hw

### Summary (mean ± std over trials)

| experiment | group | n | compliance | lock rate | compl. possible | compl. impossible | KWS acc |
|---|---|---|---|---|---|---|---|
| 1 synthetic + written | no clutter | 50 |  99.2 ±  5.6 |  99.6 ±  2.8 |  99.2 ±  5.6 | – | – |
| 1 synthetic + written | clutter | 50 |  26.4 ± 27.3 |  44.4 ± 24.1 |  26.4 ± 27.3 | – | – |
| 1 synthetic + written | all | 100 |  62.8 ± 41.4 |  72.0 ± 32.5 |  62.8 ± 41.4 | – | – |
| 3 DAVIS + written | no clutter | 50 |  98.8 ±  4.7 | 100.0 ±  0.0 |  98.8 ±  4.7 | – | – |
| 3 DAVIS + written | clutter | 50 |  69.2 ± 24.7 |  81.2 ± 22.1 |  69.2 ± 24.7 | – | – |
| 3 DAVIS + written | all | 100 |  84.0 ± 23.2 |  90.6 ± 18.2 |  84.0 ± 23.2 | – | – |
