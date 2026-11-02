# Model comparison (validation set, mean over Id/Ig/Is/Ib)

| case     | family          |     r2_log |   median_err_pct |   p95_err_pct |   worst_p95_pct |   train_median_err_pct |
|:---------|:----------------|-----------:|-----------------:|--------------:|----------------:|-----------------------:|
| NMOS_OFF | mlp             |  0.9985    |            2.861 |        13.77  |          23.42  |                 2.289  |
| NMOS_OFF | poly_elasticnet |  0.971     |           35.54  |       185.6   |         380.1   |                33.04   |
| NMOS_OFF | random_forest   |  0.9914    |           10.29  |        57.28  |          92.28  |                 3.991  |
| NMOS_OFF | xgboost         |  0.9951    |            5.243 |        39.49  |          72.47  |                 1.485  |
| NMOS_ON  | mlp             |  0.9999    |            1.088 |         6.334 |           7.821 |                 0.3431 |
| NMOS_ON  | poly_elasticnet |  0.9971    |           13.8   |        44.25  |          57.09  |                13.6    |
| NMOS_ON  | random_forest   |  0.9926    |           12.98  |        63.16  |          76.93  |                 4.631  |
| NMOS_ON  | xgboost         |  0.9987    |            6.269 |        29.9   |          36.51  |                 2.077  |
| PMOS_OFF | mlp             |  0.9966    |            2.968 |        15.04  |          29.03  |                 2.348  |
| PMOS_OFF | poly_elasticnet |  0.9607    |           31.85  |       116.6   |         215.6   |                29.54   |
| PMOS_OFF | random_forest   |  0.9857    |            8.28  |        53.18  |          83.22  |                 3.246  |
| PMOS_OFF | xgboost         |  0.992     |            4.217 |        41.25  |          73.58  |                 1.487  |
| PMOS_ON  | mlp             |  0.9997    |            2.63  |        13.88  |          19.96  |                 1.219  |
| PMOS_ON  | poly_elasticnet |  0.9965    |           13.79  |        50.8   |          68.43  |                13.31   |
| PMOS_ON  | random_forest   |  0.9938    |           12.46  |        65.93  |          82.41  |                 4.925  |
| PMOS_ON  | xgboost         |  0.9987    |            5.963 |        28.97  |          38.09  |                 2.026  |
| NMOS_ON  | baseline_mean   | -0.0007977 |           97.23  |      2405     |        7044     |               nan      |
| NMOS_OFF | baseline_mean   | -0.0008269 |          116.7   |       680.7   |        2238     |               nan      |
| PMOS_ON  | baseline_mean   | -0.0006995 |           96.79  |      2592     |        8693     |               nan      |
| PMOS_OFF | baseline_mean   | -0.001261  |          116.2   |       761.3   |        1905     |               nan      |

## Per terminal

| case     | family          | target   |     r2_log |   median_err_pct |   p95_err_pct |   sign_acc_pct |   train_r2_log |   train_median_err_pct |
|:---------|:----------------|:---------|-----------:|-----------------:|--------------:|---------------:|---------------:|-----------------------:|
| NMOS_OFF | mlp             | id       |  0.9981    |           3.712  |        18.42  |          99.8  |         0.9998 |                 2.665  |
| NMOS_OFF | mlp             | ig       |  0.9997    |           1.31   |         4.799 |         100    |         0.9999 |                 1.103  |
| NMOS_OFF | mlp             | is       |  0.9978    |           5.509  |        23.42  |          99.73 |         0.9996 |                 4.74   |
| NMOS_OFF | mlp             | ib       |  0.9985    |           0.9115 |         8.433 |         100    |         0.9989 |                 0.6461 |
| NMOS_OFF | poly_elasticnet | id       |  0.9502    |          60.97   |       320.2   |          98.4  |         0.9559 |                57.52   |
| NMOS_OFF | poly_elasticnet | ig       |  0.9995    |           1.671  |         5.994 |         100    |         0.9995 |                 1.714  |
| NMOS_OFF | poly_elasticnet | is       |  0.9469    |          66.52   |       380.1   |          98.47 |         0.9524 |                60.41   |
| NMOS_OFF | poly_elasticnet | ib       |  0.9873    |          13.01   |        35.96  |         100    |         0.9846 |                12.51   |
| NMOS_OFF | random_forest   | id       |  0.9879    |          14.98   |        92.28  |          98.8  |         0.9985 |                 5.814  |
| NMOS_OFF | random_forest   | ig       |  0.9896    |           9.842  |        39.18  |         100    |         0.9986 |                 3.879  |
| NMOS_OFF | random_forest   | is       |  0.9892    |          16.13   |        87.52  |          99.07 |         0.9984 |                 6.166  |
| NMOS_OFF | random_forest   | ib       |  0.9989    |           0.2275 |        10.14  |         100    |         0.9996 |                 0.1036 |
| NMOS_OFF | xgboost         | id       |  0.9915    |           7.893  |        72.47  |          99.4  |         1      |                 1.495  |
| NMOS_OFF | xgboost         | ig       |  0.998     |           4.925  |        16.07  |         100    |         0.9996 |                 2.377  |
| NMOS_OFF | xgboost         | is       |  0.9912    |           7.631  |        61.51  |          99.2  |         1      |                 1.64   |
| NMOS_OFF | xgboost         | ib       |  0.9995    |           0.5215 |         7.931 |         100    |         0.9998 |                 0.4298 |
| NMOS_ON  | mlp             | id       |  1         |           1.537  |         7.821 |          99.93 |         1      |                 0.5545 |
| NMOS_ON  | mlp             | ig       |  0.9995    |           0.6477 |         7.191 |         100    |         1      |                 0.1298 |
| NMOS_ON  | mlp             | is       |  1         |           1.592  |         7.674 |          99.93 |         1      |                 0.5687 |
| NMOS_ON  | mlp             | ib       |  1         |           0.5757 |         2.651 |         100    |         1      |                 0.1192 |
| NMOS_ON  | poly_elasticnet | id       |  0.9997    |          17.09   |        56.47  |          99.93 |         0.9997 |                17.06   |
| NMOS_ON  | poly_elasticnet | ig       |  0.9966    |           6.972  |        24.66  |         100    |         0.9964 |                 6.94   |
| NMOS_ON  | poly_elasticnet | is       |  0.9997    |          17.09   |        57.09  |          99.93 |         0.9997 |                17.02   |
| NMOS_ON  | poly_elasticnet | ib       |  0.9926    |          14.05   |        38.78  |         100    |         0.9933 |                13.37   |
| NMOS_ON  | random_forest   | id       |  0.9997    |          11.47   |        66.43  |         100    |         0.9999 |                 4.253  |
| NMOS_ON  | random_forest   | ig       |  0.9921    |          13.38   |        43.1   |         100    |         0.9989 |                 4.684  |
| NMOS_ON  | random_forest   | is       |  0.9997    |          11.39   |        66.17  |         100    |         0.9999 |                 4.237  |
| NMOS_ON  | random_forest   | ib       |  0.979     |          15.69   |        76.93  |         100    |         0.9974 |                 5.351  |
| NMOS_ON  | xgboost         | id       |  0.9999    |           6.114  |        36.51  |         100    |         1      |                 1.183  |
| NMOS_ON  | xgboost         | ig       |  0.9986    |           5.313  |        17.53  |         100    |         0.9998 |                 2.419  |
| NMOS_ON  | xgboost         | is       |  0.9999    |           5.689  |        35.09  |         100    |         1      |                 1.142  |
| NMOS_ON  | xgboost         | ib       |  0.9963    |           7.962  |        30.49  |         100    |         0.9994 |                 3.565  |
| PMOS_OFF | mlp             | id       |  0.9984    |           4.215  |        17.6   |          99.8  |         0.9995 |                 3.255  |
| PMOS_OFF | mlp             | ig       |  0.9996    |           1.514  |         5.809 |         100    |         0.9998 |                 1.313  |
| PMOS_OFF | mlp             | is       |  0.9916    |           4.77   |        29.03  |          99.13 |         0.9993 |                 3.799  |
| PMOS_OFF | mlp             | ib       |  0.9969    |           1.374  |         7.705 |         100    |         0.9996 |                 1.025  |
| PMOS_OFF | poly_elasticnet | id       |  0.9355    |          55.12   |       215.6   |          98.93 |         0.9332 |                53.24   |
| PMOS_OFF | poly_elasticnet | ig       |  0.999     |           1.985  |         7.848 |         100    |         0.9987 |                 2.05   |
| PMOS_OFF | poly_elasticnet | is       |  0.924     |          57.27   |       205.5   |          99.27 |         0.9333 |                50.25   |
| PMOS_OFF | poly_elasticnet | ib       |  0.9841    |          13.01   |        37.45  |         100    |         0.9854 |                12.61   |
| PMOS_OFF | random_forest   | id       |  0.9814    |          10.64   |        77.26  |          99    |         0.9973 |                 4.251  |
| PMOS_OFF | random_forest   | ig       |  0.9878    |          10.19   |        40.19  |         100    |         0.9983 |                 3.882  |
| PMOS_OFF | random_forest   | is       |  0.9761    |          11.66   |        83.22  |          98.87 |         0.9967 |                 4.421  |
| PMOS_OFF | random_forest   | ib       |  0.9975    |           0.6356 |        12.03  |         100    |         0.9996 |                 0.4279 |
| PMOS_OFF | xgboost         | id       |  0.9871    |           5.432  |        73.58  |          99.2  |         0.9999 |                 1.936  |
| PMOS_OFF | xgboost         | ig       |  0.9975    |           5.12   |        17.14  |         100    |         0.9996 |                 2.206  |
| PMOS_OFF | xgboost         | is       |  0.9838    |           5.628  |        67.2   |          98.93 |         1      |                 1.431  |
| PMOS_OFF | xgboost         | ib       |  0.9993    |           0.6889 |         7.088 |         100    |         1      |                 0.3751 |
| PMOS_ON  | mlp             | id       |  1         |           4.097  |        19.19  |          99.93 |         1      |                 2.103  |
| PMOS_ON  | mlp             | ig       |  0.9988    |           1.087  |        11.78  |         100    |         0.9999 |                 0.4404 |
| PMOS_ON  | mlp             | is       |  1         |           4.165  |        19.96  |         100    |         1      |                 1.919  |
| PMOS_ON  | mlp             | ib       |  0.9999    |           1.173  |         4.587 |         100    |         1      |                 0.4158 |
| PMOS_ON  | poly_elasticnet | id       |  0.9997    |          18.02   |        61.86  |          99.6  |         0.9997 |                17.4    |
| PMOS_ON  | poly_elasticnet | ig       |  0.9951    |           4.785  |        32.26  |         100    |         0.9959 |                 4.734  |
| PMOS_ON  | poly_elasticnet | is       |  0.9997    |          18.57   |        68.43  |          99.8  |         0.9996 |                18.06   |
| PMOS_ON  | poly_elasticnet | ib       |  0.9916    |          13.77   |        40.65  |         100    |         0.9928 |                13.04   |
| PMOS_ON  | random_forest   | id       |  0.9995    |          13.15   |        81.04  |          99.87 |         0.9999 |                 5.488  |
| PMOS_ON  | random_forest   | ig       |  0.9932    |          11.42   |        39.07  |         100    |         0.9991 |                 4.149  |
| PMOS_ON  | random_forest   | is       |  0.9996    |          13.46   |        82.41  |          99.87 |         0.9999 |                 5.485  |
| PMOS_ON  | random_forest   | ib       |  0.9828    |          11.79   |        61.19  |         100    |         0.9973 |                 4.579  |
| PMOS_ON  | xgboost         | id       |  0.9999    |           5.947  |        38.09  |          99.87 |         1      |                 1.246  |
| PMOS_ON  | xgboost         | ig       |  0.9989    |           4.59   |        14.42  |         100    |         0.9998 |                 2.093  |
| PMOS_ON  | xgboost         | is       |  0.9999    |           5.815  |        36.34  |          99.93 |         1      |                 1.277  |
| PMOS_ON  | xgboost         | ib       |  0.9963    |           7.498  |        27.04  |         100    |         0.9993 |                 3.489  |
| NMOS_ON  | baseline_mean   | id       | -0.001016  |         100      |       100     |          52.27 |       nan      |               nan      |
| NMOS_ON  | baseline_mean   | ig       | -0.001073  |          92.48   |      7044     |         100    |       nan      |               nan      |
| NMOS_ON  | baseline_mean   | is       | -0.001017  |         100      |       100     |          52.27 |       nan      |               nan      |
| NMOS_ON  | baseline_mean   | ib       | -8.471e-05 |          96.45   |      2376     |         100    |       nan      |               nan      |
| NMOS_OFF | baseline_mean   | id       | -0.001888  |          98.18   |       164.4   |          78.4  |       nan      |               nan      |
| NMOS_OFF | baseline_mean   | ig       | -5.737e-05 |          84.3    |      2238     |         100    |       nan      |               nan      |
| NMOS_OFF | baseline_mean   | is       | -0.001244  |          98.55   |       115.5   |          80.07 |       nan      |               nan      |
| NMOS_OFF | baseline_mean   | ib       | -0.0001183 |         185.7    |       205.1   |         100    |       nan      |               nan      |
| PMOS_ON  | baseline_mean   | id       | -0.001238  |         100      |       100     |          48.33 |       nan      |               nan      |
| PMOS_ON  | baseline_mean   | ig       | -0.0003077 |          89.77   |      8693     |         100    |       nan      |               nan      |
| PMOS_ON  | baseline_mean   | is       | -0.001225  |         100      |       100     |          48.87 |       nan      |               nan      |
| PMOS_ON  | baseline_mean   | ib       | -2.701e-05 |          97.39   |      1476     |         100    |       nan      |               nan      |
| PMOS_OFF | baseline_mean   | id       | -1.752e-05 |          94.22   |       409.9   |          86.67 |       nan      |               nan      |
| PMOS_OFF | baseline_mean   | ig       | -0.001058  |          84.08   |      1905     |         100    |       nan      |               nan      |
| PMOS_OFF | baseline_mean   | is       | -0.003965  |          93.4    |       523.2   |          85.2  |       nan      |               nan      |
| PMOS_OFF | baseline_mean   | ib       | -4.642e-06 |         193.1    |       206.6   |         100    |       nan      |               nan      |
