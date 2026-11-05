# Feature ablation (validation)

| case     | family                    |   r2_log |   median_err_pct |   p95_err_pct |   worst_p95_pct |   train_median_err_pct |
|:---------|:--------------------------|---------:|-----------------:|--------------:|----------------:|-----------------------:|
| NMOS_OFF | mlp (physics)             |   0.9989 |           0.5349 |         6.105 |           9.957 |                 0.1713 |
| NMOS_OFF | poly_elasticnet (physics) |   0.971  |          35.54   |       185.6   |         380.1   |                33.04   |
| NMOS_OFF | random_forest (physics)   |   0.9914 |          10.29   |        57.28  |          92.28  |                 3.991  |
| NMOS_OFF | xgboost (physics)         |   0.9951 |           5.243  |        39.49  |          72.47  |                 1.485  |
| NMOS_ON  | mlp (physics)             |   0.9999 |           0.5129 |         3.749 |           6.516 |                 0.1318 |
| NMOS_ON  | poly_elasticnet (physics) |   0.9971 |          13.8    |        44.25  |          57.09  |                13.6    |
| NMOS_ON  | random_forest (physics)   |   0.9926 |          12.98   |        63.16  |          76.93  |                 4.631  |
| NMOS_ON  | xgboost (physics)         |   0.9987 |           6.269  |        29.9   |          36.51  |                 2.077  |
| PMOS_OFF | mlp (physics)             |   0.9993 |           0.3965 |         4.692 |           7.455 |                 0.101  |
| PMOS_OFF | poly_elasticnet (physics) |   0.9607 |          31.85   |       116.6   |         215.6   |                29.54   |
| PMOS_OFF | random_forest (physics)   |   0.9857 |           8.28   |        53.18  |          83.22  |                 3.246  |
| PMOS_OFF | xgboost (physics)         |   0.992  |           4.217  |        41.25  |          73.58  |                 1.487  |
| PMOS_ON  | mlp (physics)             |   0.9997 |           0.6893 |         5.352 |          10.16  |                 0.3125 |
| PMOS_ON  | poly_elasticnet (physics) |   0.9965 |          13.79   |        50.8   |          68.43  |                13.31   |
| PMOS_ON  | random_forest (physics)   |   0.9938 |          12.46   |        65.93  |          82.41  |                 4.925  |
| PMOS_ON  | xgboost (physics)         |   0.9987 |           5.963  |        28.97  |          38.09  |                 2.026  |
| NMOS_OFF | mlp (raw)                 |   0.9975 |           1.203  |        13.44  |          21.75  |                 0.2328 |
| NMOS_OFF | xgboost (raw)             |   0.993  |           5.924  |        49.94  |          93.11  |                 1.677  |
| NMOS_ON  | mlp (raw)                 |   0.9997 |           1.72   |        14.76  |          20.69  |                 0.1568 |
| NMOS_ON  | xgboost (raw)             |   0.9876 |          12.67   |        69.88  |         100     |                 6.288  |
| PMOS_OFF | mlp (raw)                 |   0.9971 |           1.085  |        12.91  |          21.65  |                 0.1946 |
| PMOS_OFF | xgboost (raw)             |   0.9812 |           6.048  |        55.13  |          96.43  |                 1.913  |
| PMOS_ON  | mlp (raw)                 |   0.9994 |           1.81   |        19     |          26.61  |                 0.2761 |
| PMOS_ON  | xgboost (raw)             |   0.9929 |          11.88   |        64.04  |         100     |                 5.905  |
