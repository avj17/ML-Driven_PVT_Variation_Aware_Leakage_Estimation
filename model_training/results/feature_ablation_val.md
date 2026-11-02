# Feature ablation (validation)

| case     | family                    |   r2_log |   median_err_pct |   p95_err_pct |   worst_p95_pct |   train_median_err_pct |
|:---------|:--------------------------|---------:|-----------------:|--------------:|----------------:|-----------------------:|
| NMOS_OFF | mlp (physics)             |   0.9985 |            2.861 |        13.77  |          23.42  |                 2.289  |
| NMOS_OFF | poly_elasticnet (physics) |   0.971  |           35.54  |       185.6   |         380.1   |                33.04   |
| NMOS_OFF | random_forest (physics)   |   0.9914 |           10.29  |        57.28  |          92.28  |                 3.991  |
| NMOS_OFF | xgboost (physics)         |   0.9951 |            5.243 |        39.49  |          72.47  |                 1.485  |
| NMOS_ON  | mlp (physics)             |   0.9999 |            1.088 |         6.334 |           7.821 |                 0.3431 |
| NMOS_ON  | poly_elasticnet (physics) |   0.9971 |           13.8   |        44.25  |          57.09  |                13.6    |
| NMOS_ON  | random_forest (physics)   |   0.9926 |           12.98  |        63.16  |          76.93  |                 4.631  |
| NMOS_ON  | xgboost (physics)         |   0.9987 |            6.269 |        29.9   |          36.51  |                 2.077  |
| PMOS_OFF | mlp (physics)             |   0.9966 |            2.968 |        15.04  |          29.03  |                 2.348  |
| PMOS_OFF | poly_elasticnet (physics) |   0.9607 |           31.85  |       116.6   |         215.6   |                29.54   |
| PMOS_OFF | random_forest (physics)   |   0.9857 |            8.28  |        53.18  |          83.22  |                 3.246  |
| PMOS_OFF | xgboost (physics)         |   0.992  |            4.217 |        41.25  |          73.58  |                 1.487  |
| PMOS_ON  | mlp (physics)             |   0.9997 |            2.63  |        13.88  |          19.96  |                 1.219  |
| PMOS_ON  | poly_elasticnet (physics) |   0.9965 |           13.79  |        50.8   |          68.43  |                13.31   |
| PMOS_ON  | random_forest (physics)   |   0.9938 |           12.46  |        65.93  |          82.41  |                 4.925  |
| PMOS_ON  | xgboost (physics)         |   0.9987 |            5.963 |        28.97  |          38.09  |                 2.026  |
| NMOS_OFF | mlp (raw)                 |   0.9919 |            5.668 |        34.93  |          60.43  |                 3.879  |
| NMOS_OFF | xgboost (raw)             |   0.993  |            5.924 |        49.94  |          93.11  |                 1.677  |
| NMOS_ON  | mlp (raw)                 |   0.9925 |           12.45  |        58.81  |          99.96  |                 7.55   |
| NMOS_ON  | xgboost (raw)             |   0.9876 |           12.67  |        69.88  |         100     |                 6.288  |
| PMOS_OFF | mlp (raw)                 |   0.9922 |            3.518 |        27.78  |          52.26  |                 2.198  |
| PMOS_OFF | xgboost (raw)             |   0.9812 |            6.048 |        55.13  |          96.43  |                 1.913  |
| PMOS_ON  | mlp (raw)                 |   0.996  |            6.089 |        41.95  |          71.68  |                 2.553  |
| PMOS_ON  | xgboost (raw)             |   0.9929 |           11.88  |        64.04  |         100     |                 5.905  |
