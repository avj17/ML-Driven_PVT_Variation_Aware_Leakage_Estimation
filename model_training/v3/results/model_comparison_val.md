# Model comparison (validation set, mean over Id/Ig/Is/Ib)

| case     | family        |     r2_log |   median_err_pct |   p95_err_pct |   worst_p95_pct |   train_median_err_pct |
|:---------|:--------------|-----------:|-----------------:|--------------:|----------------:|-----------------------:|
| NMOS_OFF | mlp           |  1         |          0.06443 |     0.3986    |       0.5552    |                0.06023 |
| NMOS_ON  | mlp           |  1         |          0.09239 |     0.5343    |       0.674     |                0.06455 |
| PMOS_OFF | mlp           |  1         |          0.0559  |     0.3826    |       0.5045    |                0.04876 |
| PMOS_ON  | mlp           |  1         |          0.1443  |     0.6221    |       0.6956    |                0.1364  |
| NMOS_ON  | baseline_mean | -0.000891  |         97.3     |  6322         |       1.799e+04 |              nan       |
| NMOS_OFF | baseline_mean | -0.0002552 |         95.22    |     2.535e+04 |       9.891e+04 |              nan       |
| PMOS_ON  | baseline_mean | -0.0007208 |         96.97    |  8339         |       2.447e+04 |              nan       |
| PMOS_OFF | baseline_mean | -0.001002  |         92.88    |     3.017e+04 |       1.179e+05 |              nan       |

## Per terminal

| case     | family        | target   |     r2_log |   median_err_pct |   p95_err_pct |   sign_acc_pct |   train_r2_log |   train_median_err_pct |
|:---------|:--------------|:---------|-----------:|-----------------:|--------------:|---------------:|---------------:|-----------------------:|
| NMOS_OFF | mlp           | id       |  1         |          0.06272 |     0.4575    |         100    |              1 |                0.05688 |
| NMOS_OFF | mlp           | ig       |  1         |          0.0381  |     0.1694    |         100    |              1 |                0.03466 |
| NMOS_OFF | mlp           | is       |  1         |          0.05964 |     0.4121    |         100    |              1 |                0.05685 |
| NMOS_OFF | mlp           | ib       |  1         |          0.09727 |     0.5552    |         100    |              1 |                0.09254 |
| NMOS_ON  | mlp           | id       |  1         |          0.09771 |     0.5226    |          99.93 |              1 |                0.07198 |
| NMOS_ON  | mlp           | ig       |  1         |          0.08311 |     0.674     |         100    |              1 |                0.05426 |
| NMOS_ON  | mlp           | is       |  1         |          0.09768 |     0.5224    |         100    |              1 |                0.07198 |
| NMOS_ON  | mlp           | ib       |  1         |          0.09105 |     0.418     |         100    |              1 |                0.05999 |
| PMOS_OFF | mlp           | id       |  1         |          0.05091 |     0.4216    |         100    |              1 |                0.04459 |
| PMOS_OFF | mlp           | ig       |  1         |          0.03037 |     0.1425    |         100    |              1 |                0.02669 |
| PMOS_OFF | mlp           | is       |  1         |          0.05238 |     0.4615    |         100    |              1 |                0.04457 |
| PMOS_OFF | mlp           | ib       |  1         |          0.08994 |     0.5045    |         100    |              1 |                0.07918 |
| PMOS_ON  | mlp           | id       |  1         |          0.153   |     0.653     |         100    |              1 |                0.142   |
| PMOS_ON  | mlp           | ig       |  1         |          0.1185  |     0.6956    |         100    |              1 |                0.118   |
| PMOS_ON  | mlp           | is       |  1         |          0.1525  |     0.641     |         100    |              1 |                0.1417  |
| PMOS_ON  | mlp           | ib       |  1         |          0.1531  |     0.4987    |         100    |              1 |                0.1442  |
| NMOS_ON  | baseline_mean | id       | -0.001254  |        100       |   100         |          47.73 |            nan |              nan       |
| NMOS_ON  | baseline_mean | ig       | -0.0008908 |         92.43    |  7094         |         100    |            nan |              nan       |
| NMOS_ON  | baseline_mean | is       | -0.001257  |        100       |   100         |          47.73 |            nan |              nan       |
| NMOS_ON  | baseline_mean | ib       | -0.0001624 |         96.75    |     1.799e+04 |         100    |            nan |              nan       |
| NMOS_OFF | baseline_mean | id       | -0.0005919 |         98.18    |   163.2       |          78.73 |            nan |              nan       |
| NMOS_OFF | baseline_mean | ig       | -5.941e-08 |         84.36    |  2207         |         100    |            nan |              nan       |
| NMOS_OFF | baseline_mean | is       | -0.0003976 |         98.34    |   120.3       |          80.2  |            nan |              nan       |
| NMOS_OFF | baseline_mean | ib       | -3.109e-05 |         99.99    |     9.891e+04 |         100    |            nan |              nan       |
| PMOS_ON  | baseline_mean | id       | -0.001234  |        100       |   100         |          48.33 |            nan |              nan       |
| PMOS_ON  | baseline_mean | ig       | -0.0003684 |         90.01    |  8684         |         100    |            nan |              nan       |
| PMOS_ON  | baseline_mean | is       | -0.001209  |        100       |   100         |          48.87 |            nan |              nan       |
| PMOS_ON  | baseline_mean | ib       | -7.19e-05  |         97.85    |     2.447e+04 |         100    |            nan |              nan       |
| PMOS_OFF | baseline_mean | id       | -0.0002411 |         93.79    |   414.3       |          86.87 |            nan |              nan       |
| PMOS_OFF | baseline_mean | ig       | -0.0009935 |         84.08    |  1902         |         100    |            nan |              nan       |
| PMOS_OFF | baseline_mean | is       | -0.002766  |         93.66    |   485.2       |          85.2  |            nan |              nan       |
| PMOS_OFF | baseline_mean | ib       | -7.374e-06 |         99.99    |     1.179e+05 |         100    |            nan |              nan       |
