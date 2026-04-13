# Metropolis Analysis Summary

## Dataset
- Inputs: jaguar, lion, tiger, zebra
- Methods: method_1_uniform_init, method_2_rejection_init
- Samples per pixel: 1, 4, 8, 64, 256, 1024

## Aggregate Metrics by Method and SPP

| method | spp | mse | mae | psnr | ssim |
|---|---:|---:|---:|---:|---:|
| method_1_uniform_init | 1 | 0.123797 | 0.274388 | 9.2947 | 0.40292 |
| method_1_uniform_init | 4 | 0.046893 | 0.160597 | 13.5301 | 0.69191 |
| method_1_uniform_init | 8 | 0.026527 | 0.118645 | 16.0327 | 0.80566 |
| method_1_uniform_init | 64 | 0.003973 | 0.044646 | 24.3784 | 0.96700 |
| method_1_uniform_init | 256 | 0.001043 | 0.022678 | 30.2406 | 0.99123 |
| method_1_uniform_init | 1024 | 0.000268 | 0.011402 | 36.1478 | 0.99773 |
| method_2_rejection_init | 1 | 0.123798 | 0.274389 | 9.2947 | 0.40291 |
| method_2_rejection_init | 4 | 0.046893 | 0.160597 | 13.5301 | 0.69191 |
| method_2_rejection_init | 8 | 0.026527 | 0.118645 | 16.0327 | 0.80566 |
| method_2_rejection_init | 64 | 0.003973 | 0.044646 | 24.3784 | 0.96700 |
| method_2_rejection_init | 256 | 0.001043 | 0.022678 | 30.2406 | 0.99123 |
| method_2_rejection_init | 1024 | 0.000268 | 0.011402 | 36.1478 | 0.99773 |

## Best Config per Image (max PSNR)

| image | method | spp | psnr | ssim |
|---|---|---:|---:|---:|
| jaguar | method_1_uniform_init | 1024 | 37.4043 | 0.99870 |
| lion | method_1_uniform_init | 1024 | 37.9858 | 0.99830 |
| tiger | method_1_uniform_init | 1024 | 33.1487 | 0.99671 |
| zebra | method_2_rejection_init | 1024 | 36.0524 | 0.99723 |

## Suggested Insights to Highlight
- How fast each metric saturates as spp increases (diminishing returns).
- Which images are harder to reconstruct (texture-heavy regions).
- Whether rejection-based initialization helps most at low spp.
- Quality vs computation trade-off between 64/256/1024 spp.