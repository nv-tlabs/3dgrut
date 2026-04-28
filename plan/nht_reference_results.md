# NHT Reference Results

## Bonsai Comparison

| Run | Source | PSNR | SSIM | LPIPS | CC PSNR | CC SSIM | CC LPIPS | Std PSNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GSplat reference | GSplat NHT reference run | 34.163 | 0.9542 | 0.235 | - | - | - | - |
| 3DGRUT previous | Before benchmark-parity/color-refinement fixes | 33.427 | 0.949 | 0.248 | 33.559 | 0.947 | 0.247 | 2.489 |
| 3DGRUT updated | After benchmark-parity/color-refinement fixes | 33.734 | 0.951 | 0.246 | 33.908 | 0.949 | 0.246 | 2.455 |
| 3DGRUT T1-T3 | After depth-gate and EMA eval fixes | 33.702 | 0.951 | 0.246 | 33.853 | 0.949 | 0.246 | 2.520 |

## Timing Notes

- GSplat reference render time: `11.000 ms/image`.
- 3DGRUT timing needs a timing-enabled eval; the table's last 3DGRUT column is `std_psnr`, not time.

