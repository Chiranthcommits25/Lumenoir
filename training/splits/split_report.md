# Train/validation split report

- Seed: 42
- Validation fraction: 20%
- Test pairs used for leakage checking only: 215
- Train/validation/test exact ID overlap: 0
- Train/validation match count: 0

| Dataset | Source pairs | Eligible | Train | Validation |
| --- | ---: | ---: | ---: | ---: |
| lol_v1 | 485 | 389 | 311 | 78 |
| lol_v2_real | 689 | 667 | 534 | 133 |
| lol_v2_syn | 900 | 900 | 720 | 180 |

Source-level validation/train targets: lol_v1 97/388, lol_v2_real 138/551, lol_v2_syn 180/720. 118 candidate pair(s) were excluded by the test-leakage audit; the actual split is 20% of eligible data per dataset.

## Leakage and duplicate candidates

- Test-leakage pairs: 118
- Duplicate pairs inside train/validation pool: 733
- Excluded records: 118
- Hash failures: 0

No test images were placed in train or validation. Duplicate-connected candidates were kept together in one split and were not deleted; only test-leaking components were excluded. Excluded records are listed below.

| Record | Reason |
| --- | --- |
| `lol_v1:our485/low/112.png` | test leakage: lol_v1:eval15/low/111.png hamming=2 ssim=0.9959 |
| `lol_v1:our485/low/145.png` | test leakage: lol_v1:eval15/low/146.png hamming=0 ssim=0.9984 |
| `lol_v1:our485/low/2.png` | test leakage: lol_v1:eval15/low/1.png hamming=0 ssim=0.9978 |
| `lol_v1:our485/low/21.png` | test leakage: lol_v1:eval15/low/22.png hamming=0 ssim=0.9980 |
| `lol_v1:our485/low/24.png` | test leakage: lol_v1:eval15/low/23.png hamming=2 ssim=0.9975 |
| `lol_v1:our485/low/546.png` | test leakage: lol_v1:eval15/low/547.png hamming=0 ssim=0.9967 |
| `lol_v1:our485/low/56.png` | test leakage: lol_v1:eval15/low/55.png hamming=2 ssim=0.9982 |
| `lol_v1:our485/low/664.png` | test leakage: lol_v1:eval15/low/665.png hamming=0 ssim=0.9966 |
| `lol_v1:our485/low/668.png` | test leakage: lol_v1:eval15/low/669.png hamming=2 ssim=0.9932 |
| `lol_v1:our485/low/690.png` | test leakage: lol_v2_real:Test/Low/low00690.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/691.png` | test leakage: lol_v2_real:Test/Low/low00690.png hamming=0 ssim=0.9962 |
| `lol_v1:our485/low/692.png` | test leakage: lol_v2_real:Test/Low/low00692.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/693.png` | test leakage: lol_v2_real:Test/Low/low00692.png hamming=0 ssim=0.9976 |
| `lol_v1:our485/low/694.png` | test leakage: lol_v2_real:Test/Low/low00694.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/695.png` | test leakage: lol_v2_real:Test/Low/low00694.png hamming=0 ssim=0.9974 |
| `lol_v1:our485/low/696.png` | test leakage: lol_v2_real:Test/Low/low00696.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/697.png` | test leakage: lol_v2_real:Test/Low/low00697.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/698.png` | test leakage: lol_v2_real:Test/Low/low00698.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/699.png` | test leakage: lol_v2_real:Test/Low/low00698.png hamming=0 ssim=0.9963 |
| `lol_v1:our485/low/700.png` | test leakage: lol_v2_real:Test/Low/low00700.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/701.png` | test leakage: lol_v2_real:Test/Low/low00701.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/702.png` | test leakage: lol_v2_real:Test/Low/low00702.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/703.png` | test leakage: lol_v2_real:Test/Low/low00702.png hamming=0 ssim=0.9978 |
| `lol_v1:our485/low/704.png` | test leakage: lol_v2_real:Test/Low/low00702.png hamming=0 ssim=0.9974 |
| `lol_v1:our485/low/708.png` | test leakage: lol_v2_real:Test/Low/low00708.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/709.png` | test leakage: lol_v2_real:Test/Low/low00708.png hamming=0 ssim=0.9958 |
| `lol_v1:our485/low/710.png` | test leakage: lol_v2_real:Test/Low/low00710.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/711.png` | test leakage: lol_v2_real:Test/Low/low00710.png hamming=0 ssim=0.9986 |
| `lol_v1:our485/low/712.png` | test leakage: lol_v2_real:Test/Low/low00712.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/713.png` | test leakage: lol_v2_real:Test/Low/low00712.png hamming=0 ssim=0.9976 |
| `lol_v1:our485/low/714.png` | test leakage: lol_v2_real:Test/Low/low00714.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/715.png` | test leakage: lol_v2_real:Test/Low/low00714.png hamming=0 ssim=0.9980 |
| `lol_v1:our485/low/716.png` | test leakage: lol_v2_real:Test/Low/low00716.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/717.png` | test leakage: lol_v2_real:Test/Low/low00717.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/718.png` | test leakage: lol_v2_real:Test/Low/low00718.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/719.png` | test leakage: lol_v2_real:Test/Low/low00718.png hamming=0 ssim=0.9983 |
| `lol_v1:our485/low/720.png` | test leakage: lol_v2_real:Test/Low/low00720.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/721.png` | test leakage: lol_v2_real:Test/Low/low00721.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/722.png` | test leakage: lol_v2_real:Test/Low/low00722.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/723.png` | test leakage: lol_v2_real:Test/Low/low00722.png hamming=0 ssim=0.9981 |
| `lol_v1:our485/low/724.png` | test leakage: lol_v2_real:Test/Low/low00724.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/725.png` | test leakage: lol_v2_real:Test/Low/low00724.png hamming=0 ssim=0.9956 |
| `lol_v1:our485/low/726.png` | test leakage: lol_v2_real:Test/Low/low00726.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/727.png` | test leakage: lol_v2_real:Test/Low/low00727.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/728.png` | test leakage: lol_v2_real:Test/Low/low00728.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/729.png` | test leakage: lol_v2_real:Test/Low/low00729.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/730.png` | test leakage: lol_v2_real:Test/Low/low00729.png hamming=0 ssim=0.9976 |
| `lol_v1:our485/low/731.png` | test leakage: lol_v2_real:Test/Low/low00731.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/732.png` | test leakage: lol_v2_real:Test/Low/low00731.png hamming=0 ssim=0.9975 |
| `lol_v1:our485/low/733.png` | test leakage: lol_v2_real:Test/Low/low00733.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/734.png` | test leakage: lol_v2_real:Test/Low/low00733.png hamming=0 ssim=0.9983 |
| `lol_v1:our485/low/735.png` | test leakage: lol_v2_real:Test/Low/low00735.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/736.png` | test leakage: lol_v2_real:Test/Low/low00735.png hamming=0 ssim=0.9968 |
| `lol_v1:our485/low/737.png` | test leakage: lol_v2_real:Test/Low/low00737.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/738.png` | test leakage: lol_v2_real:Test/Low/low00737.png hamming=0 ssim=0.9944 |
| `lol_v1:our485/low/739.png` | test leakage: lol_v2_real:Test/Low/low00739.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/740.png` | test leakage: lol_v2_real:Test/Low/low00740.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/741.png` | test leakage: lol_v2_real:Test/Low/low00741.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/743.png` | test leakage: lol_v2_real:Test/Low/low00743.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/744.png` | test leakage: lol_v2_real:Test/Low/low00744.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/745.png` | test leakage: lol_v2_real:Test/Low/low00745.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/746.png` | test leakage: lol_v2_real:Test/Low/low00745.png hamming=0 ssim=0.9974 |
| `lol_v1:our485/low/747.png` | test leakage: lol_v1:eval15/low/748.png hamming=0 ssim=0.9956 |
| `lol_v1:our485/low/750.png` | test leakage: lol_v2_real:Test/Low/low00750.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/751.png` | test leakage: lol_v2_real:Test/Low/low00751.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/752.png` | test leakage: lol_v2_real:Test/Low/low00752.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/757.png` | test leakage: lol_v2_real:Test/Low/low00757.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/758.png` | test leakage: lol_v2_real:Test/Low/low00758.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/759.png` | test leakage: lol_v2_real:Test/Low/low00759.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/760.png` | test leakage: lol_v2_real:Test/Low/low00760.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/761.png` | test leakage: lol_v2_real:Test/Low/low00761.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/762.png` | test leakage: lol_v2_real:Test/Low/low00762.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/763.png` | test leakage: lol_v2_real:Test/Low/low00763.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/764.png` | test leakage: lol_v2_real:Test/Low/low00764.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/765.png` | test leakage: lol_v2_real:Test/Low/low00765.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/766.png` | test leakage: lol_v2_real:Test/Low/low00766.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/769.png` | test leakage: lol_v2_real:Test/Low/low00769.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/770.png` | test leakage: lol_v2_real:Test/Low/low00769.png hamming=0 ssim=0.9980 |
| `lol_v1:our485/low/771.png` | test leakage: lol_v2_real:Test/Low/low00771.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/772.png` | test leakage: lol_v2_real:Test/Low/low00772.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/773.png` | test leakage: lol_v2_real:Test/Low/low00773.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/774.png` | test leakage: lol_v2_real:Test/Low/low00774.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/775.png` | test leakage: lol_v2_real:Test/Low/low00775.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/776.png` | test leakage: lol_v2_real:Test/Low/low00775.png hamming=0 ssim=0.9941 |
| `lol_v1:our485/low/777.png` | test leakage: lol_v2_real:Test/Low/low00777.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/779.png` | test leakage: lol_v1:eval15/low/780.png hamming=0 ssim=0.9978 |
| `lol_v1:our485/low/781.png` | test leakage: lol_v2_real:Test/Low/low00781.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/782.png` | test leakage: lol_v2_real:Test/Low/low00782.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/783.png` | test leakage: lol_v2_real:Test/Low/low00782.png hamming=0 ssim=0.9979 |
| `lol_v1:our485/low/784.png` | test leakage: lol_v2_real:Test/Low/low00784.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/785.png` | test leakage: lol_v2_real:Test/Low/low00784.png hamming=0 ssim=0.9977 |
| `lol_v1:our485/low/786.png` | test leakage: lol_v2_real:Test/Low/low00784.png hamming=0 ssim=0.9977 |
| `lol_v1:our485/low/787.png` | test leakage: lol_v2_real:Test/Low/low00787.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/788.png` | test leakage: lol_v2_real:Test/Low/low00788.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/789.png` | test leakage: lol_v2_real:Test/Low/low00789.png hamming=0 ssim=1.0000 |
| `lol_v1:our485/low/80.png` | test leakage: lol_v1:eval15/low/79.png hamming=0 ssim=0.9977 |
| `lol_v2_real:Train/Low/low00001.png` | test leakage: lol_v1:eval15/low/1.png hamming=0 ssim=1.0000 |
| `lol_v2_real:Train/Low/low00002.png` | test leakage: lol_v1:eval15/low/1.png hamming=0 ssim=0.9978 |
| `lol_v2_real:Train/Low/low00021.png` | test leakage: lol_v1:eval15/low/22.png hamming=0 ssim=0.9980 |
| `lol_v2_real:Train/Low/low00022.png` | test leakage: lol_v1:eval15/low/22.png hamming=0 ssim=1.0000 |
| `lol_v2_real:Train/Low/low00023.png` | test leakage: lol_v1:eval15/low/23.png hamming=0 ssim=1.0000 |
| `lol_v2_real:Train/Low/low00024.png` | test leakage: lol_v1:eval15/low/23.png hamming=2 ssim=0.9975 |
| `lol_v2_real:Train/Low/low00055.png` | test leakage: lol_v1:eval15/low/55.png hamming=0 ssim=1.0000 |
| `lol_v2_real:Train/Low/low00056.png` | test leakage: lol_v1:eval15/low/55.png hamming=2 ssim=0.9982 |
| `lol_v2_real:Train/Low/low00079.png` | test leakage: lol_v1:eval15/low/79.png hamming=0 ssim=1.0000 |
| `lol_v2_real:Train/Low/low00080.png` | test leakage: lol_v1:eval15/low/79.png hamming=0 ssim=0.9977 |
| `lol_v2_real:Train/Low/low00111.png` | test leakage: lol_v1:eval15/low/111.png hamming=0 ssim=1.0000 |
| `lol_v2_real:Train/Low/low00112.png` | test leakage: lol_v1:eval15/low/111.png hamming=2 ssim=0.9959 |
| `lol_v2_real:Train/Low/low00145.png` | test leakage: lol_v1:eval15/low/146.png hamming=0 ssim=0.9984 |
| `lol_v2_real:Train/Low/low00146.png` | test leakage: lol_v1:eval15/low/146.png hamming=0 ssim=1.0000 |
| `lol_v2_real:Train/Low/low00179.png` | test leakage: lol_v1:eval15/low/179.png hamming=0 ssim=1.0000 |
| `lol_v2_real:Train/Low/low00493.png` | test leakage: lol_v1:eval15/low/493.png hamming=0 ssim=1.0000 |
| `lol_v2_real:Train/Low/low00546.png` | test leakage: lol_v1:eval15/low/547.png hamming=0 ssim=0.9967 |
| `lol_v2_real:Train/Low/low00547.png` | test leakage: lol_v1:eval15/low/547.png hamming=0 ssim=1.0000 |
| `lol_v2_real:Train/Low/low00664.png` | test leakage: lol_v1:eval15/low/665.png hamming=0 ssim=0.9966 |
| `lol_v2_real:Train/Low/low00665.png` | test leakage: lol_v1:eval15/low/665.png hamming=0 ssim=1.0000 |
| `lol_v2_real:Train/Low/low00668.png` | test leakage: lol_v1:eval15/low/669.png hamming=2 ssim=0.9932 |
| `lol_v2_real:Train/Low/low00669.png` | test leakage: lol_v1:eval15/low/669.png hamming=0 ssim=1.0000 |

## Files

- `train.json`
- `validation.json`
- `split_report.md`
