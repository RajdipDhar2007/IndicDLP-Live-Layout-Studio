| Pipeline | mAP50 | mAP50-95 |
|---|---|---|
| yolov10x (single scale) | 88.5 | 84.1 |
| yolov10x (multi-scale boost) | 93.0 | 89.8 |
| doclayout (single scale) | 88.9 | 85.0 |
| doclayout (multi-scale boost) | 92.9 | 90.3 |
| ensemble (equal weights) | 99.5 | 97.9 |
| ensemble (tuned weights) | 99.5 | 97.9 |

| Class | GT boxes | yolov10x (single scale) AP50-95 | yolov10x (multi-scale boost) AP50-95 | doclayout (single scale) AP50-95 | doclayout (multi-scale boost) AP50-95 | ensemble (equal weights) AP50-95 | ensemble (tuned weights) AP50-95 |
|---|---|---|---|---|---|---|---|
| Headline | 683 | 82.7 | 89.3 | 87.4 | 92.6 | 96.6 | 96.6 |
| Subtitle | 745 | 71.1 | 76.7 | 87.2 | 94.1 | 97.1 | 97.1 |
| BodyText | 2963 | 92.5 | 95.1 | 92.8 | 95.3 | 97.9 | 97.9 |
| Caption | 22 | 80.1 | 92.4 | 66.4 | 76.2 | 98.8 | 98.8 |
| Image | 693 | 94.3 | 95.8 | 91.3 | 93.5 | 98.9 | 98.9 |
| Header | 0 | - | - | - | - | - | - |
| Footer | 0 | - | - | - | - | - | - |
| Folio | 0 | - | - | - | - | - | - |
| PageNumber | 0 | - | - | - | - | - | - |
| Flag | 0 | - | - | - | - | - | - |