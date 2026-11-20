| Pipeline | mAP50 | mAP50-95 |
|---|---|---|
| yolov10x (single scale) | 37.5 | 23.0 |
| yolov10x (multi-scale boost) | 44.8 | 29.6 |
| doclayout (single scale) | 37.1 | 26.3 |
| doclayout (multi-scale boost) | 42.0 | 29.4 |
| ensemble (equal weights) | 49.4 | 33.7 |
| ensemble (tuned weights) | 48.7 | 33.7 |

| Class | GT boxes | yolov10x (single scale) AP50-95 | yolov10x (multi-scale boost) AP50-95 | doclayout (single scale) AP50-95 | doclayout (multi-scale boost) AP50-95 | ensemble (equal weights) AP50-95 | ensemble (tuned weights) AP50-95 |
|---|---|---|---|---|---|---|---|
| Headline | 683 | 33.6 | 35.6 | 32.0 | 37.5 | 39.9 | 40.5 |
| Subtitle | 745 | 10.6 | 11.8 | 5.5 | 6.4 | 11.9 | 10.0 |
| BodyText | 2963 | 44.8 | 48.3 | 60.4 | 64.7 | 64.1 | 65.1 |
| Caption | 22 | 11.5 | 9.9 | 11.6 | 13.8 | 14.2 | 12.6 |
| Image | 693 | 62.9 | 63.4 | 65.9 | 68.3 | 68.5 | 70.2 |
| Header | 62 | 24.8 | 27.2 | 32.3 | 31.1 | 30.5 | 31.1 |
| Footer | 2 | 25.0 | 65.3 | 0.0 | 10.1 | 60.3 | 55.3 |
| Folio | 99 | 1.4 | 3.0 | 4.4 | 6.2 | 5.3 | 6.0 |
| PageNumber | 71 | 0.4 | 2.9 | 0.7 | 0.8 | 2.6 | 2.3 |
| Flag | 10 | 15.0 | 29.0 | 50.2 | 55.3 | 40.1 | 43.5 |