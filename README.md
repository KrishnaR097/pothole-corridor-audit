# Pothole Corridor Audit

Modular CLI pipeline for dashcam pothole detection, metric dimension estimation, safety analytics, and municipal complaint packaging.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python audit\audit_cli.py --video path\to\video.mp4 --route "RVCE to KR Market, Bengaluru" --distance-km 4.5 --weights models\best.pt --out-dir outputs
python complaint\complaint_cli.py --route "RVCE to KR Market, Bengaluru" --distance-km 4.5 --audit-dir outputs --complaints-dir complaints
```
