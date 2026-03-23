# AggPredict v2.0

AI-based aggregation risk prediction model for high-concentration 
protein formulations (>100 mg/mL).

## Features
- Donnan effect microenvironment pH correction
- 8 independent physicochemical risk modules
- Excipient protection scoring (surfactant / sugar / amino acid)
- DOE grid scan & one-at-a-time sensitivity analysis
- Experimental data calibration interface
- ML upgrade roadmap (Ridge → GP → ESM-2)

## Supported Modalities
`mAb` · `ADC` · `Peptide` · `SC formulation` · `Intranasal` · `Microneedle`

## Quick Start
```bash
pip install numpy
python aggpredict_v2.py

# 추가 옵션
python aggpredict_v2.py --schema      # HT 스크리닝 데이터 스키마
python aggpredict_v2.py --roadmap     # AI 고도화 로드맵
python aggpredict_v2.py --visualize   # 시각화 (matplotlib, seaborn 필요)
```

## Requirements
- Python 3.9+
- numpy (core model)
- matplotlib, seaborn (시각화, optional)
- scikit-learn (데이터 보정, optional)

## License
MIT
