"""
AggPredict v2.0 — High-Concentration Protein Formulation Aggregation Risk Model
================================================================================
개발 철학:
  - 완벽한 물리 모델보다 "확장 가능하고 해석 가능한 구조"를 우선
  - 각 모듈은 독립적으로 수정/교체 가능 (plug-in architecture)
  - 경험적 가정(heuristic)과 물리 기반 수식을 명확히 구분
  - 실험 데이터로 보정(calibration) 가능한 구조
  - 향후 ML regression / Bayesian optimization으로 업그레이드 가능

사용 예시:
    model = AggregationRiskModel()
    result = model.predict(inputs)
    print(result.summary())

    # 데이터 기반 보정
    model.calibrate_from_experimental_data(exp_df)

    # DOE 그리드 스캔
    grid = model.doe_grid_scan(base_inputs, doe_factors)
"""

from __future__ import annotations

import copy
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Optional

# ---------------------------------------------------------------------------
# 섹션 1: 입력 스키마 (Input Schema)
# ---------------------------------------------------------------------------
# 모든 물리화학 파라미터를 논리적 그룹으로 분리.
# 각 dataclass는 독립적으로 인스턴스화·교체 가능.

BufferType  = Literal["acetate", "histidine", "phosphate", "citrate", "tris"]
ProteinType = Literal["mab", "adc", "peptide", "sc", "intranasal", "microneedle"]


@dataclass
class ProteinProperties:
    """
    단백질의 내재적 물리화학 특성.
    실험 데이터 / in-silico 도구로 측정·예측 가능.
    """

    # ── 필수 파라미터 ─────────────────────────────────────────────────────────
    molecular_weight_kDa: float
    """
    분자량 (kDa). mAb ≈ 150, ADC ≈ 150–160, Peptide < 10.
    측정 방법: SEC-MALS, SDS-PAGE, MS.
    모델 내 역할: Donnan 계산 시 단백질 몰농도 산출에 사용.
    """

    isoelectric_point_pI: float
    """
    등전점 (isoelectric point). 범위: 4.0–11.0.
    측정 방법: cIEF, 2D-PAGE, in-silico (ExPASy pI tool, CamSol).
    모델 내 역할: pH–pI distance 계산 → 전하 기반 colloidal stability 결정.
    [경험적 가정] 단백질 전하는 tanh 함수로 근사. 실제 적정 곡선으로 대체 가능.
    """

    formulation_pH: float
    """
    목표 제형 pH. 범위: 3.5–8.5.
    모델 내 역할: Donnan 보정 후 microenvironment pH 계산의 기준값.
    주의: bulk pH는 단백질 고농도 환경에서 microenvironment pH와 다를 수 있음.
    """

    protein_concentration_mg_per_mL: float
    """
    단백질 농도 (mg/mL). SC 제형 목표: >100 mg/mL.
    모델 내 역할: (1) 분자 혼잡 리스크 기여, (2) Donnan 계수 계산.
    [경험적 가정] 리스크는 30 mg/mL부터 선형 증가, 300 mg/mL에서 포화.
                  실제 crowding은 비선형(second virial 계수 의존).
    """

    hydrophobicity_index: float
    """
    소수성 지수. 범위: 0.0 (친수성) – 1.0 (소수성).
    측정 방법: HIC 크로마토그래피 retention time 정규화, CamSol, GRAVY score.
    모델 내 역할: 소수성 접촉 기반 aggregation 기여도.
    [경험적 가정] 선형 mapping. 실제로는 APR 위치·depth에 따라 비선형.
    """

    aggregation_hotspot_score: float
    """
    서열 기반 aggregation 취약 부위(APR) 점수. 범위: 0.0–1.0.
    측정 방법: Aggrescan3D, CamSol Intrinsic, Zyggregator 등 in-silico 도구.
    모델 내 역할: 서열 내재적 aggregation 경향성 보정.
    [경험적 가정] 외부 도구 출력을 0–1 스케일로 정규화해 입력 필요.
    """

    protein_type: ProteinType = "mab"
    """
    단백질 제형 유형. 각 유형은 multiplier로 전체 리스크를 조정.
    [경험적 가정] Multiplier 값은 문헌 및 일반적 관행 기반. 실험 데이터로 보정 권장.
    """

    # ── 선택 파라미터 (optional) ──────────────────────────────────────────────
    glycosylation_ratio: Optional[float] = None
    """
    당화 비율. 범위: 0.0 (무당화) – 1.0 (완전 당화).
    측정 방법: MS glycopeptide mapping, PNGase F 처리 후 MW 비교.
    모델 내 역할: N-glycan의 steric shielding 효과 → 안정화 기여 (음의 리스크).
    [경험적 가정] 선형 shielding. 실제로는 glycan 구조(high-mannose vs complex)에 따라 다름.
    None 입력 시 효과 무시.
    """

    diffusion_coefficient_um2_s: Optional[float] = None
    """
    확산계수 (μm²/s). DLS로 측정.
    향후 colloidal stability 정밀 계산 시 사용 예정 (현재 미사용).
    """

    second_virial_coefficient_B22: Optional[float] = None
    """
    제2 virial 계수 B22 (mL·mol/g²). DLS kD 또는 SLS로 측정.
    양수: 단백질-용매 상호작용 우세 (안정), 음수: 단백질-단백질 상호작용 우세 (불안정).
    입력 시 colloidal stability 예측 정밀도 향상 (현재 참고용).
    """


@dataclass
class BufferConditions:
    """
    완충 시스템 조건.
    Buffer 선택은 pH 안정성, 단백질과의 상호작용, 동결건조 적합성을 종합 고려.
    """

    buffer_type: BufferType = "histidine"
    """
    완충제 종류.
    - histidine: mAb 표준, 양성이온성, 동결보호 효과, pKa 6.0
    - acetate:   산성 단백질(pI < 6), pKa 4.75
    - citrate:   다가 음이온, 고pI 단백질과 비특이적 결합 위험
    - phosphate: 동결 시 pH 변화 심함 (Na₂HPO₄ 우선 결정화)
    - tris:      pKa 온도 의존성 큼 (−0.03 pH/°C)
    [경험적 가정] 버퍼별 리스크 점수는 문헌 기반 상수. 실험 데이터로 조정 가능.
    """

    buffer_concentration_mM: float = 20.0
    """
    완충제 농도 (mM). 통상 10–50 mM.
    높을수록 pH 완충 능력(buffer capacity) ↑ → Donnan effect 억제.
    단, 고농도 완충제는 이온강도 증가 → 추가 고려 필요.
    """


@dataclass
class IonicEnvironment:
    """
    이온 환경 파라미터.
    이온강도(I)는 Debye 차폐 길이를 결정 → 단백질 간 정전기적 상호작용의 범위.
    I = 0.5 × Σ(ci × zi²), 1가 이온(NaCl, KCl)은 각각 ci × 1² 기여.
    """

    NaCl_mM: float = 0.0
    """NaCl 농도 (mM). SC 제형에서는 tonicifier로도 사용."""

    KCl_mM: float = 0.0
    """KCl 농도 (mM). 세포 독성 완화 목적으로 일부 사용."""

    ionic_strength_mM: Optional[float] = None
    """
    이온강도 직접 입력 (mM). 다가 이온 포함 시 직접 계산하여 입력 권장.
    None이면 NaCl_mM + KCl_mM으로부터 자동 계산 (1가 이온 가정).
    [경험적 가정] 다가 이온(MgCl₂, CaCl₂) 무시. 필요 시 직접 입력.
    """

    def effective_ionic_strength(self) -> float:
        """이온강도 반환. ionic_strength_mM 미입력 시 1가 염으로부터 계산."""
        if self.ionic_strength_mM is not None:
            return max(0.0, self.ionic_strength_mM)
        # 1가 이온: I = 0.5 × (c_NaCl×1² + c_NaCl×1²) = c_NaCl
        return 0.5 * (self.NaCl_mM + self.KCl_mM)


@dataclass
class Surfactants:
    """
    계면활성제 (surfactant) 파라미터.
    기전: 소수성 계면(공기-액체, 용기 표면)에 우선 흡착 →
         단백질의 계면 접촉 및 전개(unfolding) 억제.
    """

    polysorbate20_percent: float = 0.0
    """
    Polysorbate 20 (Tween 20) 농도 (%, w/v). 통상 0.01–0.05%.
    PS20은 상대적으로 작은 소수성 꼬리 → 수용성 단백질에 적합.
    주의: 과산화물 생성 → Met/Trp 산화 위험 (품질 관리 필요).
    """

    polysorbate80_percent: float = 0.0
    """
    Polysorbate 80 (Tween 80) 농도 (%, w/v). 통상 0.01–0.05%.
    PS80은 올레산 기반 긴 꼬리 → 소수성 계면 보호력 우수.
    mAb SC 제형의 표준 surfactant.
    """

    poloxamer188_percent: float = 0.0
    """
    Poloxamer 188 (Pluronic F-68) 농도 (%, w/v). 통상 0.01–0.1%.
    PEO-PPO-PEO 블록 공중합체 → 입체적(steric) 안정화.
    비이온성, 세포 독성 낮음 → 세포 배양 공정에서도 사용.
    """


@dataclass
class SugarStabilizers:
    """
    당류 안정화제 파라미터.
    기전: 우선적 배제(preferential exclusion, Timasheff 이론) →
         단백질 표면에서 용질이 배제되어 native state 선호.
         동결건조 시 수분 대체(water replacement) 효과.
    """

    sucrose_percent: float = 0.0
    """
    수크로오스 농도 (%, w/v). 통상 5–10%.
    가장 널리 사용되는 동결보호제. Tm 상승 효과 우수.
    [경험적 가정] 리스크 감소는 농도에 선형 비례 (포화 있음).
    """

    trehalose_percent: float = 0.0
    """
    트레할로오스 농도 (%, w/v). 통상 5–8%.
    수크로오스 대비 유리전이온도(Tg) 약간 높음 → 동결건조 장기 안정성 우수.
    """

    mannitol_percent: float = 0.0
    """
    만니톨 농도 (%, w/v). 통상 2–5%.
    동결건조 시 bulking agent 역할 (구조 형성).
    단독 사용 시 동결보호 효과 제한적 → sucrose와 병용 권장.
    """

    sorbitol_percent: float = 0.0
    """
    소르비톨 농도 (%, w/v). 통상 2–5%.
    삼투압 조절 목적으로도 사용. 당뇨 환자 주사 제형에 주의.
    """


@dataclass
class AminoAcidStabilizers:
    """
    아미노산 안정화제 파라미터.
    각 아미노산은 서로 다른 메커니즘으로 aggregation 억제.
    """

    arginine_mM: float = 0.0
    """
    Arginine HCl 농도 (mM). 통상 50–200 mM.
    기전 (복합적):
      (1) 소수성 패치에 직접 결합 → 단백질-단백질 접촉 차단 (Shukla & Trout, 2011)
      (2) 양전하 → 전기적 반발 강화
      (3) 점도 감소 효과 (SC 제형에서 이중 이점)
    [경험적 가정] 효과는 0.002/mM 기울기 선형 증가, 150 mM에서 포화.
    """

    glycine_mM: float = 0.0
    """
    글리신 농도 (mM). 통상 50–200 mM.
    기전: kosmotropic → 우선적 배제, 완충 보완.
    고농도에서 점도 증가 부작용 가능.
    [경험적 가정] 효과는 0.0006/mM 기울기 (Arg 대비 약 1/3).
    """

    lysine_mM: float = 0.0
    """
    리신 농도 (mM). 통상 50–100 mM.
    기전: Arg와 유사한 양전하 효과, 단 소수성 패치 결합력은 약함.
    pH > 10.5에서 중성화 (pKa 10.5).
    [경험적 가정] 효과는 0.0014/mM 기울기 (Arg의 약 70%).
    """


@dataclass
class ProcessStress:
    """
    공정 스트레스 파라미터 (선택적, optional).
    각 스트레스는 0.0 (없음) – 1.0 (최대) 범위로 정규화.
    """

    agitation_risk_level: float = 0.0
    """
    교반 스트레스 강도. 0.0–1.0.
    원인: 기포 생성, 계면 전단력 → 단백질 계면 unfolding.
    관련 실험: 진탕(shaking) 48hr, 회전(rotation) 스트레스.
    [경험적 가정] 가중치 0.14. 실제 영향은 용기 headspace, fill volume에 의존.
    """

    pumping_stress_level: float = 0.0
    """
    펌핑 스트레스 강도. 0.0–1.0.
    원인: 연동 펌프 전단력, 필터 통과 시 압력 → 구조 스트레스.
    관련 실험: peristaltic pump 통과 횟수·유속 기반 평가.
    [경험적 가정] 가중치 0.10.
    """

    thermal_stress_level: float = 0.0
    """
    열 스트레스 강도. 0.0–1.0.
    원인: 고온 노출, 반복 온도 사이클 → 부분 unfolding.
    관련 실험: 40°C/2주 가속 안정성, 동결/해동(F/T) 사이클.
    [경험적 가정] 가중치 0.16 (열은 Arrhenius 지수적 영향이나 선형 근사).
    """


@dataclass
class FormulationInputs:
    """
    모델 최상위 입력 컨테이너.
    모든 서브 dataclass를 조합. 누락된 그룹은 기본값 사용.

    사용 예시:
        inputs = FormulationInputs(
            protein=ProteinProperties(mw=148, pI=8.4, pH=5.8, conc=175, hyd=0.52, hotspot=0.38),
            buffer=BufferConditions(buffer_type="histidine", buffer_concentration_mM=20),
            sugars=SugarStabilizers(sucrose_percent=9.0),
        )
    """
    protein: ProteinProperties
    buffer: BufferConditions       = field(default_factory=BufferConditions)
    ions:   IonicEnvironment       = field(default_factory=IonicEnvironment)
    surfactants: Surfactants       = field(default_factory=Surfactants)
    sugars: SugarStabilizers       = field(default_factory=SugarStabilizers)
    amino_acids: AminoAcidStabilizers = field(default_factory=AminoAcidStabilizers)
    stress: ProcessStress          = field(default_factory=ProcessStress)

    def to_flat_dict(self) -> dict[str, Any]:
        """모든 파라미터를 단일 flat dict로 직렬화 (ML 학습용)."""
        d = {}
        for sub in [self.protein, self.buffer, self.ions,
                    self.surfactants, self.sugars, self.amino_acids, self.stress]:
            for k, v in asdict(sub).items():
                d[k] = v if v is not None else float("nan")
        return d


# ---------------------------------------------------------------------------
# 섹션 2: Donnan Effect 모듈
# ---------------------------------------------------------------------------

@dataclass
class DonnanResult:
    """Donnan 효과 계산 결과."""
    bulk_pH: float
    micro_pH: float
    """Donnan 보정된 단백질 microenvironment pH."""
    delta_pH: float
    """pH 이동량 (양수 = 알칼리 방향, 음수 = 산성 방향)."""
    net_charge_estimate: float
    """단백질 분자당 추정 순전하 [전자 단위, e]."""
    donnan_coefficient: float
    """무차원 Donnan 계수 K_D. 클수록 microenvironment 편차 큼."""
    local_ionic_strength_mM: float
    """Donnan 재분배 후 국소 이온강도 추정값 (mM)."""


def compute_donnan_effect(
    ph: float,
    pI: float,
    concentration_mg_mL: float,
    molecular_weight_kDa: float,
    ionic_strength_mM: float,
    buffer_concentration_mM: float,
) -> DonnanResult:
    """
    고농도 단백질 환경에서 Donnan 효과에 의한 microenvironment pH 변화 추정.

    물리적 배경:
        단백질 용액 내 단백질 분자는 이동 불가 → Donnan 평형 발생.
        단백질 순전하(Z)에 의해 반대 이온이 국소 축적 → 국소 pH 변화.
        - pH < pI (순양전하) → Cl⁻ 배제 → 국소 [H⁺] 증가 → pH 감소
        - pH > pI (순음전하) → H⁺ 배제 → 국소 pH 증가

    적용된 수식:
        net_charge Z ≈ −15 × tanh(1.2 × (pH − pI))
            [경험적 가정] 최대 전하 ±15, 기울기 1.2는 일반적 mAb 근사.
                         실제 적정 곡선(potentiometric titration)으로 교체 가능.

        단백질 몰농도 C_p = conc_mg_mL / (MW_kDa × 1000)  [mol/L]

        전하 몰농도 = |Z| × C_p

        이온강도 합계 I_total = (IS + buffer_conc) / 1000  [mol/L]

        Donnan 계수 K_D = |Z| × C_p / (2 × I_total)
            [경험적 가정] K_D ≤ 3.0으로 물리적 상한 적용.

        ΔpH = sign × 0.18 × ln(1 + 2 × K_D)
            [경험적 가정] 비례 계수 0.18은 문헌 기반 추정값.
                         실측 NMR pH 또는 pH 민감 형광 프로브로 보정 가능.

        국소 IS = IS_bulk × (1 + 0.15 × K_D)
            [경험적 가정] 이온 재분배에 의한 국소 IS 증가를 선형 근사.

    독립 모듈: 이 함수는 AggregationRiskModel 없이도 단독 호출 가능.
    """
    # 순전하 추정 (tanh 근사)
    net_charge = -15.0 * math.tanh(1.2 * (ph - pI))

    # 단백질 몰농도
    protein_molarity = concentration_mg_mL / (molecular_weight_kDa * 1000.0)

    # 전하 유발 이온 불균형
    charge_conc = abs(net_charge) * protein_molarity

    # 전체 이온강도 (버퍼 기여 포함)
    I_total = (ionic_strength_mM + buffer_concentration_mM) / 1000.0

    # Donnan 계수 (물리적 상한 3.0)
    donnan_K = charge_conc / (2.0 * I_total + 1e-9)
    donnan_K = min(donnan_K, 3.0)

    # pH 이동 방향 결정
    sign = -1.0 if net_charge > 0 else 1.0
    delta_pH = sign * 0.18 * math.log1p(2.0 * donnan_K)
    micro_pH = ph + delta_pH

    # 국소 이온강도
    local_IS = ionic_strength_mM * (1.0 + 0.15 * donnan_K)

    return DonnanResult(
        bulk_pH=ph,
        micro_pH=round(micro_pH, 3),
        delta_pH=round(delta_pH, 3),
        net_charge_estimate=round(net_charge, 2),
        donnan_coefficient=round(donnan_K, 3),
        local_ionic_strength_mM=round(local_IS, 1),
    )


# ---------------------------------------------------------------------------
# 섹션 3: 리스크 팩터 모듈 (각 함수 독립 교체 가능)
# ---------------------------------------------------------------------------
# 각 함수의 반환값: (risk_score: float, explanation: str)
# risk_score: 0.0 (리스크 없음) – 1.0 (최대 리스크)
# 안정화 기여는 음수 반환 가능 (glycosylation).
#
# [보정 방법] 각 함수의 계수를 dict(coef_*)로 외부화하여
#            실험 데이터 기반 회귀(regression)로 조정 가능.

# ── 3-1. pI 근접도 리스크 ─────────────────────────────────────────────────

# [보정 가능한 계수]
PI_RISK_COEF = {
    "max_distance": 3.0,   # 이 거리 이상에서 리스크 ≈ 0
    "power":        0.8,   # 감소 곡선의 지수 (1.0 = 선형, <1 = 볼록)
}

def compute_pi_proximity_risk(
    micro_pH: float,
    pI: float,
    coef: dict | None = None,
) -> tuple[float, str]:
    """
    제형 pH(Donnan 보정 후)와 등전점(pI)의 근접도에 따른 aggregation 리스크.

    물리적 배경:
        pH ≈ pI에서 단백질 순전하 ≈ 0 → 정전기적 반발(electrostatic repulsion) 소멸
        → colloidal stability 급격히 감소 → 소수성 접촉 확률 증가 → 응집.
        이는 DLVO 이론에서 'primary maximum' 소멸에 해당.

    수식: risk = max(0, 1 − (|ΔpH| / max_distance)^power)

    [경험적 가정]
        - max_distance = 3.0 pH unit: 이 이상에서 pI 영향 무시 (단순화).
        - power = 0.8: 약간 볼록 곡선 (pI 근방에서 급격한 상승 반영).
        - 실제로는 단백질 전하 분포, 쌍극자 모멘트에 의존.

    개선 방향:
        - 실측 zeta potential vs pH 곡선으로 대체.
        - B22(제2 virial 계수) vs pH 데이터로 직접 보정.
    """
    c = coef or PI_RISK_COEF
    distance = abs(micro_pH - pI)
    score = max(0.0, 1.0 - (distance / c["max_distance"]) ** c["power"])
    level = (
        "CRITICAL — within 0.5 unit of pI" if distance < 0.5 else
        "HIGH"     if distance < 1.2 else
        "MODERATE" if distance < 2.0 else
        "LOW"
    )
    return round(score, 4), f"micro_pH={micro_pH:.2f}, pI={pI:.1f}, |ΔpH|={distance:.2f} → {level}"


# ── 3-2. 농도 리스크 (혼잡 효과) ────────────────────────────────────────────

CONC_RISK_COEF = {
    "onset_mg_mL":    30.0,   # 리스크 시작 농도 (mg/mL)
    "saturation_range": 270.0, # 포화 범위 (onset ~ onset+range = 최대)
}

def compute_concentration_risk(
    conc_mg_mL: float,
    coef: dict | None = None,
) -> tuple[float, str]:
    """
    단백질 농도에 의한 분자 혼잡(molecular crowding) 리스크.

    물리적 배경:
        고농도에서 단백질-단백질 만남 빈도 증가 (확산 제한 반응).
        >100 mg/mL에서 effective diffusion coefficient 감소.
        excluded volume 효과로 실질 농도 > 명목 농도.

    [경험적 가정]
        - 30 mg/mL 이하에서 리스크 ≈ 0 (명목상).
        - 300 mg/mL에서 포화 리스크 1.0.
        - 실제로는 B22, osmotic second virial coefficient 의존.

    개선 방향:
        - DLS로 측정한 kD (concentration-dependent diffusion coefficient) 활용.
        - osmotic pressure 측정값으로 crowding 계수 보정.
    """
    c = coef or CONC_RISK_COEF
    score = max(0.0, min(1.0, (conc_mg_mL - c["onset_mg_mL"]) / c["saturation_range"]))
    level = (
        "CRITICAL" if conc_mg_mL > 200 else
        "HIGH"     if conc_mg_mL > 120 else
        "MODERATE" if conc_mg_mL > 70  else
        "LOW"
    )
    return round(score, 4), f"{conc_mg_mL} mg/mL → {level}"


# ── 3-3. 소수성 리스크 ───────────────────────────────────────────────────────

def compute_hydrophobicity_risk(hyd_index: float) -> tuple[float, str]:
    """
    단백질 내재적 소수성에 의한 aggregation 리스크.

    물리적 배경:
        소수성 패치(hydrophobic patch)의 용매 노출 시 intermolecular 접촉 발생.
        HIC 크로마토그래피: 소수성 표면과 결합 → retention time 길수록 소수성 큼.
        GRAVY (Grand Average of Hydropathy) 또는 CamSol 점수로 정량화.

    [경험적 가정]
        - index 0–1 선형 매핑. 실제로는 비선형 (패치 분포, 깊이 의존).
        - index > 0.6: HIC 고 retention mAb → aggregation 빈번.

    개선 방향:
        - HIC relative retention (RR) 값을 직접 입력.
        - Molecular dynamics 시뮬레이션 기반 SASA (solvent-accessible surface area) 활용.
    """
    score = min(1.0, max(0.0, hyd_index))
    level = "HIGH" if score > 0.6 else "MODERATE" if score > 0.35 else "LOW"
    return round(score, 4), f"hydrophobicity_index={hyd_index:.2f} → {level}"


# ── 3-4. 서열 기반 hotspot 리스크 ────────────────────────────────────────────

def compute_hotspot_risk(hotspot_score: float) -> tuple[float, str]:
    """
    서열 기반 aggregation 취약 부위(APR) 리스크.

    물리적 배경:
        특정 서열 모티프(예: 연속 소수성 아미노산, 베타-시트 형성 경향 서열)는
        구조 교란 시 β-aggregation nucleus로 기능.
        Aggrescan3D, CamSol Intrinsic, Zyggregator로 예측.

    [경험적 가정]
        - 외부 도구 출력의 0–1 정규화 방법에 따라 절대값 의미 달라짐.
        - 도구별 보정 계수 적용 권장.

    개선 방향:
        - 여러 in-silico 도구 출력의 앙상블 점수 사용.
        - 실측 HDXMS (hydrogen-deuterium exchange MS) 데이터로 검증.
    """
    score = min(1.0, max(0.0, hotspot_score))
    level = "HIGH" if score > 0.6 else "MODERATE" if score > 0.3 else "LOW"
    return round(score, 4), f"APR hotspot score={hotspot_score:.2f} → {level}"


# ── 3-5. 이온강도 리스크 ─────────────────────────────────────────────────────

IS_RISK_COEF = {
    "low_threshold_mM":  30.0,   # 이하: 저이온강도 불안정 구간
    "opt_low_mM":        50.0,   # 최적 구간 하한
    "opt_high_mM":      150.0,   # 최적 구간 상한
    "high_onset_mM":    200.0,   # 고이온강도 리스크 시작
    "low_risk_base":      0.35,
    "high_risk_slope":    0.001,
}

def compute_ionic_strength_risk(IS_mM: float, coef: dict | None = None) -> tuple[float, str]:
    """
    이온강도(ionic strength)에 따른 colloidal stability 리스크.

    물리적 배경:
        Debye-Hückel 이론: Debye 길이 κ⁻¹ = 0.304 / √I(M)  [nm, 25°C, 1:1 전해질]
        이온강도 증가 → κ⁻¹ 감소 → 정전기 반발 범위 감소.

        ▸ 저이온강도 (<50 mM):
          κ⁻¹ 길어짐 → 먼 거리 반발 유지.
          그러나 극저이온강도에서는 이온 기여가 없어 오히려
          단백질 쌍극자 불균형, 표면 흡착 등 비특이적 불안정화 가능.

        ▸ 적정이온강도 (50–150 mM):
          Cl⁻, Na⁺ 이온이 단백질 표면 전하를 적절히 차폐.
          colloidal stability 최적화 구간.

        ▸ 고이온강도 (>200 mM):
          과도한 Debye 차폐 → 정전기 반발 소멸.
          Hofmeister 효과: 소수성 hydration shell 약화 → 소수성 노출 증가.
          SO₄²⁻, (NH₄)₂SO₄ 등 음이온 kosmotrope 효과.

    [경험적 가정]
        - 각 구간의 리스크 값은 문헌 및 일반적 관행 기반.
        - 이온 특이성(Hofmeister series) 무시 (단순화).

    개선 방향:
        - 이온 종류별 Hofmeister 가중치 적용.
        - DLS로 측정한 kD vs IS 곡선으로 보정.
    """
    c = coef or IS_RISK_COEF
    if IS_mM < c["low_threshold_mM"]:
        score = c["low_risk_base"]
        note = f"very low IS ({IS_mM:.0f} mM) — electrostatic instability, κ⁻¹ too long"
    elif IS_mM <= c["opt_high_mM"]:
        # 최적 구간: 점진적 감소
        score = max(0.02, 0.25 - (IS_mM - c["opt_low_mM"]) / 500.0)
        note = f"IS={IS_mM:.0f} mM — optimal range (Debye length balanced)"
    else:
        # 고이온강도: 완만히 증가
        score = min(0.55, 0.10 + (IS_mM - c["opt_high_mM"]) * c["high_risk_slope"])
        note = f"IS={IS_mM:.0f} mM — elevated, hydrophobic promotion risk"
    return round(score, 4), note


# ── 3-6. 버퍼 타입 리스크 ────────────────────────────────────────────────────

# [보정 가능] 각 버퍼의 기본 리스크 점수를 실험 데이터로 조정 가능
BUFFER_RISK_SCORES: dict[str, tuple[float, str]] = {
    "histidine": (0.05,
        "preferred — zwitterionic, cryo-protective, pKa 6.0, widely validated for mAb"),
    "acetate":   (0.10,
        "mild risk — pKa temperature-stable (4.75), good for acidic proteins (pI<6)"),
    "citrate":   (0.18,
        "chelation risk — trivalent anion can non-specifically bind basic patches (pI>7)"),
    "phosphate": (0.14,
        "freeze risk — Na₂HPO₄ crystallizes on freezing → dramatic pH drop"),
    "tris":      (0.13,
        "temperature-sensitive pKa (−0.03/°C) — pH shifts on freeze-thaw"),
}

def compute_buffer_risk(buffer_type: str) -> tuple[float, str]:
    """
    완충제 종류에 따른 고유 aggregation 리스크.

    [경험적 가정]
        - 점수는 문헌 기반 상대적 위험도. 절대값보다 상대 비교에 의미.
        - BUFFER_RISK_SCORES dict 수정으로 즉시 보정 가능.

    개선 방향:
        - 단백질별 버퍼 상호작용 실험 (DLS, SEC-MALS, 단백질 특이적) 데이터 반영.
    """
    score, note = BUFFER_RISK_SCORES.get(buffer_type, (0.12, f"unknown buffer type: {buffer_type}"))
    return score, f"{buffer_type}: {note}"


# ── 3-7. 당화 안정화 효과 ────────────────────────────────────────────────────

GLYCOSYLATION_COEF = {
    "max_protection": 0.22,  # 완전 당화 시 최대 안정화 기여 (음수 리스크)
}

def compute_glycosylation_effect(
    glycosylation_ratio: Optional[float],
    coef: dict | None = None,
) -> tuple[float, str]:
    """
    N-/O-당화에 의한 단백질 안정화 효과.

    물리적 배경:
        N-glycan은 단백질 표면의 소수성 패치를 steric shielding.
        수화 shell(hydration shell) 강화 → 소수성 aggregation 억제.
        고만노오스형(high-mannose) < 복합형(complex) N-glycan 안정화 효과 차이 있으나 단순화.

    반환값: 음수 (안정화 기여).
    None 입력 시 효과 무시 (0.0 반환).

    [경험적 가정]
        - 선형 mapping. 실제로는 glycan 위치, 유형, 크기에 의존.
        - 최대 보정값 0.22는 문헌 기반 추정.

    개선 방향:
        - 탈당화(PNGase F 처리) 전후 SEC-MALS Tm 비교로 보정.
    """
    if glycosylation_ratio is None:
        return 0.0, "glycosylation not specified — effect ignored"
    c = coef or GLYCOSYLATION_COEF
    protection = min(c["max_protection"], glycosylation_ratio * c["max_protection"])
    return round(-protection, 4), f"glycosylation={glycosylation_ratio:.2f} → −{protection:.3f} stabilisation"


# ── 3-8. 공정 스트레스 리스크 ────────────────────────────────────────────────

STRESS_RISK_COEF = {
    "agitation_weight": 0.14,
    "pumping_weight":   0.10,
    "thermal_weight":   0.16,
    "max_total":        0.35,
}

def compute_process_stress_risk(
    ps: ProcessStress,
    coef: dict | None = None,
) -> tuple[float, str]:
    """
    공정 스트레스에 의한 기계적·열적 aggregation 유발 리스크.

    물리적 배경:
        ▸ 교반(agitation): 기포 형성 → 공기-액체 계면에서 단백질 unfolding.
          표면 전단력(shear stress) → β-sheet 전환 촉진.
        ▸ 펌핑(pumping): 연동 펌프 벽 접촉 전단. 필터 통과 압력 스파이크.
        ▸ 열 스트레스(thermal): Arrhenius 지수 의존. Tm 이하 부분 unfolding.
          반복 동결-해동 시 계면 농축 효과.

    [경험적 가정]
        - 각 가중치는 상대적 스트레스 기여도 추정값.
        - 선형 합산. 실제로는 시너지 효과 가능 (예: 고온 + 교반).
        - max_total = 0.35로 포화.

    개선 방향:
        - 각 스트레스 실험 결과(aggregation % after stress)로 개별 가중치 보정.
        - agitation의 경우 rpm × time × headspace 비율로 정규화.
    """
    c = coef or STRESS_RISK_COEF
    agit  = ps.agitation_risk_level * c["agitation_weight"]
    pump  = ps.pumping_stress_level * c["pumping_weight"]
    therm = ps.thermal_stress_level * c["thermal_weight"]
    total = min(c["max_total"], agit + pump + therm)
    detail = (
        f"agitation={ps.agitation_risk_level:.2f}(+{agit:.3f}), "
        f"pumping={ps.pumping_stress_level:.2f}(+{pump:.3f}), "
        f"thermal={ps.thermal_stress_level:.2f}(+{therm:.3f})"
    )
    return round(total, 4), detail


# ---------------------------------------------------------------------------
# 섹션 4: Excipient 보호 효과 모듈
# ---------------------------------------------------------------------------

# ── 4-1. 계면활성제 보호 ─────────────────────────────────────────────────────

SURFACTANT_COEF = {
    "ps20_slope":  2.5, "ps20_max":  0.22,
    "ps80_slope":  2.2, "ps80_max":  0.22,
    "p188_slope":  0.7, "p188_max":  0.14,
    "total_max":   0.28,  # CMC 포화에 의한 additive 상한
}

def compute_surfactant_protection(
    s: Surfactants,
    coef: dict | None = None,
) -> tuple[float, str]:
    """
    계면활성제에 의한 단백질 계면 보호 효과.

    물리적 배경 (Langmuir 흡착 기반):
        계면활성제는 CMC 이하에서 소수성 계면에 단분자층 형성.
        단백질의 계면 접근을 경쟁적으로 차단.
        CMC 초과 시: 미셀 형성 → free surfactant 농도 일정 → 추가 효과 없음.

        PS20/PS80: 소수성 계면(공기-액체) 흡착 우수.
        P188: PEO block의 친수성 steric layer → 단백질 계면 접근 입체 장벽.

    [경험적 가정]
        - 선형 slope + 최대값 포화 (단순 Langmuir 근사).
        - CMC 이하에서 선형, CMC 이상에서 포화 (실제 CMC 고려 필요).
        - PS20 CMC ≈ 0.002%, PS80 CMC ≈ 0.001% (참고용).
        - total_max = 0.28: 병용 시 추가 상승 제한 (계면 포화).

    개선 방향:
        - Dynamic surface tension 측정값으로 계면 흡착 효율 보정.
        - PS 품질(산화도, 지방산 분포)을 변수로 추가.
    """
    c = coef or SURFACTANT_COEF
    ps20  = min(c["ps20_max"], s.polysorbate20_percent  * c["ps20_slope"])
    ps80  = min(c["ps80_max"], s.polysorbate80_percent  * c["ps80_slope"])
    p188  = min(c["p188_max"], s.poloxamer188_percent   * c["p188_slope"])
    total = min(c["total_max"], ps20 + ps80 + p188)
    detail = (
        f"PS20={s.polysorbate20_percent:.3f}%(−{ps20:.3f}), "
        f"PS80={s.polysorbate80_percent:.3f}%(−{ps80:.3f}), "
        f"P188={s.poloxamer188_percent:.3f}%(−{p188:.3f})"
    )
    return round(total, 4), detail


# ── 4-2. 당류 안정화제 보호 ──────────────────────────────────────────────────

SUGAR_COEF = {
    "sucrose_slope":   0.040, "sucrose_max":   0.30,
    "trehalose_slope": 0.042, "trehalose_max": 0.32,
    "mannitol_slope":  0.018, "mannitol_max":  0.12,
    "sorbitol_slope":  0.015, "sorbitol_max":  0.10,
    "total_max":       0.38,
}

def compute_sugar_protection(
    sg: SugarStabilizers,
    coef: dict | None = None,
) -> tuple[float, str]:
    """
    당류 안정화제의 우선적 배제 기반 보호 효과.

    물리적 배경 (Timasheff preferential exclusion):
        다당류가 단백질 표면 수화층에서 배제 →
        단백질 주변의 화학 포텐셜 증가 → native state 열역학적 선호.
        ΔGstab ≈ kT × ln(a_water / a_cosolute) 로 표현 가능.

        Sucrose/Trehalose: 이당류 → 큰 배제 부피 → 강한 stabilization.
        Mannitol/Sorbitol: 단당류 폴리올 → 배제 효과 약함 + bulking 역할.

    [경험적 가정]
        - slope 값은 sucrose 문헌 kJ/mol/%당 안정화 에너지로부터 역산.
        - trehalose > sucrose: Tg ≈ 117°C vs 75°C 반영 (동결건조).
        - total_max = 0.38: 과도한 삼투압 증가 전 포화.

    개선 방향:
        - DSC(differential scanning calorimetry)로 Tm 변화 측정 → slope 보정.
        - 동결건조 샘플은 Tg 측정값 반영.
    """
    c = coef or SUGAR_COEF
    su = min(c["sucrose_max"],   sg.sucrose_percent   * c["sucrose_slope"])
    tr = min(c["trehalose_max"], sg.trehalose_percent * c["trehalose_slope"])
    ma = min(c["mannitol_max"],  sg.mannitol_percent  * c["mannitol_slope"])
    so = min(c["sorbitol_max"],  sg.sorbitol_percent  * c["sorbitol_slope"])
    total = min(c["total_max"], su + tr + ma + so)
    detail = (
        f"sucrose={sg.sucrose_percent}%(−{su:.3f}), "
        f"trehalose={sg.trehalose_percent}%(−{tr:.3f}), "
        f"mannitol={sg.mannitol_percent}%(−{ma:.3f}), "
        f"sorbitol={sg.sorbitol_percent}%(−{so:.3f})"
    )
    return round(total, 4), detail


# ── 4-3. 아미노산 안정화제 보호 ──────────────────────────────────────────────

AA_COEF = {
    "arg_slope": 0.0020, "arg_max": 0.28,
    "gly_slope": 0.0006, "gly_max": 0.12,
    "lys_slope": 0.0014, "lys_max": 0.16,
    "total_max": 0.35,
}

def compute_amino_acid_protection(
    aa: AminoAcidStabilizers,
    coef: dict | None = None,
) -> tuple[float, str]:
    """
    아미노산 안정화제의 aggregation 억제 효과.

    물리적 배경:
        ▸ Arginine (Shukla & Trout 2011):
          가장 효과적인 aggregation 억제제.
          (1) 소수성 패치에 직접 결합 → 단백질-단백질 계면 차단.
          (2) 과잉 양전하 → 단백질 간 정전기 반발 증가.
          (3) 점도 감소 → SC 제형에서 이중 이점.
          (4) 계면활성제 없이도 계면 활성 (mild surfactant effect).

        ▸ Glycine:
          Kosmotropic amino acid → 우선적 배제 기전.
          물에 비해 단백질 표면을 강하게 수화 → 안정화.
          고농도(>200 mM)에서 점도 증가 주의.

        ▸ Lysine:
          Arginine과 유사한 양전하 기여.
          소수성 패치 결합력은 Arg보다 약함 (guanidinium group 없음).
          pKa 10.5 → 생리 pH에서 양전하 유지.

    [경험적 가정]
        - slope 값: Arg 효능 문헌 및 일반적 제형 경험 기반.
        - Gly slope ≈ Arg slope / 3.3, Lys slope ≈ Arg slope × 0.7.
        - 세 성분의 메커니즘이 다르므로 독립 additive 근사 (상호작용 무시).

    개선 방향:
        - HP-SEC %HMW(고분자량 응집체)를 Y, 아미노산 농도를 X로 한 dose-response 피팅.
        - Arg × Sucrose 시너지 효과 (interaction term) DOE로 정량화.
    """
    c = coef or AA_COEF
    arg = min(c["arg_max"], aa.arginine_mM * c["arg_slope"])
    gly = min(c["gly_max"], aa.glycine_mM  * c["gly_slope"])
    lys = min(c["lys_max"], aa.lysine_mM   * c["lys_slope"])
    total = min(c["total_max"], arg + gly + lys)
    detail = (
        f"Arg={aa.arginine_mM}mM(−{arg:.3f}), "
        f"Gly={aa.glycine_mM}mM(−{gly:.3f}), "
        f"Lys={aa.lysine_mM}mM(−{lys:.3f})"
    )
    return round(total, 4), detail


# ---------------------------------------------------------------------------
# 섹션 5: 단백질 타입 보정 계수
# ---------------------------------------------------------------------------

# [경험적 가정] 각 계수는 제형 유형별 평균적 위험 증배 요인.
# 실험 데이터 축적 시 단백질 타입별 독립 모델로 분리 권장.
PROTEIN_TYPE_MODIFIERS: dict[str, tuple[float, str]] = {
    "mab":        (1.00, "Reference modality — standard IgG1/IgG4"),
    "adc":        (1.18, "DAR heterogeneity → hydrophobicity variance increased"),
    "peptide":    (0.82, "Smaller MW → fewer intermolecular contacts, faster diffusion"),
    "sc":         (1.22, "Highest conc regime; viscosity + crowding effect amplified"),
    "intranasal": (1.06, "Mucosal interface stress + nasal pH variability"),
    "microneedle":(0.97, "Semi-solid state; aggregation kinetics slower than liquid"),
}


# ---------------------------------------------------------------------------
# 섹션 6: 가중치 설정 (Weight Configuration)
# ---------------------------------------------------------------------------
# [보정 가능] 이 dict를 실험 데이터 기반 회귀로 최적화할 수 있음.
# 가중치 합 = 1.0 (normalised). 음수 가중치 없음 (안정화는 별도 차감).
# 단, glycosylation은 음수 점수를 양수 가중치와 곱해 안정화 기여.

DEFAULT_WEIGHTS: dict[str, float] = {
    "pi_proximity":   0.26,  # pI 근접 → 정전기 안정성 소실 (가장 중요)
    "concentration":  0.20,  # 농도 → 분자 혼잡, 만남 빈도
    "hydrophobicity": 0.15,  # 소수성 → 소수성 접촉 aggregation
    "hotspot":        0.10,  # 서열 APR → 핵 형성 경향
    "ionic_strength": 0.08,  # 이온강도 → Debye 차폐 / Hofmeister
    "buffer":         0.05,  # 버퍼 → pH 안정성, 단백질 상호작용
    "glycosylation":  0.04,  # 당화 → steric shielding (음수 기여)
    "process_stress": 0.12,  # 공정 → 기계적·열적 변성
}
# 합계: 1.00

EXCIPIENT_PROTECTION_WEIGHT: float = 0.45
"""
Excipient 보호 효과의 전체 리스크에 대한 기여 비율.
[경험적 가정] 최상의 excipient 조합이 raw risk를 최대 45% 감소.
실험 데이터로 보정 권장.
"""


# ---------------------------------------------------------------------------
# 섹션 7: 결과 컨테이너
# ---------------------------------------------------------------------------

@dataclass
class FactorScore:
    """개별 리스크 팩터 점수 및 해석."""
    name: str
    raw_score: float        # 모듈 반환 원래 점수 (0–1, 안정화는 음수)
    weight: float           # 가중치
    weighted_score: float   # raw_score × weight
    explanation: str        # 물리화학적 해석 텍스트


@dataclass
class PredictionResult:
    """모델 예측 최종 결과 컨테이너."""
    aggregation_risk_score: float
    """최종 aggregation 리스크 점수. 0.0 (완전 안정) – 1.0 (최대 위험)."""

    risk_level: str
    """LOW / MODERATE / HIGH / CRITICAL."""

    colloidal_stability_estimate: float
    """colloidal stability 대리 지표 (1 − risk). B22, kD 부호 참고용."""

    donnan: DonnanResult
    """Donnan 효과 상세 결과."""

    factor_scores: list[FactorScore]
    """개별 리스크 팩터별 해석 결과."""

    excipient_protection_detail: dict[str, tuple[float, str]]
    """excipient 그룹별 보호 효과 상세."""

    excipient_protection_total: float
    """전체 excipient 보호 효과 (raw risk에서 차감된 값)."""

    protein_type_modifier: float
    """단백질 유형 보정 계수."""

    protein_type_note: str
    """단백질 유형 보정 근거."""

    recommendations: list[str]
    """제형 최적화 권고사항."""

    input_flat_dict: dict[str, Any] = field(default_factory=dict)
    """ML 학습용 flat 입력 dict (to_flat_dict() 출력)."""

    def summary(self, compact: bool = False) -> str:
        """사람이 읽기 쉬운 결과 요약 문자열 출력."""
        fill  = int(self.aggregation_risk_score * 20)
        bar   = "█" * fill + "░" * (20 - fill)
        level_symbol = {"LOW":"✓", "MODERATE":"△", "HIGH":"▲", "CRITICAL":"✕"}.get(self.risk_level, "?")

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════╗",
            "║    AggPredict v2.0 — Aggregation Risk Prediction Report          ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            f"  Risk Score   : {self.aggregation_risk_score:.4f}  [{bar}]",
            f"  Risk Level   : {level_symbol} {self.risk_level}",
            f"  Colloidal Stability (proxy) : {self.colloidal_stability_estimate:.4f}",
            "",
            "── Donnan Effect ─────────────────────────────────────────────────",
            f"  Bulk pH               : {self.donnan.bulk_pH:.2f}",
            f"  Microenvironment pH   : {self.donnan.micro_pH:.3f}  (ΔpH = {self.donnan.delta_pH:+.3f})",
            f"  Net charge estimate   : {self.donnan.net_charge_estimate:+.1f} e",
            f"  Donnan coefficient    : {self.donnan.donnan_coefficient:.3f}",
            f"  Local ionic strength  : {self.donnan.local_ionic_strength_mM:.1f} mM",
        ]

        if not compact:
            lines += [
                "",
                "── Risk Factor Breakdown ─────────────────────────────────────────",
            ]
            for f in self.factor_scores:
                sign = "+" if f.raw_score >= 0 else ""
                lines.append(
                    f"  {f.name:<22}  raw={sign}{f.raw_score:.4f}  "
                    f"w={f.weight:.2f}  Δrisk={f.weighted_score:+.4f}"
                )
                lines.append(f"    └ {f.explanation}")

            lines += [
                "",
                "── Excipient Protection ──────────────────────────────────────────",
            ]
            for grp, (val, note) in self.excipient_protection_detail.items():
                lines.append(f"  {grp:<20}  protection=−{val:.4f}  {note}")
            lines.append(f"  {'TOTAL (applied)':<20}  −{self.excipient_protection_total:.4f}")

        lines += [
            "",
            f"  Protein-type modifier : ×{self.protein_type_modifier:.2f}  ({self.protein_type_note})",
            "",
            "── Recommendations ───────────────────────────────────────────────",
        ]
        for r in self.recommendations:
            lines.append(f"  • {r}")
        lines.append("")
        return "\n".join(lines)

    def to_record(self) -> dict[str, Any]:
        """
        실험 데이터 수집 DataFrame 행으로 직렬화.
        HT 스크리닝 결과와 병합하여 ML 학습에 사용.
        """
        record = dict(self.input_flat_dict)
        record["predicted_risk"]    = self.aggregation_risk_score
        record["risk_level"]        = self.risk_level
        record["donnan_delta_pH"]   = self.donnan.delta_pH
        record["donnan_micro_pH"]   = self.donnan.micro_pH
        record["donnan_K"]          = self.donnan.donnan_coefficient
        record["colloidal_stability"] = self.colloidal_stability_estimate
        record["excipient_protection"] = self.excipient_protection_total
        # 측정값은 나중에 병합 (실험 후)
        record["measured_HMW_pct"]  = None  # HP-SEC %HMW (실험 후 입력)
        record["measured_Tm_C"]     = None  # DSF/nanoDSF Tm (실험 후 입력)
        record["measured_B22"]      = None  # SLS B22 (실험 후 입력)
        record["measured_kD"]       = None  # DLS kD (실험 후 입력)
        return record


# ---------------------------------------------------------------------------
# 섹션 8: 메인 모델 클래스
# ---------------------------------------------------------------------------

class AggregationRiskModel:
    """
    고농도 단백질 제형 aggregation 리스크 예측 모델.

    설계 원칙:
        1. 각 물리화학 모듈은 독립적으로 수정·교체 가능.
        2. 모든 경험적 가정은 명시적으로 문서화.
        3. 가중치와 계수는 실험 데이터로 보정 가능.
        4. ML regression / Bayesian optimization 업그레이드 경로 제공.

    확장 방법:
        • 새 리스크 팩터 추가:
            (1) 새 함수 compute_xxx_risk() 작성
            (2) DEFAULT_WEIGHTS에 키와 가중치 추가
            (3) predict() 내 modules dict에 등록
        • 가중치 보정:
            model.calibrate_from_experimental_data(df)
        • 계수 교체:
            model.coef_overrides = {"pi_proximity": {"max_distance": 2.5}}
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        coef_overrides: dict[str, dict] | None = None,
    ):
        """
        Args:
            weights: 리스크 팩터별 가중치 dict. None이면 DEFAULT_WEIGHTS 사용.
            coef_overrides: 각 모듈 계수 오버라이드. 예: {"buffer": {...}}.
        """
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.coef_overrides = coef_overrides or {}

        # 향후 ML 모델 대체 시 여기서 호출할 함수 등록 (기본: mechanistic)
        self._module_overrides: dict[str, Callable] = {}

    # ── 8-1. 예측 ─────────────────────────────────────────────────────────────

    def predict(self, inputs: FormulationInputs) -> PredictionResult:
        """
        메인 예측 함수.

        실행 순서:
            1. Donnan effect 계산 (microenvironment pH 보정)
            2. 각 리스크 팩터 모듈 실행
            3. 가중 합산 → raw_risk
            4. Excipient 보호 효과 차감
            5. 단백질 타입 보정
            6. 최종 리스크 클리핑 [0.01, 0.99]
            7. 권고사항 생성
        """
        p  = inputs.protein
        b  = inputs.buffer
        io = inputs.ions
        IS = io.effective_ionic_strength()

        # ── Step 1: Donnan 효과 ──────────────────────────────────────────────
        donnan = compute_donnan_effect(
            ph=p.formulation_pH,
            pI=p.isoelectric_point_pI,
            concentration_mg_mL=p.protein_concentration_mg_per_mL,
            molecular_weight_kDa=p.molecular_weight_kDa,
            ionic_strength_mM=IS,
            buffer_concentration_mM=b.buffer_concentration_mM,
        )

        # ── Step 2: 리스크 팩터 모듈 실행 ────────────────────────────────────
        # modules dict: {이름: (score, explanation)}
        # ML 모델 등록 시 self._module_overrides[이름]으로 교체 가능
        modules: dict[str, tuple[float, str]] = {
            "pi_proximity": (
                self._module_overrides.get("pi_proximity", compute_pi_proximity_risk)(
                    donnan.micro_pH, p.isoelectric_point_pI,
                    self.coef_overrides.get("pi_proximity")
                ) if "pi_proximity" in self._module_overrides else
                compute_pi_proximity_risk(
                    donnan.micro_pH, p.isoelectric_point_pI,
                    self.coef_overrides.get("pi_proximity")
                )
            ),
            "concentration": compute_concentration_risk(
                p.protein_concentration_mg_per_mL,
                self.coef_overrides.get("concentration")
            ),
            "hydrophobicity": compute_hydrophobicity_risk(p.hydrophobicity_index),
            "hotspot":        compute_hotspot_risk(p.aggregation_hotspot_score),
            "ionic_strength": compute_ionic_strength_risk(
                donnan.local_ionic_strength_mM,
                self.coef_overrides.get("ionic_strength")
            ),
            "buffer":         compute_buffer_risk(b.buffer_type),
            "glycosylation":  compute_glycosylation_effect(
                p.glycosylation_ratio,
                self.coef_overrides.get("glycosylation")
            ),
            "process_stress": compute_process_stress_risk(
                inputs.stress,
                self.coef_overrides.get("process_stress")
            ),
        }

        # ── Step 3: 가중 합산 ────────────────────────────────────────────────
        factor_scores: list[FactorScore] = []
        raw_risk = 0.0
        for name, (score, explanation) in modules.items():
            w = self.weights.get(name, 0.0)
            weighted = score * w
            raw_risk += weighted
            factor_scores.append(FactorScore(
                name=name,
                raw_score=score,
                weight=w,
                weighted_score=weighted,
                explanation=explanation,
            ))

        # ── Step 4: Excipient 보호 효과 차감 ─────────────────────────────────
        surf_prot,  surf_note  = compute_surfactant_protection(
            inputs.surfactants, self.coef_overrides.get("surfactant"))
        sugar_prot, sugar_note = compute_sugar_protection(
            inputs.sugars, self.coef_overrides.get("sugar"))
        aa_prot,    aa_note    = compute_amino_acid_protection(
            inputs.amino_acids, self.coef_overrides.get("amino_acid"))

        total_protection = min(0.60, surf_prot + sugar_prot + aa_prot)
        applied_protection = total_protection * EXCIPIENT_PROTECTION_WEIGHT

        excipient_detail = {
            "Surfactant":   (surf_prot,  surf_note),
            "Sugar":        (sugar_prot, sugar_note),
            "Amino acid":   (aa_prot,    aa_note),
        }

        raw_risk = max(0.0, raw_risk - applied_protection)

        # ── Step 5: 단백질 타입 보정 ─────────────────────────────────────────
        type_mod, type_note = PROTEIN_TYPE_MODIFIERS.get(
            p.protein_type, (1.0, "unknown"))
        final_risk = raw_risk * type_mod

        # ── Step 6: 클리핑 및 분류 ───────────────────────────────────────────
        final_risk = max(0.01, min(0.99, final_risk))
        if   final_risk < 0.25: level = "LOW"
        elif final_risk < 0.50: level = "MODERATE"
        elif final_risk < 0.75: level = "HIGH"
        else:                   level = "CRITICAL"

        # ── Step 7: 권고사항 ─────────────────────────────────────────────────
        recs = self._generate_recommendations(inputs, donnan, final_risk, IS)

        return PredictionResult(
            aggregation_risk_score=round(final_risk, 4),
            risk_level=level,
            colloidal_stability_estimate=round(1.0 - final_risk, 4),
            donnan=donnan,
            factor_scores=factor_scores,
            excipient_protection_detail=excipient_detail,
            excipient_protection_total=round(applied_protection, 4),
            protein_type_modifier=type_mod,
            protein_type_note=type_note,
            recommendations=recs,
            input_flat_dict=inputs.to_flat_dict(),
        )

    # ── 8-2. 실험 데이터 기반 보정 ────────────────────────────────────────────

    def calibrate_from_experimental_data(
        self,
        exp_df: "pd.DataFrame",              # type: ignore[name-defined]
        target_col: str = "measured_HMW_pct",
        method: str = "linear_regression",
    ) -> dict[str, float]:
        """
        실험 데이터로 가중치를 보정하는 메서드.

        Args:
            exp_df: 실험 결과 DataFrame.
                    필수 컬럼: FormulationInputs.to_flat_dict() 키 +
                    measured_HMW_pct (또는 measured_Tm_C, measured_B22 등)
            target_col: 보정 대상 측정값 컬럼명.
            method: "linear_regression" | "ridge" | "random_forest" | "gp"

        Returns:
            보정된 가중치 dict.

        [현재 상태] 아직 mechanistic 모델만 구현.
        이 메서드는 향후 ML 연동을 위한 인터페이스 예약.

        구현 방향 (실험 데이터 확보 후):
            1. linear_regression: scikit-learn Ridge/Lasso.
               각 factor_score를 X, measured_HMW_pct를 y로 선형 회귀.
               계수가 새 가중치가 됨.

            2. random_forest: 비선형 interaction 포착.
               feature importance → 가중치 재계산.

            3. gp (Gaussian Process): Bayesian optimization용.
               불확실성 정량화 → acquisition function으로 다음 실험 조건 제안.

        사용 예시 (향후):
            import pandas as pd
            df = pd.read_csv("ht_screening_results.csv")
            new_weights = model.calibrate_from_experimental_data(df)
            model.weights = new_weights
        """
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            print("[calibrate] pandas/numpy 미설치. pip install pandas numpy.")
            return self.weights

        if target_col not in exp_df.columns:
            warnings.warn(f"[calibrate] '{target_col}' 컬럼 없음. 보정 건너뜀.")
            return self.weights

        # 각 실험 조건에 대해 factor score 계산
        factor_names = list(self.weights.keys())
        X_rows = []
        y_vals = []

        for _, row in exp_df.dropna(subset=[target_col]).iterrows():
            try:
                inp = _row_to_inputs(row)   # helper: dict → FormulationInputs
                result = self.predict(inp)
                x_vec = [fs.raw_score for fs in result.factor_scores]
                X_rows.append(x_vec)
                y_vals.append(float(row[target_col]))
            except Exception as e:
                warnings.warn(f"[calibrate] 행 처리 실패: {e}")
                continue

        if len(X_rows) < 5:
            warnings.warn("[calibrate] 보정에 최소 5개 데이터 포인트 필요.")
            return self.weights

        X = np.array(X_rows)
        y = np.array(y_vals)

        if method == "linear_regression":
            from numpy.linalg import lstsq
            # 비음수 가중치: 간단한 최소제곱 → 보정 계수 추출
            coefs, *_ = lstsq(X, y, rcond=None)
            # 비음수 클리핑 후 정규화
            coefs = np.clip(coefs, 0.001, None)
            coefs /= coefs.sum()
            new_weights = dict(zip(factor_names, coefs.tolist()))
        elif method == "ridge":
            from sklearn.linear_model import Ridge  # type: ignore
            model = Ridge(alpha=1.0, fit_intercept=False, positive=True)
            model.fit(X, y)
            coefs = np.clip(model.coef_, 0.001, None)
            coefs /= coefs.sum()
            new_weights = dict(zip(factor_names, coefs.tolist()))
        else:
            warnings.warn(f"[calibrate] method='{method}' 미구현. 기존 가중치 유지.")
            return self.weights

        self.weights = new_weights
        print(f"[calibrate] 보정 완료 (n={len(X_rows)}, method={method}):")
        for k, v in new_weights.items():
            print(f"  {k:<22} : {v:.4f}")
        return new_weights

    # ── 8-3. Sensitivity Analysis ─────────────────────────────────────────────

    def sensitivity_analysis(
        self,
        base_inputs: FormulationInputs,
        param_ranges: dict[str, tuple[float, float, int]] | None = None,
    ) -> dict[str, list[tuple[float, float]]]:
        """
        One-at-a-time (OAT) 민감도 분석.
        하나의 파라미터를 변화시키며 나머지는 고정 → 각 파라미터의 영향력 파악.

        Args:
            base_inputs: 기준 제형 조건.
            param_ranges: {파라미터명: (최솟값, 최댓값, 스텝수)}.

        Returns:
            {파라미터명: [(값, 리스크), ...]}

        사용 예시:
            sa = model.sensitivity_analysis(inputs, {"formulation_pH": (4.5, 8.0, 20)})
            for ph, risk in sa["formulation_pH"]:
                print(f"pH {ph:.1f} → risk {risk:.3f}")
        """
        if param_ranges is None:
            param_ranges = {
                "formulation_pH":                  (4.5, 8.0,  20),
                "protein_concentration_mg_per_mL": (20,  300,  15),
                "ionic_strength_mM":               (0,   400,  15),
                "sucrose_percent":                 (0,   15,   10),
                "arginine_mM":                     (0,   200,  10),
                "polysorbate80_percent":           (0,   0.15,  8),
                "hydrophobicity_index":            (0.1, 0.9,  10),
            }

        results: dict[str, list[tuple[float, float]]] = {}
        for param, (lo, hi, steps) in param_ranges.items():
            curve = []
            for i in range(steps):
                val = lo + (hi - lo) * i / max(1, steps - 1)
                inp = copy.deepcopy(base_inputs)
                _set_param(inp, param, val)
                risk = self.predict(inp).aggregation_risk_score
                curve.append((round(val, 4), risk))
            results[param] = curve
        return results

    # ── 8-4. DOE 그리드 스캔 ──────────────────────────────────────────────────

    def doe_grid_scan(
        self,
        base_inputs: FormulationInputs,
        doe_factors: dict[str, list[float]],
    ) -> list[dict[str, Any]]:
        """
        DOE(Design of Experiments) 그리드 스캔.
        지정된 팩터의 모든 조합(full factorial)에 대해 예측 실행.

        Args:
            base_inputs: 기준 제형 조건.
            doe_factors: {파라미터명: [레벨1, 레벨2, ...]}.
                         총 조합 수 = Π(각 팩터 레벨 수).

        Returns:
            각 조합의 예측 결과 dict 리스트 (DataFrame 변환 가능).

        사용 예시:
            grid = model.doe_grid_scan(inputs, {
                "formulation_pH":   [5.5, 6.0, 6.5],
                "arginine_mM":      [0, 50, 100, 150],
                "sucrose_percent":  [5, 10],
            })
            # 24가지 조합 결과
            import pandas as pd
            df = pd.DataFrame(grid)
        """
        import itertools

        factor_names = list(doe_factors.keys())
        factor_levels = list(doe_factors.values())
        records = []

        for combo in itertools.product(*factor_levels):
            inp = copy.deepcopy(base_inputs)
            row: dict[str, Any] = {}
            for name, val in zip(factor_names, combo):
                _set_param(inp, name, val)
                row[name] = val
            result = self.predict(inp)
            row["risk_score"]    = result.aggregation_risk_score
            row["risk_level"]    = result.risk_level
            row["donnan_dPH"]    = result.donnan.delta_pH
            row["micro_pH"]      = result.donnan.micro_pH
            row["excipient_protection"] = result.excipient_protection_total
            records.append(row)

        return records

    # ── 8-5. 권고사항 생성 ────────────────────────────────────────────────────

    @staticmethod
    def _generate_recommendations(
        inputs: FormulationInputs,
        donnan: DonnanResult,
        final_risk: float,
        IS_mM: float,
    ) -> list[str]:
        """
        예측 결과 기반 제형 최적화 권고사항 생성.
        각 조건문은 독립적으로 평가되므로 여러 권고사항이 동시 출력 가능.
        """
        p  = inputs.protein
        b  = inputs.buffer
        sf = inputs.surfactants
        sg = inputs.sugars
        aa = inputs.amino_acids
        ps = inputs.stress

        recs: list[str] = []
        pi_dist = abs(donnan.micro_pH - p.isoelectric_point_pI)

        # pI 근접도
        if pi_dist < 0.5:
            recs.append(
                f"[CRITICAL] microPH {donnan.micro_pH:.2f}가 pI {p.isoelectric_point_pI}에 "
                f"0.5 unit 이내 → 정전기 반발 소멸. pH를 ±1.5 unit 이상 조정하거나 pI 엔지니어링 검토."
            )
        elif pi_dist < 1.2:
            recs.append(
                f"[HIGH] pH–pI 거리 {pi_dist:.2f} unit. ≥1.5 unit 이격 권장."
            )

        # Donnan shift
        if abs(donnan.delta_pH) > 0.15:
            direction = "알칼리" if donnan.delta_pH > 0 else "산성"
            recs.append(
                f"[Donnan] microenvironment pH가 bulk 대비 {direction}으로 {abs(donnan.delta_pH):.3f} 이동. "
                f"버퍼 농도 ≥40 mM 상향 또는 NaCl 50–100 mM 추가 검토."
            )

        # 고농도
        if p.protein_concentration_mg_per_mL > 150:
            recs.append(
                f"[농도] {p.protein_concentration_mg_per_mL} mg/mL 고농도 → "
                f"Arg-HCl 100–150 mM 및/또는 Proline 100–200 mM 추가 권장."
            )

        # 계면활성제 부재 + 물리 스트레스
        total_surf = sf.polysorbate20_percent + sf.polysorbate80_percent + sf.poloxamer188_percent
        if total_surf < 0.01 and (ps.agitation_risk_level > 0.3 or ps.pumping_stress_level > 0.3):
            recs.append("[스트레스] 계면활성제 미첨가 + 물리 스트레스 존재 → PS80 0.02–0.05% 즉시 추가 필요.")

        # 이당류 부재
        total_disaccharide = sg.sucrose_percent + sg.trehalose_percent
        if total_disaccharide < 2.0 and p.protein_concentration_mg_per_mL > 80:
            recs.append("[안정화] 이당류 미첨가 → Sucrose 5–10% 또는 Trehalose 5–8% 추가 권장 (우선적 배제 효과).")

        # 이온강도
        if IS_mM < 30:
            recs.append(f"[IS] 이온강도 {IS_mM:.0f} mM 매우 낮음 → 50–150 mM 목표 (NaCl 또는 히스티딘 증량).")
        elif IS_mM > 300:
            recs.append(f"[IS] 이온강도 {IS_mM:.0f} mM 높음 → 소수성 aggregation 촉진 가능. 250 mM 이하 검토.")

        # 버퍼 특이 사항
        if b.buffer_type == "citrate" and p.isoelectric_point_pI > 7.0:
            recs.append("[버퍼] Citrate + 고pI 단백질 → 비특이적 결합 위험. Histidine 전환 검토.")
        if b.buffer_type == "tris":
            recs.append("[버퍼] Tris pKa: −0.03 pH/°C → 동결 시 pH 드리프트 검증 필수.")
        if b.buffer_type == "phosphate":
            recs.append("[버퍼] Phosphate: 동결 시 Na₂HPO₄ 우선 결정화 → pH 급격히 산성화. F/T 검증 권장.")

        # 단백질 타입 특이 권고
        if p.protein_type == "adc":
            recs.append("[ADC] DAR 불균일성 → HIC 프로파일 확인 및 고DAR 종 제거 검토.")
        if p.protein_type == "sc":
            recs.append("[SC] 점도 동시 모니터링 필요 → Arg, HTAB, 또는 sodium camphorsulfonate 검토.")
        if p.protein_type == "microneedle":
            recs.append("[MN] 고체/반고체 상태 → 고체 분산 안정성(XRPD, DSC) 추가 평가 권장.")

        # 소수성
        if p.hydrophobicity_index > 0.65:
            recs.append(
                f"[소수성] HIC index {p.hydrophobicity_index:.2f} 높음 → "
                f"IS 100–150 mM 최적화, Arg 첨가, 또는 표면 엔지니어링 고려."
            )

        # 저위험 권고
        if final_risk < 0.25:
            recs.append("[결과] 현재 조건 LOW risk → DOE 실험 범위 확장 및 가속 안정성 시험 진행 가능.")

        return recs


# ---------------------------------------------------------------------------
# 섹션 9: 유틸리티 헬퍼 함수
# ---------------------------------------------------------------------------

def _set_param(inputs: FormulationInputs, param: str, value: float) -> None:
    """파라미터명을 받아 올바른 서브 dataclass에 값 설정 (DOE·sensitivity 용)."""
    for sub in [inputs.protein, inputs.buffer, inputs.ions,
                inputs.surfactants, inputs.sugars, inputs.amino_acids, inputs.stress]:
        if hasattr(sub, param):
            setattr(sub, param, value)
            return
    warnings.warn(f"[_set_param] '{param}' 파라미터를 찾을 수 없음.")


def _row_to_inputs(row: dict | "pd.Series") -> FormulationInputs:  # type: ignore[name-defined]
    """
    flat dict (DataFrame 행) → FormulationInputs 변환.
    calibrate_from_experimental_data() 내부에서 사용.
    HT 스크리닝 CSV와 직접 연동 가능.
    """
    r = dict(row)
    def g(key: str, default=0.0):
        val = r.get(key, default)
        return default if (val is None or (isinstance(val, float) and math.isnan(val))) else float(val)

    return FormulationInputs(
        protein=ProteinProperties(
            molecular_weight_kDa=g("molecular_weight_kDa", 150),
            isoelectric_point_pI=g("isoelectric_point_pI", 7.0),
            formulation_pH=g("formulation_pH", 6.0),
            protein_concentration_mg_per_mL=g("protein_concentration_mg_per_mL", 100),
            hydrophobicity_index=g("hydrophobicity_index", 0.4),
            aggregation_hotspot_score=g("aggregation_hotspot_score", 0.3),
            protein_type=r.get("protein_type", "mab"),
            glycosylation_ratio=r.get("glycosylation_ratio"),
        ),
        buffer=BufferConditions(
            buffer_type=r.get("buffer_type", "histidine"),
            buffer_concentration_mM=g("buffer_concentration_mM", 20),
        ),
        ions=IonicEnvironment(
            NaCl_mM=g("NaCl_mM"),
            KCl_mM=g("KCl_mM"),
            ionic_strength_mM=r.get("ionic_strength_mM"),
        ),
        surfactants=Surfactants(
            polysorbate20_percent=g("polysorbate20_percent"),
            polysorbate80_percent=g("polysorbate80_percent"),
            poloxamer188_percent=g("poloxamer188_percent"),
        ),
        sugars=SugarStabilizers(
            sucrose_percent=g("sucrose_percent"),
            trehalose_percent=g("trehalose_percent"),
            mannitol_percent=g("mannitol_percent"),
            sorbitol_percent=g("sorbitol_percent"),
        ),
        amino_acids=AminoAcidStabilizers(
            arginine_mM=g("arginine_mM"),
            glycine_mM=g("glycine_mM"),
            lysine_mM=g("lysine_mM"),
        ),
        stress=ProcessStress(
            agitation_risk_level=g("agitation_risk_level"),
            pumping_stress_level=g("pumping_stress_level"),
            thermal_stress_level=g("thermal_stress_level"),
        ),
    )


# ---------------------------------------------------------------------------
# 섹션 10: HT 스크리닝 데이터 구조 예시
# ---------------------------------------------------------------------------
# 이 섹션은 실험실에서 수집해야 할 데이터의 스키마를 정의합니다.
# CSV/Excel로 수집 후 calibrate_from_experimental_data()에 바로 사용 가능.

HT_SCREENING_SCHEMA = {
    "description": "High-Throughput Formulation Screening Data Schema v1.0",

    "input_columns": {
        # 단백질 특성 (고정 or 변수)
        "protein_id":                     "str   — 단백질 식별자 (예: mAb-001)",
        "molecular_weight_kDa":           "float — kDa",
        "isoelectric_point_pI":           "float — 4.0–11.0",
        "protein_type":                   "str   — mab/adc/peptide/sc/...",
        "hydrophobicity_index":           "float — 0–1 (HIC 정규화)",
        "aggregation_hotspot_score":      "float — 0–1 (in-silico APR)",
        "glycosylation_ratio":            "float — 0–1 (optional)",

        # 제형 조건 (DOE 변수)
        "formulation_pH":                 "float — 스크리닝 pH (예: 5.0, 5.5, 6.0, 6.5)",
        "protein_concentration_mg_per_mL":"float — 제형 농도",
        "buffer_type":                    "str   — histidine/acetate/citrate/...",
        "buffer_concentration_mM":        "float — 통상 10–50 mM",
        "NaCl_mM":                        "float — 0–200 mM",
        "ionic_strength_mM":              "float — 직접 계산값 (optional)",
        "sucrose_percent":                "float — 0–15%",
        "trehalose_percent":              "float — 0–10%",
        "mannitol_percent":               "float — 0–5%",
        "arginine_mM":                    "float — 0–200 mM",
        "glycine_mM":                     "float — 0–200 mM",
        "lysine_mM":                      "float — 0–100 mM",
        "polysorbate20_percent":          "float — 0–0.1%",
        "polysorbate80_percent":          "float — 0–0.1%",
        "poloxamer188_percent":           "float — 0–0.1%",

        # 공정 스트레스 조건
        "agitation_risk_level":           "float — 0–1 (교반 강도 정규화)",
        "pumping_stress_level":           "float — 0–1",
        "thermal_stress_level":           "float — 0–1",

        # 실험 메타데이터
        "experiment_id":                  "str   — 실험 배치 ID",
        "experiment_date":                "date  — YYYY-MM-DD",
        "analyst":                        "str   — 담당자",
        "plate_well":                     "str   — 384-well 위치 (예: A01)",
        "storage_condition":              "str   — 4°C / 25°C / 40°C",
        "storage_duration_weeks":         "int   — 보관 기간 (주)",
    },

    "output_columns": {
        # 주요 측정값 (Y 변수)
        "measured_HMW_pct":               "float — HP-SEC %HMW (고분자량 종) — 핵심 Y 변수",
        "measured_monomer_pct":           "float — HP-SEC %Monomer",
        "measured_LMW_pct":               "float — HP-SEC %LMW (저분자량, 단편화)",
        "measured_Tm1_C":                 "float — nanoDSF Tm1 (°C) — 1차 unfolding onset",
        "measured_Tm2_C":                 "float — nanoDSF Tm2 (°C) — 2차 도메인",
        "measured_Tagg_C":                "float — nanoDSF Tagg (°C) — aggregation onset",
        "measured_B22_mL_mol_g2":         "float — SLS B22 (mL·mol/g²) — colloidal stability",
        "measured_kD_mL_g":               "float — DLS kD (mL/g) — 농도 의존적 확산",
        "measured_viscosity_cP":          "float — 마이크로점도계 (고농도 제형)",
        "measured_particle_count_subvis": "float — MFI 1–100 μm 입자수/mL",
        "measured_turbidity_NTU":         "float — 탁도",
        "measured_pH_actual":             "float — 실측 pH (목표 pH와 비교용)",

        # 비주얼 평가
        "visual_appearance":              "str   — clear/slightly turbid/turbid/particulate",
        "color_change":                   "bool  — 색상 변화 여부",
    },

    "recommended_hts_format": """
    # HT 스크리닝 권장 플레이트 설계:
    # - 384-well 플레이트, 각 웰 100–200 μL
    # - 단백질 농도: 1–10 mg/mL (HT 스케일 다운)
    # - pH: 4–5 레벨 (5.0, 5.5, 6.0, 6.5, 7.0)
    # - 주요 excipient: 3–4 레벨
    # - 복제(replicate): 3회 이상
    # - 대조군(control): 표준 버퍼(His/Suc/PS80) 포함
    # - 판독(readout): Dynapro Plate Reader (DLS), UNCLE (nanoDSF+SLS)
    """
}


# ---------------------------------------------------------------------------
# 섹션 11: DOE 팩터 가이드라인
# ---------------------------------------------------------------------------

DOE_FACTOR_GUIDE = {
    "title": "DOE Factor Selection Guide for Aggregation Screening",

    "critical_factors": {
        "formulation_pH": {
            "range": "pI ± 2.0 (단, ≥1.5 unit 이격)",
            "levels": 5,
            "rationale": "가장 중요한 팩터. Donnan 효과, 전하 분포 결정.",
            "example": [5.0, 5.5, 6.0, 6.5, 7.0],
        },
        "arginine_mM": {
            "range": "0–200 mM",
            "levels": 4,
            "rationale": "고농도 제형의 핵심 stabilizer. 점도 감소 이중 효과.",
            "example": [0, 50, 100, 150],
        },
        "sucrose_percent": {
            "range": "0–15%",
            "levels": 3,
            "rationale": "동결보호 + 우선적 배제. pH와 상호작용 작음.",
            "example": [5, 10, 15],
        },
    },

    "important_factors": {
        "buffer_concentration_mM": {
            "range": "10–50 mM",
            "levels": 3,
            "rationale": "Donnan effect 억제. 높을수록 IS 기여 증가.",
            "example": [10, 20, 40],
        },
        "NaCl_mM": {
            "range": "0–150 mM",
            "levels": 4,
            "rationale": "IS 조절, 삼투압, Donnan 보정.",
            "example": [0, 50, 100, 150],
        },
        "polysorbate80_percent": {
            "range": "0.005–0.05%",
            "levels": 3,
            "rationale": "계면 보호. 교반/펌핑 스트레스 있을 때 필수.",
            "example": [0.01, 0.02, 0.04],
        },
    },

    "optional_factors": {
        "trehalose_percent": {
            "range": "0–10%",
            "levels": 2,
            "rationale": "sucrose 대체 또는 병용. 동결건조 우선.",
            "example": [0, 8],
        },
        "glycine_mM": {
            "range": "0–150 mM",
            "levels": 3,
            "rationale": "Arg 병용 시 시너지. 점도 개선 한계.",
            "example": [0, 75, 150],
        },
        "lysine_mM": {
            "range": "0–100 mM",
            "levels": 2,
            "rationale": "ADC 또는 고소수성 단백질에서 검토.",
            "example": [0, 100],
        },
    },

    "suggested_doe_designs": {
        "phase1_broad": {
            "design": "Fractional Factorial (Resolution IV)",
            "factors": ["pH", "arginine", "sucrose", "NaCl", "PS80"],
            "runs": 16,
            "purpose": "주요 효과 및 2인자 상호작용 파악",
        },
        "phase2_optimisation": {
            "design": "Central Composite Design (CCD) or Box-Behnken",
            "factors": ["pH", "arginine", "sucrose"],
            "runs": 15,
            "purpose": "반응표면법(RSM)으로 최적 조건 예측",
        },
        "phase3_confirmation": {
            "design": "Mixture Design (simplex-centroid)",
            "factors": ["sucrose", "trehalose", "arginine", "glycine"],
            "runs": 10,
            "purpose": "excipient 혼합 비율 최적화",
        },
    },
}


# ---------------------------------------------------------------------------
# 섹션 12: AI 고도화 로드맵
# ---------------------------------------------------------------------------

AI_UPGRADE_ROADMAP = """
════════════════════════════════════════════════════════════════════════
AggPredict AI 고도화 로드맵
════════════════════════════════════════════════════════════════════════

Phase 0 (현재): Mechanistic Prototype
─────────────────────────────────────
• 물리화학 공식 기반 규칙 엔진
• 해석 가능, 수식 투명
• 데이터 불필요
• 정확도: 상대적 순위 신뢰, 절대값 ±20%

Phase 1 (데이터 50–200개): 가중치 보정
──────────────────────────────────────
• calibrate_from_experimental_data() 활용
• Ridge Regression / Lasso
  - X: 각 factor_score (8개 feature)
  - Y: measured_HMW_pct 또는 measured_Tm1_C
• 모듈별 slope/intercept 보정
• 정확도 목표: R² > 0.7

Phase 2 (데이터 200–1000개): ML Hybrid
───────────────────────────────────────
• Random Forest / XGBoost 비선형 포착
  - feature = to_flat_dict() 전체 파라미터
  - SHAP value로 feature importance 해석 (해석 유지)
• Gaussian Process (GP) 회귀
  - 불확실성 정량화 (예측 신뢰구간 제공)
  - Bayesian Optimization acquisition function 연동
    → 다음 실험 조건 자동 제안
• 정확도 목표: R² > 0.85, %HMW 오차 < 2%

Phase 3 (데이터 1000개+): 딥러닝 + 서열 통합
────────────────────────────────────────────────
• 단백질 서열 → ESM-2 임베딩 → 제형 조건 concat
  → DNN 통합 예측
• Graph Neural Network:
  단백질 구조(PDB) + 제형 조건 → 직접 aggregation 예측
• Transfer Learning:
  공개 데이터셋(ProThermDB, ProtaBank) pre-training
  → 사내 데이터 fine-tuning
• Active Learning 루프:
  GP uncertainty가 높은 조건 → HT 실험 우선 제안
  → 데이터 수집 → 모델 업데이트 (자동화)

실험 데이터 수집 우선순위:
───────────────────────────
1순위 (필수): HP-SEC %HMW — 가장 직접적 aggregation 지표
2순위 (권장): nanoDSF Tm1, Tagg — 열안정성 (Tm과 aggregation 상관)
3순위 (유용): DLS kD, SLS B22 — colloidal stability (Donnan 보정용)
4순위 (고급): 점도 (고농도 SC 제형), MFI 입자수 (가시/비가시 입자)
5순위 (선택): NMR pH 측정 — Donnan 모듈 직접 보정

구현 도구 스택:
───────────────
scikit-learn  → Phase 1 (Ridge, RF)
XGBoost       → Phase 2 (gradient boosting)
GPyTorch      → Phase 2–3 (Gaussian Process, GPU 지원)
SHAP          → 모든 Phase (해석 가능성 유지)
BoTorch       → Phase 2–3 (Bayesian Optimization)
ESM-2         → Phase 3 (protein language model)
MLflow        → 실험 추적, 모델 버전 관리
════════════════════════════════════════════════════════════════════════
"""


# ---------------------------------------------------------------------------
# 섹션 13: 시각화 예시 코드
# ---------------------------------------------------------------------------

VISUALIZATION_EXAMPLES = '''
"""
시각화 예시 코드 (matplotlib + seaborn 필요: pip install matplotlib seaborn)

실행 방법:
    python aggpredict_v2.py --visualize
또는 별도 스크립트에서 import:
    from aggpredict_v2 import AggregationRiskModel, FormulationInputs, ...
    exec(VISUALIZATION_EXAMPLES)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

model = AggregationRiskModel()

# ─────────────────────────────────────────────────────────────────────────────
# 시각화 1: Heatmap — pH vs Arginine concentration
# ─────────────────────────────────────────────────────────────────────────────

base = FormulationInputs(
    protein=ProteinProperties(
        molecular_weight_kDa=148, isoelectric_point_pI=8.4,
        formulation_pH=6.0, protein_concentration_mg_per_mL=175,
        hydrophobicity_index=0.52, aggregation_hotspot_score=0.38,
        protein_type="sc",
    ),
    buffer=BufferConditions(buffer_type="histidine", buffer_concentration_mM=20),
    sugars=SugarStabilizers(sucrose_percent=9.0),
    surfactants=Surfactants(polysorbate80_percent=0.04),
)

pH_range  = np.linspace(4.5, 8.5, 30)
arg_range = np.linspace(0, 200, 30)
Z = np.zeros((len(arg_range), len(pH_range)))

for i, arg in enumerate(arg_range):
    for j, ph in enumerate(pH_range):
        inp = __import__('copy').deepcopy(base)
        inp.protein.formulation_pH = ph
        inp.amino_acids.arginine_mM = arg
        Z[i, j] = model.predict(inp).aggregation_risk_score

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
cmap = mcolors.LinearSegmentedColormap.from_list(
    "risk", ["#1D9E75", "#EF9F27", "#E24B4A"])
im = axes[0].contourf(pH_range, arg_range, Z, levels=20, cmap=cmap)
axes[0].set_xlabel("Formulation pH", fontsize=12)
axes[0].set_ylabel("Arginine concentration (mM)", fontsize=12)
axes[0].set_title("Aggregation Risk — pH × Arginine", fontsize=13)
plt.colorbar(im, ax=axes[0], label="Risk Score")
# pI 기준선
axes[0].axvline(x=8.4, color="white", linestyle="--", alpha=0.7, label=f"pI = 8.4")
axes[0].legend(fontsize=10)

# ─────────────────────────────────────────────────────────────────────────────
# 시각화 2: Sensitivity tornado chart
# ─────────────────────────────────────────────────────────────────────────────
sa = model.sensitivity_analysis(base, param_ranges={
    "formulation_pH":                  (4.5, 8.5, 20),
    "protein_concentration_mg_per_mL": (50, 300, 20),
    "arginine_mM":                     (0, 200, 20),
    "sucrose_percent":                 (0, 15, 20),
    "ionic_strength_mM":               (0, 300, 20),
    "hydrophobicity_index":            (0.1, 0.9, 20),
    "polysorbate80_percent":           (0, 0.1, 20),
})
ranges = {k: (max(v, key=lambda x: x[1])[1] - min(v, key=lambda x: x[1])[1])
          for k, v in sa.items()}
sorted_params = sorted(ranges, key=ranges.get, reverse=True)
colors = ["#E24B4A" if ranges[p] > 0.15 else "#EF9F27" if ranges[p] > 0.08 else "#1D9E75"
          for p in sorted_params]
axes[1].barh(sorted_params, [ranges[p] for p in sorted_params], color=colors)
axes[1].set_xlabel("Risk range (max − min)", fontsize=12)
axes[1].set_title("Parameter Sensitivity (Tornado Chart)", fontsize=13)
axes[1].axvline(x=0, color="gray", linewidth=0.5)

plt.tight_layout()
plt.savefig("aggpredict_heatmap_tornado.png", dpi=150, bbox_inches="tight")
plt.show()
print("시각화 저장: aggpredict_heatmap_tornado.png")

# ─────────────────────────────────────────────────────────────────────────────
# 시각화 3: DOE 결과 heatmap (pH × Sucrose)
# ─────────────────────────────────────────────────────────────────────────────
grid = model.doe_grid_scan(base, {
    "formulation_pH":   [5.0, 5.5, 6.0, 6.5, 7.0],
    "sucrose_percent":  [0, 5, 10, 15],
    "arginine_mM":      [0, 100],
})
import pandas as pd
df_grid = pd.DataFrame(grid)
for arg_val, grp in df_grid.groupby("arginine_mM"):
    pivot = grp.pivot(index="sucrose_percent", columns="formulation_pH", values="risk_score")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r",
                vmin=0, vmax=1, ax=ax2,
                cbar_kws={"label": "Risk Score"})
    ax2.set_title(f"DOE Risk Map — Arg={arg_val} mM")
    ax2.set_xlabel("Formulation pH")
    ax2.set_ylabel("Sucrose (%)")
    plt.tight_layout()
    plt.savefig(f"doe_heatmap_arg{int(arg_val)}.png", dpi=150)
    plt.show()
'''


# ---------------------------------------------------------------------------
# 섹션 14: 실행 예시
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    model = AggregationRiskModel()

    # ── 예시 1: 고농도 mAb SC 제형 ───────────────────────────────────────────
    print("\n" + "="*68)
    print("예시 1: 고농도 mAb SC 제형 (175 mg/mL)")
    print("="*68)

    mab_sc = FormulationInputs(
        protein=ProteinProperties(
            molecular_weight_kDa=148,
            isoelectric_point_pI=8.4,
            formulation_pH=5.8,
            protein_concentration_mg_per_mL=175,
            hydrophobicity_index=0.52,
            aggregation_hotspot_score=0.38,
            protein_type="sc",
            glycosylation_ratio=0.12,
        ),
        buffer=BufferConditions(buffer_type="histidine", buffer_concentration_mM=20),
        ions=IonicEnvironment(NaCl_mM=0),
        surfactants=Surfactants(polysorbate80_percent=0.04),
        sugars=SugarStabilizers(sucrose_percent=9.0),
        amino_acids=AminoAcidStabilizers(arginine_mM=100),
        stress=ProcessStress(agitation_risk_level=0.5, pumping_stress_level=0.4),
    )
    r1 = model.predict(mab_sc)
    print(r1.summary())

    # ── 예시 2: ADC 제형 ─────────────────────────────────────────────────────
    print("="*68)
    print("예시 2: ADC 저농도 제형 (소수성 높음)")
    print("="*68)

    adc = FormulationInputs(
        protein=ProteinProperties(
            molecular_weight_kDa=152,
            isoelectric_point_pI=7.8,
            formulation_pH=5.5,
            protein_concentration_mg_per_mL=8,
            hydrophobicity_index=0.71,
            aggregation_hotspot_score=0.55,
            protein_type="adc",
        ),
        buffer=BufferConditions(buffer_type="histidine", buffer_concentration_mM=10),
        ions=IonicEnvironment(NaCl_mM=85),
        surfactants=Surfactants(polysorbate20_percent=0.04),
        sugars=SugarStabilizers(sucrose_percent=8.5),
        stress=ProcessStress(thermal_stress_level=0.3),
    )
    r2 = model.predict(adc)
    print(r2.summary())

    # ── 예시 3: DOE 그리드 스캔 ──────────────────────────────────────────────
    print("="*68)
    print("예시 3: DOE 그리드 스캔 — pH × Arginine × Sucrose")
    print("="*68)

    grid = model.doe_grid_scan(mab_sc, doe_factors={
        "formulation_pH":  [5.5, 6.0, 6.5],
        "arginine_mM":     [0, 100, 150],
        "sucrose_percent": [5, 10],
    })
    print(f"\n  총 {len(grid)}개 조건 예측 완료\n")
    print(f"  {'pH':>5}  {'Arg(mM)':>8}  {'Suc(%)':>7}  {'Risk':>7}  Level")
    print(f"  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*10}")
    for row in sorted(grid, key=lambda x: x["risk_score"]):
        print(f"  {row['formulation_pH']:>5.1f}  {row['arginine_mM']:>8.0f}  "
              f"{row['sucrose_percent']:>7.1f}  {row['risk_score']:>7.4f}  {row['risk_level']}")

    # ── 예시 4: 민감도 분석 ───────────────────────────────────────────────────
    print("\n" + "="*68)
    print("예시 4: pH 민감도 분석 (SC mAb)")
    print("="*68)
    sa = model.sensitivity_analysis(mab_sc, {"formulation_pH": (4.5, 8.5, 10)})
    print(f"\n  {'pH':>6}  {'Risk':>7}  Bar")
    for ph, risk in sa["formulation_pH"]:
        bar = "█" * int(risk * 20) + "░" * (20 - int(risk * 20))
        print(f"  {ph:>6.2f}  {risk:>7.4f}  {bar}")

    # ── HT 스크리닝 스키마 출력 ───────────────────────────────────────────────
    if "--schema" in sys.argv:
        print("\n" + "="*68)
        print("HT 스크리닝 데이터 스키마")
        print("="*68)
        print(json.dumps(HT_SCREENING_SCHEMA, ensure_ascii=False, indent=2))

    # ── AI 고도화 로드맵 출력 ─────────────────────────────────────────────────
    if "--roadmap" in sys.argv:
        print(AI_UPGRADE_ROADMAP)

    # ── 시각화 ────────────────────────────────────────────────────────────────
    if "--visualize" in sys.argv:
        print("\n시각화 실행 중...")
        try:
            exec(VISUALIZATION_EXAMPLES)
        except ImportError as e:
            print(f"시각화 라이브러리 미설치: {e}")
            print("pip install matplotlib seaborn")
