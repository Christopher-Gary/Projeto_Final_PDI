"""Utilitários compartilhados de inferência, regra de decisão em 3 faixas e visualização.

Reutilizado por `src/4_predict_single.py` e `src/5_predict_batch.py` para manter
exatamente o mesmo pipeline (YOLO-Seg -> remoção de fundo -> LAB-CLAHE ->
extração de features -> classificador -> decisão em 3 faixas -> card visual),
sem duplicação de código e com carregamento único de modelos em modo lote.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import numpy as np

from pdi_core import (
    DEFAULT_DINO_IMG_SIZE,
    DEFAULT_HSV_BINS,
    DEFAULT_LBP_CANVAS_SIZE,
    DEFAULT_LBP_CONFIGS,
    DINOv2Extractor,
    PreprocessResult,
    extract_pdi_features,
    get_device,
    load_yolo_seg_model,
    read_image_bgr,
    segment_and_preprocess_cat,
)

# Limiares padrão da regra de decisão em 3 faixas
DEFAULT_CONFIRM_THRESHOLD = 0.50
DEFAULT_UNKNOWN_THRESHOLD = 0.40

# Constantes de status e rótulos
STATUS_CONFIRMADO = "CONFIRMADO"
STATUS_INCERTO = "INCERTO"
STATUS_DESCONHECIDO = "DESCONHECIDO"

UNKNOWN_LABEL = "Gato Desconhecido"
UNCERTAINTY_FLAG = "[ALERTA: BAIXA CONFIANÇA / INCERTEZA]"
UNKNOWN_FLAG = "[ALERTA: GATO DESCONHECIDO]"

# Paleta padronizada de cores por faixa de confiança
STATUS_STYLES: dict[str, dict[str, Any]] = {
    STATUS_CONFIRMADO: {
        "banner_color": "#1b5e20",
        "accent": "#2e7d32",
        "bg": "#e8f5e9",
        "box_bgr": (46, 180, 46),  # Verde
        "badge": "CONFIRMADO",
    },
    STATUS_INCERTO: {
        "banner_color": "#e65100",
        "accent": "#f57c00",
        "bg": "#fff3e0",
        "box_bgr": (0, 140, 255),  # Laranja/Âmbar
        "badge": "INCERTO",
    },
    STATUS_DESCONHECIDO: {
        "banner_color": "#b71c1c",
        "accent": "#c62828",
        "bg": "#ffebee",
        "box_bgr": (40, 40, 220),  # Vermelho
        "badge": "DESCONHECIDO",
    },
}


@dataclass
class PredictionResult:
    """Resultado completo da inferência de Cat Re-ID para uma imagem."""

    image_path: Path
    pre_res: PreprocessResult
    ranked_probs: list[tuple[str, float]]
    top_guess_label: str
    pred_label: str
    max_prob: float
    status: str
    alert_flag: str
    confirm_threshold: float
    unknown_threshold: float
    scenario_label: str
    classifier_name: str
    true_label: str | None = None

    @property
    def top3_probs(self) -> list[tuple[str, float]]:
        """Retorna as 3 maiores probabilidades ordenadas de forma decrescente."""
        return self.ranked_probs[:3]

    @property
    def top3_inline(self) -> str:
        """Formata o ranking Top-3 em linha única para cards e terminal."""
        return " | ".join(
            f"{idx}. {name} ({prob * 100.0:.1f}%)"
            for idx, (name, prob) in enumerate(self.top3_probs, start=1)
        )

    @property
    def is_correct(self) -> bool | None:
        """Indica se a predição acertou `true_label` (None quando não há rótulo real)."""
        if self.true_label is None:
            return None
        return self.status != STATUS_DESCONHECIDO and self.pred_label == self.true_label


def validate_thresholds(
    confirm_threshold: float,
    unknown_threshold: float,
) -> None:
    """Valida a consistência dos limiares da regra de decisão em 3 faixas."""
    if not (0.0 <= unknown_threshold <= confirm_threshold <= 1.0):
        raise ValueError(
            "Limiares inválidos: é necessário 0.0 <= unknown_threshold <= confirm_threshold <= 1.0 "
            f"(recebido unknown={unknown_threshold:.2f}, confirm={confirm_threshold:.2f})."
        )


def classify_confidence_tier(
    top_label: str,
    max_prob: float,
    confirm_threshold: float = DEFAULT_CONFIRM_THRESHOLD,
    unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
) -> tuple[str, str, str]:
    """Aplica a regra de decisão em 3 faixas sobre a maior probabilidade prevista.

    Faixas:
      - max_prob >= confirm_threshold (padrão 0.50): CONFIRMADO (verde)
      - unknown_threshold <= max_prob < confirm_threshold (0.40 a 0.50): INCERTO (laranja)
      - max_prob < unknown_threshold (< 0.40): DESCONHECIDO -> 'Gato Desconhecido' (vermelho)
    """
    validate_thresholds(confirm_threshold, unknown_threshold)

    if max_prob >= confirm_threshold:
        return top_label, STATUS_CONFIRMADO, ""
    if max_prob >= unknown_threshold:
        return top_label, STATUS_INCERTO, UNCERTAINTY_FLAG
    return UNKNOWN_LABEL, STATUS_DESCONHECIDO, UNKNOWN_FLAG


def sanitize_filename_part(text: str, max_len: int = 45) -> str:
    """Normaliza trechos de nomes de classes/arquivos para nomes seguros no Windows/WSL."""
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', "_", str(text).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("_.")
    return cleaned or "item"


class CatReIDPredictor:
    """Motor reutilizável de inferência Cat Re-ID com carregamento único de modelos."""

    def __init__(
        self,
        model_path: Path,
        yolo_model_name: str = "yolo11n-seg.pt",
        device: str = "auto",
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("cat_reid")
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists() or not self.model_path.is_file():
            raise FileNotFoundError(
                f"Modelo treinado não encontrado: {self.model_path}. "
                "Execute primeiro 'python src/3_train_and_evaluate.py'."
            )

        with open(self.model_path, "rb") as f:
            bundle = pickle.load(f)

        self.pipeline = bundle["pipeline"]
        self.scenario: str = str(bundle["scenario"])
        self.scenario_label: str = str(bundle.get("scenario_label", self.scenario))
        self.classifier_name: str = str(bundle.get("classifier_name", "Classificador"))
        self.class_names: list[str] = [str(c) for c in bundle["class_names"]]
        metadata: dict[str, Any] = bundle.get("metadata", {})

        self.hsv_bins = tuple(int(v) for v in metadata.get("hsv_bins", DEFAULT_HSV_BINS))
        self.lbp_configs = tuple(
            (int(cfg[0]), int(cfg[1]))
            for cfg in metadata.get("lbp_configs", DEFAULT_LBP_CONFIGS)
        )
        self.lbp_canvas_size = int(metadata.get("lbp_canvas_size", DEFAULT_LBP_CANVAS_SIZE))
        dino_model_name = str(metadata.get("dino_model", "dinov2_vits14"))
        dino_img_size = int(metadata.get("dino_img_size", DEFAULT_DINO_IMG_SIZE))

        if self.scenario not in ("pdi", "dino", "hybrid"):
            raise ValueError(f"Cenário desconhecido salvo no modelo: '{self.scenario}'.")

        self.device = get_device(device, logger=self.logger)
        self.logger.info(f"Carregando modelo YOLO-Seg ({yolo_model_name})...")
        self.yolo_model = load_yolo_seg_model(yolo_model_name)

        self.dino_extractor: DINOv2Extractor | None = None
        if self.scenario in ("dino", "hybrid"):
            self.logger.info(f"Carregando extrator DINOv2 ({dino_model_name}) em modo eval()...")
            self.dino_extractor = DINOv2Extractor(
                device=self.device,
                model_name=dino_model_name,
                img_size=dino_img_size,
            )

    def _extract_feature_vector(self, pre_res: PreprocessResult) -> np.ndarray:
        """Extrai o vetor de características compatível com o cenário do modelo."""
        if self.scenario == "pdi":
            return extract_pdi_features(
                pre_res.normalized_bgr,
                fg_mask=pre_res.crop_mask,
                hsv_bins=self.hsv_bins,
                lbp_configs=self.lbp_configs,
                lbp_canvas_size=self.lbp_canvas_size,
            )

        if self.scenario == "dino":
            assert self.dino_extractor is not None
            return self.dino_extractor.extract_single(pre_res.normalized_bgr)

        # Cenário híbrido: PDI + DINOv2
        assert self.dino_extractor is not None
        pdi_vec = extract_pdi_features(
            pre_res.normalized_bgr,
            fg_mask=pre_res.crop_mask,
            hsv_bins=self.hsv_bins,
            lbp_configs=self.lbp_configs,
            lbp_canvas_size=self.lbp_canvas_size,
        )
        dino_vec = self.dino_extractor.extract_single(pre_res.normalized_bgr)
        return np.concatenate([pdi_vec, dino_vec]).astype(np.float32)

    def predict_image(
        self,
        image_path: Path,
        true_label: str | None = None,
        confirm_threshold: float = DEFAULT_CONFIRM_THRESHOLD,
        unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
        yolo_conf: float = 0.25,
        margin_ratio: float = 0.05,
        clahe_clip: float = 2.0,
        smooth_method: str = "bilateral",
    ) -> PredictionResult | None:
        """Executa o pipeline completo em uma imagem e retorna `PredictionResult` ou `None`."""
        validate_thresholds(confirm_threshold, unknown_threshold)

        img_bgr = read_image_bgr(image_path)
        if img_bgr is None:
            raise ValueError(f"Não foi possível decodificar a imagem: {image_path}")

        pre_res = segment_and_preprocess_cat(
            img_bgr=img_bgr,
            yolo_model=self.yolo_model,
            device=self.device,
            conf_threshold=yolo_conf,
            margin_ratio=margin_ratio,
            clahe_clip=clahe_clip,
            smooth_method=smooth_method,
        )
        if pre_res is None:
            return None

        feat_vec = self._extract_feature_vector(pre_res)
        X_query = feat_vec.reshape(1, -1)
        probs_raw = self.pipeline.predict_proba(X_query)[0]

        estimator_classes = getattr(self.pipeline, "classes_", np.arange(len(self.class_names)))
        ranked_probs = sorted(
            [
                (self.class_names[int(cls_idx)], float(prob))
                for cls_idx, prob in zip(estimator_classes, probs_raw)
            ],
            key=lambda item: item[1],
            reverse=True,
        )

        top_guess_label, max_prob = ranked_probs[0]
        pred_label, status, alert_flag = classify_confidence_tier(
            top_label=top_guess_label,
            max_prob=max_prob,
            confirm_threshold=confirm_threshold,
            unknown_threshold=unknown_threshold,
        )

        return PredictionResult(
            image_path=Path(image_path),
            pre_res=pre_res,
            ranked_probs=ranked_probs,
            top_guess_label=top_guess_label,
            pred_label=pred_label,
            max_prob=max_prob,
            status=status,
            alert_flag=alert_flag,
            confirm_threshold=float(confirm_threshold),
            unknown_threshold=float(unknown_threshold),
            scenario_label=self.scenario_label,
            classifier_name=self.classifier_name,
            true_label=true_label,
        )


def draw_annotated_original_rgb(pre_res: PreprocessResult, status: str) -> np.ndarray:
    """Desenha a bounding box colorida conforme o status e retorna em RGB."""
    style = STATUS_STYLES[status]
    orig_boxed = pre_res.original_bgr.copy()
    x1, y1, x2, y2 = pre_res.bbox_xyxy
    thickness = max(2, int(round(min(orig_boxed.shape[:2]) * 0.0055)))
    cv2.rectangle(orig_boxed, (x1, y1), (x2, y2), style["box_bgr"], thickness)
    return cv2.cvtColor(orig_boxed, cv2.COLOR_BGR2RGB)


def render_prediction_card(
    pred: PredictionResult,
    output_path: Path,
    dpi: int = 160,
) -> None:
    """Renderiza o card visual completo (Original | Segmentado+CLAHE | Veredito + Top-3)."""
    style = STATUS_STYLES[pred.status]
    orig_rgb = draw_annotated_original_rgb(pred.pre_res, pred.status)
    seg_rgb = cv2.cvtColor(pred.pre_res.normalized_bgr, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(17.8, 6.2), facecolor="white")
    # Coluna 2 (spacer=0.40) reserva folga dedicada aos rótulos longos do eixo Y do Top-3
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1.10, 0.98, 0.40, 1.18],
        height_ratios=[0.44, 0.56],
        hspace=0.30,
        wspace=0.06,
    )

    # Painel esquerdo: Foto original com bounding box detectada
    ax_orig = fig.add_subplot(gs[:, 0])
    ax_orig.imshow(orig_rgb)
    ax_orig.set_title(
        f"Foto Original (YOLO conf: {pred.pre_res.confidence:.1%})",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_orig.axis("off")

    # Painel central: Recorte segmentado com fundo removido + CLAHE
    ax_seg = fig.add_subplot(gs[:, 1])
    ax_seg.imshow(seg_rgb)
    ax_seg.set_title(
        "Recorte Segmentado + CLAHE\n(Entrada do Extrator)",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_seg.axis("off")

    # Painel direito superior: Cartão estruturado do veredito
    ax_verdict = fig.add_subplot(gs[0, 3])
    ax_verdict.set_xlim(0, 1)
    ax_verdict.set_ylim(0, 1)
    ax_verdict.axis("off")

    card_box = FancyBboxPatch(
        (0.02, 0.04),
        0.96,
        0.92,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=style["bg"],
        edgecolor=style["accent"],
        linewidth=2.2,
    )
    ax_verdict.add_patch(card_box)

    if pred.true_label is not None:
        outcome_str = "ACERTO" if pred.is_correct else "ERRO"
        status_line = f"STATUS: {style['badge']}  ({outcome_str})"
    else:
        status_line = f"STATUS: {style['badge']}"

    ax_verdict.text(
        0.50,
        0.82,
        status_line,
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=style["banner_color"],
    )

    if pred.status == STATUS_DESCONHECIDO:
        main_line = f"{UNKNOWN_LABEL} ({pred.max_prob:.1%})"
        detail_line = (
            f"Mais próximo: {pred.top_guess_label} ({pred.max_prob:.1%}) "
            f"< {pred.unknown_threshold:.0%}"
        )
    elif pred.status == STATUS_INCERTO:
        main_line = f"Palpite: {pred.pred_label} ({pred.max_prob:.1%})"
        detail_line = UNCERTAINTY_FLAG
    else:
        main_line = f"Identidade: {pred.pred_label} ({pred.max_prob:.1%})"
        detail_line = f"Alta confiança (>= {pred.confirm_threshold:.0%})"

    ax_verdict.text(
        0.50,
        0.54,
        main_line,
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111111",
    )
    ax_verdict.text(
        0.50,
        0.31,
        detail_line,
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold" if pred.status != STATUS_CONFIRMADO else "normal",
        color=style["banner_color"],
    )

    if pred.true_label is not None:
        ax_verdict.text(
            0.50,
            0.13,
            f"Rótulo real da pasta: {pred.true_label}",
            ha="center",
            va="center",
            fontsize=9.0,
            color="#37474f",
        )

    # Painel direito inferior: Gráfico horizontal com Top-3 probabilidades
    ax_bars = fig.add_subplot(gs[1, 3])
    top3 = pred.top3_probs
    names_plot = [
        (item[0][:24] + "…") if len(item[0]) > 25 else item[0]
        for item in reversed(top3)
    ]
    probs_plot = [item[1] * 100.0 for item in reversed(top3)]
    bar_colors = [
        style["accent"] if idx == len(top3) - 1 else "#607d8b"
        for idx in range(len(top3))
    ]

    bars = ax_bars.barh(
        names_plot,
        probs_plot,
        color=bar_colors,
        edgecolor="#222222",
        linewidth=0.6,
        height=0.58,
    )
    ax_bars.axvline(
        pred.unknown_threshold * 100.0,
        color="#c62828",
        linestyle="--",
        linewidth=1.4,
        label=f"Desconhecido (< {pred.unknown_threshold:.0%})",
    )
    ax_bars.axvline(
        pred.confirm_threshold * 100.0,
        color="#2e7d32",
        linestyle="-.",
        linewidth=1.4,
        label=f"Confirmado (>= {pred.confirm_threshold:.0%})",
    )
    ax_bars.set_xlim(0, 120)
    ax_bars.set_xlabel("Probabilidade (%)", fontsize=9.5, fontweight="bold")
    ax_bars.set_title("Top-3 Probabilidades", fontsize=10.5, fontweight="bold", pad=22)
    ax_bars.tick_params(axis="y", labelsize=9.0)
    ax_bars.grid(axis="x", linestyle=":", alpha=0.6)
    ax_bars.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8.2,
    )

    for bar in bars:
        w_val = bar.get_width()
        ax_bars.annotate(
            f"{w_val:.1f}%",
            xy=(w_val, bar.get_y() + bar.get_height() / 2.0),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    # Cabeçalho superior colorido
    if pred.status == STATUS_CONFIRMADO:
        status_tag = "[CONFIRMADO]"
    elif pred.status == STATUS_INCERTO:
        status_tag = UNCERTAINTY_FLAG
    else:
        status_tag = UNKNOWN_FLAG

    if pred.true_label is not None:
        banner = (
            f"Real: {pred.true_label} | Previsto: {pred.pred_label} "
            f"({pred.max_prob:.1%})   {status_tag}"
        )
    else:
        banner = (
            f"Previsto: {pred.pred_label} ({pred.max_prob:.1%})   {status_tag}"
        )

    fig.suptitle(
        banner,
        fontsize=13,
        fontweight="bold",
        color=style["banner_color"],
        y=0.96,
    )

    # Rodapé estruturado com Top-3 inline e metadados do modelo
    fig.text(
        0.50,
        0.025,
        f"Top-3: {pred.top3_inline}   •   Modelo: {pred.scenario_label} + {pred.classifier_name}",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#263238",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f7fa", edgecolor="#cfd8dc"),
    )

    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.025, right=0.975)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def format_prediction_terminal_summary(
    pred: PredictionResult,
    output_path: Path | None = None,
) -> str:
    """Formata o resumo textual estruturado da predição para impressão no terminal."""
    lines = [
        "",
        "=" * 72,
        f"RESULTADO DA IDENTIFICAÇÃO ({pred.scenario_label} | {pred.classifier_name})",
        "=" * 72,
    ]
    if pred.true_label is not None:
        lines.append(f"Classe Real    : {pred.true_label}")

    if pred.status == STATUS_CONFIRMADO:
        lines.append(f"Status         : CONFIRMADO (alta confiança >= {pred.confirm_threshold:.0%})")
        lines.append(f"Gato Previsto  : {pred.pred_label}")
        lines.append(f"Confiança      : {pred.max_prob:.2%}")
    elif pred.status == STATUS_INCERTO:
        lines.append(
            f"Status         : INCERTO {UNCERTAINTY_FLAG} "
            f"({pred.unknown_threshold:.0%} <= prob < {pred.confirm_threshold:.0%})"
        )
        lines.append(f"Palpite (Incerto): {pred.pred_label}")
        lines.append(f"Confiança      : {pred.max_prob:.2%}")
    else:
        lines.append(
            f"Status         : DESCONHECIDO {UNKNOWN_FLAG} "
            f"(prob < {pred.unknown_threshold:.0%})"
        )
        lines.append(f"Classificação  : {UNKNOWN_LABEL}")
        lines.append(f"Mais próximo   : {pred.top_guess_label} ({pred.max_prob:.2%})")

    lines.append(f"Top-3          : {pred.top3_inline}")
    if output_path is not None:
        lines.append(f"Card salvo em  : {output_path}")
    lines.append("=" * 72 + "\n")
    return "\n".join(lines)
