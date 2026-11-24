"""Etapa 4: Script de inferência individual (Cat Re-ID) para uma nova foto.

Uso:
    python src/4_predict_single.py --image caminho/foto.jpg

Executa todo o pipeline de ponta a ponta (reutilizando `inference_utils.py`):
  1. Detecção e segmentação do gato principal com YOLO-Seg.
  2. Remoção completa do fundo ([0, 0, 0]) e recorte com margem de 5%.
  3. Normalização luminosa (CLAHE no canal L do LAB) + suavização.
  4. Extração das features compatíveis com o modelo exportado (PDI, DINOv2 ou Híbrido).
  5. Regra de decisão em 3 faixas de confiança:
       - >= 50%: CONFIRMADO (verde)
       - 40% a 50%: INCERTO [ALERTA: BAIXA CONFIANÇA / INCERTEZA] (laranja)
       - < 40%: Gato Desconhecido (vermelho)
  6. Geração do card visual em outputs/prediction_result.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference_utils import (
    DEFAULT_CONFIRM_THRESHOLD,
    DEFAULT_UNKNOWN_THRESHOLD,
    CatReIDPredictor,
    format_prediction_terminal_summary,
    render_prediction_card,
    validate_thresholds,
)
from pdi_core import PROJECT_ROOT, configure_stdio_utf8, setup_logging


def parse_args() -> argparse.Namespace:
    """Define e interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Etapa 4: Inferência individual de Cat Re-ID com regra de 3 faixas."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Caminho da imagem de entrada a ser identificada.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "cat_reid_model.pkl",
        help="Caminho do modelo treinado (.pkl) gerado na Etapa 3.",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo11n-seg.pt",
        help="Modelo YOLO-Seg usado na segmentação (padrão: yolo11n-seg.pt).",
    )
    parser.add_argument(
        "--confirm-threshold",
        "--threshold",
        dest="confirm_threshold",
        type=float,
        default=DEFAULT_CONFIRM_THRESHOLD,
        help="Limiar para identificação confirmada sem alerta (padrão: 0.50).",
    )
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=DEFAULT_UNKNOWN_THRESHOLD,
        help="Limiar mínimo abaixo do qual classifica como 'Gato Desconhecido' (padrão: 0.40).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Limiar de detecção do YOLO para classe 'cat' (padrão: 0.25).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Margem proporcional aplicada à bounding box do recorte (padrão: 0.05).",
    )
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        help="Parâmetro clipLimit do CLAHE no canal L (padrão: 2.0).",
    )
    parser.add_argument(
        "--smooth",
        type=str,
        choices=["bilateral", "gaussian"],
        default="bilateral",
        help="Filtro de suavização pós-CLAHE (padrão: bilateral).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "prediction_result.png",
        help="Caminho da imagem visual de resultado (padrão: outputs/prediction_result.png).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo de execução: 'auto', 'cuda' ou 'cpu' (padrão: auto).",
    )
    return parser.parse_args()


def main() -> None:
    """Executa inferência completa para uma foto individual."""
    configure_stdio_utf8()
    logger = setup_logging()
    args = parse_args()

    image_path: Path = args.image.resolve()
    model_path: Path = args.model.resolve()
    output_path: Path = args.output.resolve()

    if not image_path.exists() or not image_path.is_file():
        logger.error(f"Imagem de entrada não encontrada: {image_path}")
        sys.exit(1)

    try:
        validate_thresholds(args.confirm_threshold, args.unknown_threshold)
        predictor = CatReIDPredictor(
            model_path=model_path,
            yolo_model_name=args.yolo_model,
            device=args.device,
            logger=logger,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(f"Executando inferência na imagem '{image_path.name}'...")
    try:
        pred = predictor.predict_image(
            image_path=image_path,
            true_label=None,
            confirm_threshold=args.confirm_threshold,
            unknown_threshold=args.unknown_threshold,
            yolo_conf=args.conf,
            margin_ratio=args.margin,
            clahe_clip=args.clahe_clip,
            smooth_method=args.smooth,
        )
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if pred is None:
        logger.warning(
            f"Nenhum gato detectado na imagem '{image_path}' com confiança >= {args.conf:.2f}. "
            "Tente reduzir '--conf' ou utilizar outra foto mais nítida."
        )
        sys.exit(2)

    render_prediction_card(pred=pred, output_path=output_path, dpi=180)
    print(format_prediction_terminal_summary(pred=pred, output_path=output_path))


if __name__ == "__main__":
    main()
