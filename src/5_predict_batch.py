"""Etapa 5: Inferência em lote sobre pastas de teste (`dataset/test/<nome_do_gato>/`).

Uso:
    python src/5_predict_batch.py
    python src/5_predict_batch.py --test_dir dataset/test2

Funcionalidades:
  - Percorre dinamicamente todas as subpastas de gatos em `--test_dir`.
  - Executa exatamente o mesmo pipeline completo da inferência individual
    (YOLO-Seg -> remoção de fundo -> LAB-CLAHE -> features -> classificador -> 3 faixas).
  - Salva os cards individuais em `outputs/batch_predictions/` com rótulo real e previsto.
  - Copia automaticamente os casos de erro ou `Gato Desconhecido` para
    `outputs/batch_predictions/erros/` para análise de falhas.
  - Imprime resumo quantitativo completo e gera o mosaico `outputs/batch_summary_grid.png`
    com até 12 exemplos variados para relatório e slides.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference_utils import (
    DEFAULT_CONFIRM_THRESHOLD,
    DEFAULT_UNKNOWN_THRESHOLD,
    STATUS_CONFIRMADO,
    STATUS_DESCONHECIDO,
    STATUS_INCERTO,
    CatReIDPredictor,
    PredictionResult,
    draw_annotated_original_rgb,
    render_prediction_card,
    sanitize_filename_part,
    validate_thresholds,
)
from pdi_core import (
    PROJECT_ROOT,
    configure_stdio_utf8,
    letterbox_square,
    list_class_dirs,
    list_images_in_dir,
    setup_logging,
)


@dataclass
class GridTileRecord:
    """Registro leve para montagem do mosaico resumo sem reter imagens em alta resolução."""

    pair_rgb: np.ndarray
    true_label: str
    pred_label: str
    top_guess_label: str
    max_prob: float
    status: str
    category: str  # 'acerto_confirmado' | 'acerto_alerta' | 'desconhecido' | 'erro'
    top3_probs: list[tuple[str, float]]


def parse_args() -> argparse.Namespace:
    """Define e interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Etapa 5: Inferência em lote de Cat Re-ID com análise de erros e mosaico."
    )
    parser.add_argument(
        "--test_dir",
        "--test-dir",
        dest="test_dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "test",
        help="Diretório contendo subpastas <nome_do_gato>/ (padrão: dataset/test).",
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
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch_predictions",
        help="Pasta onde serão salvos os cards individuais (padrão: outputs/batch_predictions/).",
    )
    parser.add_argument(
        "--summary_grid",
        "--summary-grid",
        dest="summary_grid",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch_summary_grid.png",
        help="Caminho do mosaico resumo final (padrão: outputs/batch_summary_grid.png).",
    )
    parser.add_argument(
        "--max-grid-items",
        type=int,
        default=12,
        help="Número máximo de exemplos variados exibidos no mosaico (padrão: 12).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo de execução: 'auto', 'cuda' ou 'cpu' (padrão: auto).",
    )
    return parser.parse_args()


def _clean_previous_pngs(folder: Path) -> None:
    """Remove cards .png de execuções anteriores para evitar mistura de lotes."""
    if not folder.exists():
        return
    for old_png in folder.glob("*.png"):
        try:
            old_png.unlink()
        except OSError:
            pass


def _unique_card_filename(
    true_label: str,
    pred_label: str,
    max_prob: float,
    img_stem: str,
    used_names: set[str],
) -> str:
    """Gera nome organizado e único para cada card salvo em batch_predictions/."""
    true_slug = sanitize_filename_part(true_label, max_len=36)
    pred_slug = sanitize_filename_part(pred_label, max_len=36)
    stem_slug = sanitize_filename_part(img_stem, max_len=45)
    conf_pct = int(round(max_prob * 100.0))

    base = f"{true_slug}__pred_{pred_slug}_{conf_pct:02d}pct_{stem_slug}"
    candidate = f"{base}.png"
    counter = 2
    while candidate in used_names:
        candidate = f"{base}_{counter}.png"
        counter += 1
    used_names.add(candidate)
    return candidate


def _make_pair_thumbnail(pred: PredictionResult, tile_size: int = 200) -> np.ndarray:
    """Cria miniatura lado a lado (Original+BBox | Segmentado+CLAHE) para o mosaico."""
    orig_rgb = draw_annotated_original_rgb(pred.pre_res, pred.status)
    seg_rgb = cv2.cvtColor(pred.pre_res.normalized_bgr, cv2.COLOR_BGR2RGB)

    orig_sq = letterbox_square(orig_rgb, target_size=tile_size, interpolation=cv2.INTER_AREA)
    seg_sq = letterbox_square(seg_rgb, target_size=tile_size, interpolation=cv2.INTER_AREA)

    divider = np.full((tile_size, 6, 3), 240, dtype=np.uint8)
    return np.hstack([orig_sq, divider, seg_sq])


def _select_diverse_grid_items(
    records: list[GridTileRecord],
    max_items: int,
) -> list[GridTileRecord]:
    """Seleciona até `max_items` exemplos variados (erros, alertas, desconhecidos e acertos)."""
    if max_items <= 0 or not records:
        return []

    by_cat: dict[str, list[GridTileRecord]] = {
        "erro": [],
        "acerto_alerta": [],
        "desconhecido": [],
        "acerto_confirmado": [],
    }
    for rec in records:
        by_cat[rec.category].append(rec)

    def _pick_diverse_from_bucket(bucket: list[GridTileRecord], quota: int) -> list[GridTileRecord]:
        picked: list[GridTileRecord] = []
        seen_classes: set[str] = set()
        # Primeiro prioriza diversidade de pastas/classes reais
        for item in bucket:
            if len(picked) >= quota:
                break
            if item.true_label not in seen_classes:
                picked.append(item)
                seen_classes.add(item.true_label)
        # Completa a cota caso ainda haja vagas e itens disponíveis
        if len(picked) < quota:
            picked_ids = {id(x) for x in picked}
            for item in bucket:
                if len(picked) >= quota:
                    break
                if id(item) not in picked_ids:
                    picked.append(item)
                    picked_ids.add(id(item))
        return picked

    selected: list[GridTileRecord] = []
    selected_ids: set[int] = set()

    # Reserva espaço para casos interessantes (erros, alertas 40-50%, desconhecidos <40%)
    for cat, cap in (("erro", 3), ("acerto_alerta", 3), ("desconhecido", 3)):
        quota = min(cap, max_items - len(selected))
        if quota <= 0:
            break
        for item in _pick_diverse_from_bucket(by_cat[cat], quota):
            selected.append(item)
            selected_ids.add(id(item))

    # Preenche o restante com acertos confirmados distribuídos entre diferentes gatos
    remaining = max_items - len(selected)
    if remaining > 0:
        for item in _pick_diverse_from_bucket(by_cat["acerto_confirmado"], remaining):
            selected.append(item)
            selected_ids.add(id(item))

    # Se ainda sobrar espaço (ex.: poucos confirmados), preenche com quaisquer itens restantes
    if len(selected) < max_items:
        for cat in ("erro", "acerto_alerta", "desconhecido", "acerto_confirmado"):
            for item in by_cat[cat]:
                if len(selected) >= max_items:
                    break
                if id(item) not in selected_ids:
                    selected.append(item)
                    selected_ids.add(id(item))

    return selected


def _save_summary_grid(
    selected: list[GridTileRecord],
    stats: dict[str, int],
    summary_path: Path,
) -> None:
    """Renderiza e salva o mosaico resumo com até 12 casos variados."""
    if not selected:
        return

    n_items = len(selected)
    ncols = min(4, max(1, n_items))
    nrows = math.ceil(n_items / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.35 * nrows + 0.95),
        facecolor="white",
        squeeze=False,
    )

    frame_styles = {
        "acerto_confirmado": ("#2e7d32", "#1b5e20", "ACERTO"),
        "acerto_alerta": ("#f57c00", "#e65100", "ALERTA 40-50%"),
        "desconhecido": ("#c62828", "#b71c1c", "DESCONHECIDO"),
        "erro": ("#c62828", "#b71c1c", "ERRO"),
    }

    flat_axes = axes.flatten()
    for idx, ax in enumerate(flat_axes):
        if idx >= n_items:
            ax.axis("off")
            continue

        rec = selected[idx]
        border_col, text_col, tag = frame_styles[rec.category]

        ax.imshow(rec.pair_rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(3.0)

        pred_display = (
            f"Desconhecido ({rec.max_prob:.0%}, ~{rec.top_guess_label})"
            if rec.status == STATUS_DESCONHECIDO
            else f"{rec.pred_label} ({rec.max_prob:.1%})"
        )
        ax.set_title(
            f"Real: {rec.true_label}\nPrev: {pred_display} [{tag}]",
            fontsize=8.6,
            fontweight="bold",
            color=text_col,
            pad=6,
        )

        top2_str = " | ".join(
            f"{rank}. {name} ({prob * 100.0:.0f}%)"
            for rank, (name, prob) in enumerate(rec.top3_probs[:2], start=1)
        )
        ax.set_xlabel(top2_str, fontsize=7.8, color="#37474f", labelpad=5)

    fig.suptitle(
        f"Resumo em Lote (Cat Re-ID) — {stats['processed']} avaliadas | "
        f"Confirmados (>=50%): {stats['acertos_confirmados']} | "
        f"Alertas (40–50%): {stats['acertos_alerta']} | "
        f"Desconhecidos (<40%): {stats['desconhecidos']} | "
        f"Erros: {stats['erros']}",
        fontsize=12.5,
        fontweight="bold",
        color="#111111",
        y=0.975,
    )
    plt.tight_layout(rect=(0.01, 0.01, 0.99, 0.945))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(summary_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Executa inferência em lote sobre todas as subpastas de teste."""
    configure_stdio_utf8()
    logger = setup_logging()
    args = parse_args()

    test_dir: Path = args.test_dir.resolve()
    model_path: Path = args.model.resolve()
    output_dir: Path = args.output_dir.resolve()
    erros_dir: Path = output_dir / "erros"
    summary_grid_path: Path = args.summary_grid.resolve()

    if not test_dir.exists() or not test_dir.is_dir():
        logger.error(f"Diretório de teste não encontrado: {test_dir}")
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

    class_dirs = list_class_dirs(test_dir)
    tasks: list[tuple[str, Path]] = []
    unseen_folders: list[str] = []

    for cat_dir in class_dirs:
        imgs = list_images_in_dir(cat_dir)
        if not imgs:
            logger.info(f"Pasta vazia ignorada: {cat_dir.name}/")
            continue
        if cat_dir.name not in predictor.class_names:
            unseen_folders.append(cat_dir.name)
        for img_path in imgs:
            tasks.append((cat_dir.name, img_path))

    if not tasks:
        logger.error(f"Nenhuma imagem válida encontrada nas subpastas de '{test_dir}'.")
        sys.exit(1)

    if unseen_folders:
        logger.warning(
            "Subpastas sem correspondência nas classes treinadas (úteis para testar detecção "
            f"de desconhecidos): {', '.join(unseen_folders)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    erros_dir.mkdir(parents=True, exist_ok=True)
    _clean_previous_pngs(output_dir)
    _clean_previous_pngs(erros_dir)

    stats: dict[str, int] = {
        "total_scanned": len(tasks),
        "processed": 0,
        "acertos_confirmados": 0,
        "acertos_alerta": 0,
        "desconhecidos": 0,
        "erros": 0,
        "erros_confirmados": 0,
        "erros_alerta": 0,
        "skipped_no_cat": 0,
        "skipped_read_error": 0,
        "copied_to_erros": 0,
        "known_total": 0,
        "known_hits": 0,
        "unseen_total": 0,
        "unseen_rejected": 0,
        "unseen_false_accept": 0,
    }

    used_names: set[str] = set()
    grid_records: list[GridTileRecord] = []
    known_class_set = set(predictor.class_names)

    logger.info(
        f"Iniciando inferência em lote em '{test_dir}' ({len(tasks)} imagens | "
        f"Confirmado >= {args.confirm_threshold:.0%} | Desconhecido < {args.unknown_threshold:.0%})..."
    )

    pbar = tqdm(tasks, desc="Batch Re-ID", unit="img")
    for true_label, img_path in pbar:
        try:
            pred = predictor.predict_image(
                image_path=img_path,
                true_label=true_label,
                confirm_threshold=args.confirm_threshold,
                unknown_threshold=args.unknown_threshold,
                yolo_conf=args.conf,
                margin_ratio=args.margin,
                clahe_clip=args.clahe_clip,
                smooth_method=args.smooth,
            )
        except ValueError as exc:
            logger.warning(f"Falha na leitura da imagem '{img_path}': {exc}")
            stats["skipped_read_error"] += 1
            continue
        except Exception as exc:
            logger.warning(f"Erro inesperado ao processar '{img_path}': {exc}")
            stats["skipped_read_error"] += 1
            continue

        if pred is None:
            logger.warning(
                f"Nenhum gato detectado pelo YOLO (conf >= {args.conf:.2f}) em '{img_path}'. Ignorada."
            )
            stats["skipped_no_cat"] += 1
            continue

        stats["processed"] += 1
        is_known_folder = true_label in known_class_set

        if is_known_folder:
            stats["known_total"] += 1
        else:
            stats["unseen_total"] += 1

        if pred.status == STATUS_DESCONHECIDO:
            stats["desconhecidos"] += 1
            category = "desconhecido"
            if not is_known_folder:
                stats["unseen_rejected"] += 1
        elif pred.pred_label == true_label:
            if is_known_folder:
                stats["known_hits"] += 1
            if pred.status == STATUS_CONFIRMADO:
                stats["acertos_confirmados"] += 1
                category = "acerto_confirmado"
            else:
                stats["acertos_alerta"] += 1
                category = "acerto_alerta"
        else:
            stats["erros"] += 1
            category = "erro"
            if pred.status == STATUS_CONFIRMADO:
                stats["erros_confirmados"] += 1
            else:
                stats["erros_alerta"] += 1
            if not is_known_folder:
                stats["unseen_false_accept"] += 1

        card_name = _unique_card_filename(
            true_label=true_label,
            pred_label=pred.pred_label,
            max_prob=pred.max_prob,
            img_stem=img_path.stem,
            used_names=used_names,
        )
        card_path = output_dir / card_name
        render_prediction_card(pred=pred, output_path=card_path, dpi=150)

        # Salva cópia em erros/ caso erre a classe OU caia em Gato Desconhecido
        if pred.pred_label != true_label or pred.status == STATUS_DESCONHECIDO:
            shutil.copy2(card_path, erros_dir / card_name)
            stats["copied_to_erros"] += 1

        grid_records.append(
            GridTileRecord(
                pair_rgb=_make_pair_thumbnail(pred, tile_size=200),
                true_label=true_label,
                pred_label=pred.pred_label,
                top_guess_label=pred.top_guess_label,
                max_prob=pred.max_prob,
                status=pred.status,
                category=category,
                top3_probs=pred.top3_probs,
            )
        )

    if stats["processed"] == 0:
        logger.error("Nenhuma imagem pôde ser segmentada/avaliada no conjunto de teste.")
        sys.exit(1)

    selected_tiles = _select_diverse_grid_items(grid_records, max_items=args.max_grid_items)
    _save_summary_grid(selected=selected_tiles, stats=stats, summary_path=summary_grid_path)

    total_acertos = stats["acertos_confirmados"] + stats["acertos_alerta"]
    acc_total = total_acertos / float(stats["processed"])

    print("\n" + "=" * 78)
    print(f"RESUMO QUANTITATIVO DA INFERÊNCIA EM LOTE ({test_dir.name}/)")
    print("=" * 78)
    print(f"Modelo utilizado                  : {predictor.scenario_label} + {predictor.classifier_name}")
    print(f"Total de arquivos encontrados     : {stats['total_scanned']}")
    print(f"Total processado (com gato válido): {stats['processed']}")
    print(f"Ignoradas (sem gato pelo YOLO)    : {stats['skipped_no_cat']}")
    print(f"Ignoradas (erro de leitura)       : {stats['skipped_read_error']}")
    print("-" * 78)
    print(f"Acertos CONFIRMADOS (>= {args.confirm_threshold:.0%})      : {stats['acertos_confirmados']}")
    print(
        f"Acertos c/ ALERTA ({args.unknown_threshold:.0%}–{args.confirm_threshold:.0%})      : "
        f"{stats['acertos_alerta']}"
    )
    print(f"Gato Desconhecido (< {args.unknown_threshold:.0%})        : {stats['desconhecidos']}")
    print(
        f"Erros de classe (trocou o gato)   : {stats['erros']} "
        f"(confirmados: {stats['erros_confirmados']} | alerta 40–50%: {stats['erros_alerta']})"
    )
    print(f"Acurácia geral (sobre processadas): {acc_total:.2%} ({total_acertos}/{stats['processed']})")

    if stats["known_total"] > 0 and stats["unseen_total"] > 0:
        acc_known = stats["known_hits"] / float(stats["known_total"])
        rej_rate = stats["unseen_rejected"] / float(stats["unseen_total"])
        print("-" * 78)
        print(
            f"  • Apenas pastas conhecidas      : {acc_known:.2%} "
            f"({stats['known_hits']}/{stats['known_total']})"
        )
        print(
            f"  • Pastas inéditas (ex.: fakes)  : {stats['unseen_rejected']}/{stats['unseen_total']} "
            f"rejeitadas como Desconhecido ({rej_rate:.2%}) | "
            f"{stats['unseen_false_accept']} falsos aceites"
        )

    print("-" * 78)
    print(f"Cards individuais salvos em       : {output_dir}")
    print(f"Cópias de erros/desconhecidos     : {erros_dir} ({stats['copied_to_erros']} arquivos)")
    print(f"Mosaico resumo (até {args.max_grid_items} itens)     : {summary_grid_path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
