"""Etapa 1: Detecção, segmentação do gato, remoção de fundo e normalização PDI.

Percorre dinamicamente as pastas:
    dataset/train/<nome_do_gato>/
    dataset/test/<nome_do_gato>/

Para cada imagem:
1. Detecta o gato (classe COCO 15) com YOLO-Seg, escolhendo o indivíduo principal.
2. Gera máscara binária da silhueta, remove o fundo ([0, 0, 0]) e recorta com margem de 5%.
3. Converte para LAB, aplica CLAHE no canal L e suaviza ruídos (Bilateral/Gaussiano).
4. Salva o resultado em dataset/processed/<split>/<nome_do_gato>/*.png.
5. Exporta painéis comparativos de inspeção em outputs/debug_pdi/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pdi_core import (
    PROJECT_ROOT,
    configure_stdio_utf8,
    get_device,
    list_class_dirs,
    list_images_in_dir,
    load_yolo_seg_model,
    read_image_bgr,
    save_pdi_debug_panel,
    segment_and_preprocess_cat,
    setup_logging,
    write_image,
)


def parse_args() -> argparse.Namespace:
    """Define e interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Etapa 1: Segmentação YOLO-Seg + remoção de fundo + CLAHE em LAB."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Diretório raiz contendo as subpastas train/ e test/ (padrão: dataset/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "processed",
        help="Diretório de saída das imagens normalizadas (padrão: dataset/processed/).",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "debug_pdi",
        help="Diretório para painéis comparativos de PDI (padrão: outputs/debug_pdi/).",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo11n-seg.pt",
        help="Modelo YOLO de segmentação (ex.: yolo11n-seg.pt ou yolov8n-seg.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Limiar mínimo de confiança do YOLO para classe 'cat' (padrão: 0.25).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Margem proporcional adicionada à bounding box no recorte (padrão: 0.05).",
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
        "--max-debug-per-class",
        type=int,
        default=2,
        help="Máximo de painéis de depuração salvos por gato/split (padrão: 2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo de inferência: 'auto', 'cuda' ou 'cpu' (padrão: auto).",
    )
    return parser.parse_args()


def _unique_stem(base_stem: str, used_stems: set[str]) -> str:
    """Garante nome de arquivo único quando há extensões diferentes com mesmo stem."""
    if base_stem not in used_stems:
        used_stems.add(base_stem)
        return base_stem
    counter = 2
    while f"{base_stem}_{counter}" in used_stems:
        counter += 1
    candidate = f"{base_stem}_{counter}"
    used_stems.add(candidate)
    return candidate


def main() -> None:
    """Executa o pré-processamento PDI completo para os conjuntos train e test."""
    configure_stdio_utf8()
    logger = setup_logging()
    args = parse_args()

    dataset_dir: Path = args.dataset_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    debug_dir: Path = args.debug_dir.resolve()

    if not dataset_dir.exists():
        logger.error(f"Diretório de dataset não encontrado: {dataset_dir}")
        sys.exit(1)

    splits = ("train", "test")
    split_tasks: dict[str, list[tuple[str, list[Path]]]] = {}
    total_found_images = 0

    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            logger.warning(f"Subpasta '{split}/' ausente em {dataset_dir}.")
            split_tasks[split] = []
            continue

        class_dirs = list_class_dirs(split_dir)
        tasks_for_split: list[tuple[str, list[Path]]] = []
        for cat_dir in class_dirs:
            imgs = list_images_in_dir(cat_dir)
            if not imgs:
                logger.warning(f"Nenhuma imagem válida encontrada em: {cat_dir}")
                continue
            tasks_for_split.append((cat_dir.name, imgs))
            total_found_images += len(imgs)

        split_tasks[split] = tasks_for_split

    if total_found_images == 0:
        logger.error(
            "Nenhuma imagem encontrada nas pastas dataset/train/<gato>/ ou dataset/test/<gato>/. "
            "Adicione as fotos dos gatos antes de executar este script."
        )
        sys.exit(1)

    device = get_device(args.device, logger=logger)
    logger.info(f"Carregando modelo YOLO de segmentação: {args.yolo_model} ...")
    yolo_model = load_yolo_seg_model(args.yolo_model)

    debug_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}

    for split in splits:
        tasks = split_tasks[split]
        if not tasks:
            summary[split] = {"scanned": 0, "processed": 0, "skipped_no_cat": 0, "skipped_error": 0}
            continue

        split_items = [
            (cat_name, img_path)
            for cat_name, img_paths in tasks
            for img_path in img_paths
        ]
        scanned = len(split_items)
        processed = 0
        skipped_no_cat = 0
        skipped_error = 0

        used_stems_by_cat: dict[str, set[str]] = {}
        debug_counts_by_cat: dict[str, int] = {}

        logger.info(
            f"Processando split '{split}' ({len(tasks)} gato(s), {scanned} imagens)..."
        )

        pbar = tqdm(split_items, desc=f"PDI [{split}]", unit="img")
        for cat_name, img_path in pbar:
            try:
                img_bgr = read_image_bgr(img_path)
                if img_bgr is None:
                    logger.warning(f"Não foi possível ler a imagem (arquivo corrompido/inválido): {img_path}")
                    skipped_error += 1
                    continue

                result = segment_and_preprocess_cat(
                    img_bgr=img_bgr,
                    yolo_model=yolo_model,
                    device=device,
                    conf_threshold=args.conf,
                    margin_ratio=args.margin,
                    clahe_clip=args.clahe_clip,
                    smooth_method=args.smooth,
                )

                if result is None:
                    logger.warning(
                        f"Nenhum gato detectado pelo YOLO (conf >= {args.conf:.2f}) em '{img_path}'. "
                        "Imagem ignorada."
                    )
                    skipped_no_cat += 1
                    continue

                stems = used_stems_by_cat.setdefault(cat_name, set())
                out_stem = _unique_stem(img_path.stem, stems)

                # Salva em PNG sem perda para preservar exatamente fundo [0, 0, 0] e textura
                out_path = output_dir / split / cat_name / f"{out_stem}.png"
                if not write_image(out_path, result.normalized_bgr):
                    logger.warning(f"Falha ao salvar imagem processada: {out_path}")
                    skipped_error += 1
                    continue

                processed += 1

                # Gera exemplos comparativos para o relatório técnico
                cur_debug = debug_counts_by_cat.get(cat_name, 0)
                if cur_debug < args.max_debug_per_class:
                    debug_path = debug_dir / f"{split}_{cat_name}_{out_stem}_pdi.png"
                    save_pdi_debug_panel(
                        result=result,
                        output_path=debug_path,
                        title_prefix=f"[{split.upper()}] {cat_name} ({img_path.name})",
                    )
                    debug_counts_by_cat[cat_name] = cur_debug + 1

            except Exception as exc:
                logger.warning(f"Erro inesperado ao processar '{img_path}': {exc}")
                skipped_error += 1
                continue

        summary[split] = {
            "scanned": scanned,
            "processed": processed,
            "skipped_no_cat": skipped_no_cat,
            "skipped_error": skipped_error,
        }

    # Resumo final detalhado
    logger.info("=" * 66)
    logger.info("RESUMO DO PRÉ-PROCESSAMENTO PDI (ETAPA 1)")
    logger.info("=" * 66)
    for split in splits:
        st = summary[split]
        logger.info(
            f"  {split.upper():<5} | Total: {st['scanned']:>4} | "
            f"Processadas: {st['processed']:>4} | "
            f"Sem gato (YOLO): {st['skipped_no_cat']:>3} | "
            f"Erros leitura/escrita: {st['skipped_error']:>3}"
        )
    logger.info(f"Imagens processadas salvas em: {output_dir}")
    logger.info(f"Painéis comparativos salvos em: {debug_dir}")
    logger.info("=" * 66)


if __name__ == "__main__":
    main()
