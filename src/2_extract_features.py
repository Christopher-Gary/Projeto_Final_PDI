"""Etapa 2: Data Augmentation (apenas treino) e extração híbrida de características.

Lê as imagens pré-processadas de:
    dataset/processed/train/<nome_do_gato>/
    dataset/processed/test/<nome_do_gato>/

Extrai dois vetores por imagem (ou por variação aumentada no treino):
  a) Descritores Clássicos de PDI (Baseline): Histograma HSV mascarado + LBP multiescala.
  b) Embeddings Profundos: DINOv2 (dinov2_vits14) pré-treinado em modo eval(), sem fine-tuning.

Salva todos os vetores e rótulos separados em outputs/features.npz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pdi_core import (
    DEFAULT_DINO_IMG_SIZE,
    DEFAULT_HSV_BINS,
    DEFAULT_LBP_CANVAS_SIZE,
    DEFAULT_LBP_CONFIGS,
    PROJECT_ROOT,
    DINOv2Extractor,
    augment_training_image,
    configure_stdio_utf8,
    extract_pdi_features,
    get_device,
    get_foreground_mask,
    list_class_dirs,
    list_images_in_dir,
    read_image_bgr,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    """Define e interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Etapa 2: Data Augmentation no treino e extração de features (PDI + DINOv2)."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "processed",
        help="Diretório com imagens já segmentadas e normalizadas (padrão: dataset/processed/).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "features.npz",
        help="Arquivo .npz de saída com features de treino e teste (padrão: outputs/features.npz).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do lote para inferência do DINOv2 (padrão: 32).",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Desativa o espelhamento horizontal (flip) no conjunto de treino.",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=10.0,
        help="Ângulo das pequenas rotações (±graus) aplicadas apenas no treino (padrão: 10.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo de execução: 'auto', 'cuda' ou 'cpu' (padrão: auto).",
    )
    return parser.parse_args()


def _extract_split_features(
    split_name: str,
    items: list[tuple[int, str, Path]],
    dino_extractor: DINOv2Extractor,
    batch_size: int,
    apply_augmentation: bool,
    enable_flip: bool,
    rotation_deg: float,
    logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extrai features PDI e embeddings DINOv2 em lotes para um split completo."""
    pdi_vectors: list[np.ndarray] = []
    dino_batches: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []

    pending_imgs: list[np.ndarray] = []

    def _flush_dino_buffer() -> None:
        if pending_imgs:
            feats = dino_extractor.extract_batch(pending_imgs, batch_size=batch_size)
            dino_batches.append(feats)
            pending_imgs.clear()

    rot_angles = (-rotation_deg, rotation_deg) if (apply_augmentation and rotation_deg > 0) else ()

    pbar = tqdm(items, desc=f"Features [{split_name}]", unit="img")
    for class_idx, cat_name, img_path in pbar:
        img_bgr = read_image_bgr(img_path)
        if img_bgr is None:
            logger.warning(f"Imagem ignorada por falha na leitura: {img_path}")
            continue

        fg_mask = get_foreground_mask(img_bgr, bg_threshold=0)
        if np.count_nonzero(fg_mask) == 0:
            logger.warning(f"Imagem totalmente preta sem primeiro plano detectável: {img_path}")
            continue

        if apply_augmentation:
            views = augment_training_image(
                img_bgr=img_bgr,
                fg_mask=fg_mask,
                enable_flip=enable_flip,
                rotation_angles=rot_angles,
            )
        else:
            views = [("orig", img_bgr, fg_mask)]

        for tag, view_bgr, view_mask in views:
            pdi_vec = extract_pdi_features(view_bgr, fg_mask=view_mask)
            pdi_vectors.append(pdi_vec)
            pending_imgs.append(view_bgr)
            labels.append(class_idx)
            sample_ids.append(f"{cat_name}/{img_path.name}#{tag}")

            if len(pending_imgs) >= batch_size:
                _flush_dino_buffer()

    _flush_dino_buffer()

    if not pdi_vectors:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 384), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=str),
        )

    X_pdi = np.vstack(pdi_vectors).astype(np.float32)
    X_dino = np.vstack(dino_batches).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    paths_arr = np.asarray(sample_ids)
    return X_pdi, X_dino, y, paths_arr


def main() -> None:
    """Executa a extração de características de treino (com augmentation) e teste (intacto)."""
    configure_stdio_utf8()
    logger = setup_logging()
    args = parse_args()

    processed_dir: Path = args.processed_dir.resolve()
    train_dir = processed_dir / "train"
    test_dir = processed_dir / "test"
    output_file: Path = args.output_file.resolve()

    if not train_dir.exists() or not test_dir.exists():
        logger.error(
            f"Estrutura esperada não encontrada em '{processed_dir}'. "
            "Execute primeiro 'python src/1_preprocess_pdi.py'."
        )
        sys.exit(1)

    train_class_dirs = list_class_dirs(train_dir)
    if not train_class_dirs:
        logger.error(f"Nenhuma pasta de classe encontrada em: {train_dir}")
        sys.exit(1)

    class_names = [d.name for d in train_class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    if len(class_names) < 2:
        logger.warning(
            f"Apenas {len(class_names)} classe encontrada ({class_names}). "
            "Para classificação Re-ID são necessárias pelo menos 2 classes de gatos."
        )

    train_items: list[tuple[int, str, Path]] = []
    for cat_dir in train_class_dirs:
        imgs = list_images_in_dir(cat_dir)
        for img_path in imgs:
            train_items.append((class_to_idx[cat_dir.name], cat_dir.name, img_path))

    test_items: list[tuple[int, str, Path]] = []
    for cat_dir in list_class_dirs(test_dir):
        if cat_dir.name not in class_to_idx:
            logger.warning(
                f"Pasta '{cat_dir.name}' existe em test/, mas não existe em train/. "
                "Essas imagens serão ignoradas na avaliação supervisionada."
            )
            continue
        imgs = list_images_in_dir(cat_dir)
        for img_path in imgs:
            test_items.append((class_to_idx[cat_dir.name], cat_dir.name, img_path))

    if not train_items or not test_items:
        logger.error(
            f"Conjuntos insuficientes: treino={len(train_items)} imgs, teste={len(test_items)} imgs."
        )
        sys.exit(1)

    device = get_device(args.device, logger=logger)
    logger.info("Carregando backbone DINOv2 pré-treinado (dinov2_vits14) em modo eval()...")
    dino_extractor = DINOv2Extractor(device=device, model_name="dinov2_vits14")

    logger.info(
        f"Extraindo treino ({len(train_items)} imagens base | "
        f"flip={'SIM' if not args.no_flip else 'NÃO'} | rotação=±{args.rotation_deg:.1f}°)"
    )
    X_train_pdi, X_train_dino, y_train, train_paths = _extract_split_features(
        split_name="train",
        items=train_items,
        dino_extractor=dino_extractor,
        batch_size=args.batch_size,
        apply_augmentation=True,
        enable_flip=not args.no_flip,
        rotation_deg=args.rotation_deg,
        logger=logger,
    )

    logger.info(f"Extraindo teste ({len(test_items)} imagens originais, sem augmentation)")
    X_test_pdi, X_test_dino, y_test, test_paths = _extract_split_features(
        split_name="test",
        items=test_items,
        dino_extractor=dino_extractor,
        batch_size=args.batch_size,
        apply_augmentation=False,
        enable_flip=False,
        rotation_deg=0.0,
        logger=logger,
    )

    if len(y_train) == 0 or len(y_test) == 0:
        logger.error("Não foi possível extrair vetores válidos para treino e/ou teste.")
        sys.exit(1)

    metadata = {
        "hsv_bins": list(DEFAULT_HSV_BINS),
        "lbp_configs": [list(cfg) for cfg in DEFAULT_LBP_CONFIGS],
        "lbp_canvas_size": DEFAULT_LBP_CANVAS_SIZE,
        "dino_model": "dinov2_vits14",
        "dino_img_size": DEFAULT_DINO_IMG_SIZE,
        "train_flip": not args.no_flip,
        "train_rotation_deg": float(args.rotation_deg),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        X_train_pdi=X_train_pdi,
        X_train_dino=X_train_dino,
        y_train=y_train,
        train_paths=train_paths,
        X_test_pdi=X_test_pdi,
        X_test_dino=X_test_dino,
        y_test=y_test,
        test_paths=test_paths,
        class_names=np.asarray(class_names),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )

    logger.info("=" * 66)
    logger.info("EXTRAÇÃO DE CARACTERÍSTICAS CONCLUÍDA (ETAPA 2)")
    logger.info("=" * 66)
    logger.info(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    logger.info(f"Treino PDI   : {X_train_pdi.shape} | Treino DINOv2: {X_train_dino.shape}")
    logger.info(f"Teste  PDI   : {X_test_pdi.shape}  | Teste  DINOv2: {X_test_dino.shape}")
    logger.info(f"Arquivo salvo: {output_file}")
    logger.info("=" * 66)


if __name__ == "__main__":
    main()
