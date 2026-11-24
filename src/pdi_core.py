"""Núcleo compartilhado de Processamento Digital de Imagens (PDI) e Visão Computacional.

Centraliza funções reutilizáveis entre os scripts de pré-processamento,
extração de características, treinamento e inferência individual para garantir
consistência matemática e zero duplicação de lógica no pipeline de Cat Re-ID.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib

# Usa backend não interativo para permitir geração de gráficos em servidores/WSL/headless
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import local_binary_pattern

# Diretório raiz do projeto (um nível acima de src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Extensões de imagem suportadas (verificação case-insensitive)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Índice da classe 'cat' no dataset COCO usado pelos modelos YOLO pré-treinados
CAT_COCO_CLASS_ID = 15

# Hiperparâmetros padrão dos descritores clássicos de PDI
# HSV: 12 x 4 x 6 = 288 bins (boa resolução de matiz e brilho para pelagens variadas)
DEFAULT_HSV_BINS = (12, 4, 6)
# LBP uniforme multiescala: (8,1)->10 + (16,2)->18 + (24,3)->26 = 54 bins (total PDI = 342 dims)
DEFAULT_LBP_CONFIGS = ((8, 1), (16, 2), (24, 3))
DEFAULT_LBP_CANVAS_SIZE = 224
DEFAULT_DINO_IMG_SIZE = 224


@dataclass
class PreprocessResult:
    """Resultado estruturado da segmentação e normalização PDI de um gato."""

    original_bgr: np.ndarray
    binary_mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    crop_mask: np.ndarray
    bg_removed_bgr: np.ndarray
    normalized_bgr: np.ndarray
    confidence: float
    mask_area_ratio: float


def configure_stdio_utf8() -> None:
    """Configura stdout e stderr para UTF-8 evitando erros de codificação no Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configura e retorna o logger padrão padronizado do projeto."""
    logger = logging.getLogger("cat_reid")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_device(prefer_device: str = "auto", logger: logging.Logger | None = None) -> str:
    """Detecta automaticamente o dispositivo de execução ('cuda' ou 'cpu')."""
    log = logger or logging.getLogger("cat_reid")
    prefer = prefer_device.strip().lower()

    try:
        import torch
    except ImportError:
        log.warning("PyTorch não encontrado ao detectar dispositivo; usando 'cpu'.")
        return "cpu"

    if prefer == "cpu":
        log.info("Dispositivo selecionado manualmente: CPU.")
        return "cpu"

    if prefer == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            log.info(f"Dispositivo selecionado: CUDA ({gpu_name}).")
            return "cuda"
        log.warning("CUDA solicitado, porém nenhuma GPU compatível foi detectada. Alternando para CPU.")
        return "cpu"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        log.info(f"Detecção automática de hardware: CUDA ativo ({gpu_name}).")
        return "cuda"

    log.info("Detecção automática de hardware: utilizando CPU.")
    return "cpu"


def list_class_dirs(root_dir: Path) -> list[Path]:
    """Lista subpastas de classes (nomes dos gatos) ordenadas, ignorando diretórios ocultos."""
    if not root_dir.exists() or not root_dir.is_dir():
        return []
    return sorted(
        [d for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda p: p.name.lower(),
    )


def list_images_in_dir(folder: Path) -> list[Path]:
    """Lista arquivos de imagem válidos em uma pasta (ordenados pelo nome)."""
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )


def read_image_bgr(path: Path) -> np.ndarray | None:
    """Lê uma imagem com suporte a caminhos Unicode (Windows) e correção de orientação EXIF."""
    try:
        with Image.open(path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback com np.fromfile para caminhos com acentos no Windows caso PIL falhe
        try:
            raw = np.fromfile(str(path), dtype=np.uint8)
            if raw.size == 0:
                return None
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None


def write_image(path: Path, img_bgr: np.ndarray) -> bool:
    """Salva uma imagem BGR com suporte seguro a caminhos Unicode no Windows/WSL."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix if path.suffix else ".png"
        ok, encoded = cv2.imencode(ext, img_bgr)
        if not ok:
            return False
        encoded.tofile(str(path))
        return True
    except Exception:
        return False


def load_yolo_seg_model(model_name: str = "yolo11n-seg.pt"):
    """Carrega o modelo de segmentação de instâncias da Ultralytics (YOLO)."""
    from ultralytics import YOLO

    return YOLO(model_name)


def apply_lab_clahe_and_smooth(
    crop_bgr: np.ndarray,
    crop_mask: np.ndarray,
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
    smooth_method: str = "bilateral",
) -> np.ndarray:
    """Normaliza iluminação no espaço LAB (CLAHE em L) e suaviza ruídos preservando fundo preto.

    Preenche temporariamente o fundo com a cor média do gato antes do CLAHE para
    evitar halos artificiais de contraste nas bordas da silhueta, e restaura o
    fundo estritamente para [0, 0, 0] após o processamento.
    """
    fg_bool = crop_mask > 0
    if not np.any(fg_bool):
        return np.zeros_like(crop_bgr, dtype=np.uint8)

    # Preenche fundo temporariamente com a média do primeiro plano para estabilizar tiles do CLAHE
    working = crop_bgr.copy()
    mean_fg_bgr = np.round(working[fg_bool].mean(axis=0)).astype(np.uint8)
    working[~fg_bool] = mean_fg_bgr

    # Conversão BGR -> LAB e equalização adaptativa (CLAHE) apenas na luminância (canal L)
    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=clahe_grid)
    l_eq = clahe.apply(l_chan)
    lab_eq = cv2.merge((l_eq, a_chan, b_chan))
    enhanced_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Suavização leve para atenuar ruído de sensor preservando texturas/listras da pelagem
    method = smooth_method.strip().lower()
    if method == "bilateral":
        smoothed_bgr = cv2.bilateralFilter(enhanced_bgr, d=5, sigmaColor=35, sigmaSpace=35)
    elif method == "gaussian":
        smoothed_bgr = cv2.GaussianBlur(enhanced_bgr, (3, 3), sigmaX=0.8)
    else:
        raise ValueError(f"Método de suavização inválido: '{smooth_method}'. Use 'bilateral' ou 'gaussian'.")

    # Garante fundo estritamente preto [0, 0, 0]
    output = smoothed_bgr.copy()
    output[~fg_bool] = 0

    # Evita que pixels pretos legítimos de gatos escuros sejam confundidos com fundo [0, 0, 0]
    black_fg = fg_bool & np.all(output == 0, axis=2)
    if np.any(black_fg):
        output[black_fg] = [1, 1, 1]

    return output


def segment_and_preprocess_cat(
    img_bgr: np.ndarray,
    yolo_model,
    device: str = "cpu",
    conf_threshold: float = 0.25,
    margin_ratio: float = 0.05,
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
    smooth_method: str = "bilateral",
) -> PreprocessResult | None:
    """Detecta e segmenta o gato principal via YOLO-Seg, remove o fundo e normaliza com PDI.

    Retorna None caso nenhum gato (classe COCO 15) seja detectado com confiança suficiente.
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return None

    results = yolo_model.predict(
        source=img_bgr,
        classes=[CAT_COCO_CLASS_ID],
        conf=float(conf_threshold),
        device=device,
        verbose=False,
    )
    if not results:
        return None

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0 or res.masks is None:
        return None

    best_mask: np.ndarray | None = None
    best_conf = 0.0
    best_area = 0
    best_score = -1.0

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    num_dets = len(res.boxes)

    for idx in range(num_dets):
        conf = float(res.boxes.conf[idx].item())
        mask = np.zeros((h, w), dtype=np.uint8)

        # Reconstrói máscara nas coordenadas originais a partir do polígono sempre que possível
        used_polygon = False
        if res.masks.xy is not None and idx < len(res.masks.xy):
            poly = res.masks.xy[idx]
            if poly is not None and len(poly) >= 3:
                pts = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
                used_polygon = True

        if not used_polygon and res.masks.data is not None and idx < len(res.masks.data):
            raw_mask = res.masks.data[idx].detach().cpu().numpy()
            resized_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = np.where(resized_mask > 0.5, 255, 0).astype(np.uint8)

        # Fechamento morfológico leve para remover microfuros internos na silhueta
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue

        # Combina confiança e área relativa para priorizar o gato principal da cena
        area_ratio = area / float(h * w)
        score = conf * area_ratio

        if score > best_score:
            best_score = score
            best_mask = mask
            best_conf = conf
            best_area = area

    if best_mask is None or best_area == 0:
        return None

    # Recorte pela bounding box da máscara com margem proporcional (padrão 5%)
    bx, by, bw, bh = cv2.boundingRect(best_mask)
    pad_x = int(round(bw * margin_ratio))
    pad_y = int(round(bh * margin_ratio))

    x1 = max(0, bx - pad_x)
    y1 = max(0, by - pad_y)
    x2 = min(w, bx + bw + pad_x)
    y2 = min(h, by + bh + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop_bgr = img_bgr[y1:y2, x1:x2].copy()
    crop_mask = best_mask[y1:y2, x1:x2].copy()

    bg_removed_bgr = crop_bgr.copy()
    bg_removed_bgr[crop_mask == 0] = 0

    normalized_bgr = apply_lab_clahe_and_smooth(
        crop_bgr=crop_bgr,
        crop_mask=crop_mask,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
        smooth_method=smooth_method,
    )

    return PreprocessResult(
        original_bgr=img_bgr,
        binary_mask=best_mask,
        bbox_xyxy=(x1, y1, x2, y2),
        crop_mask=crop_mask,
        bg_removed_bgr=bg_removed_bgr,
        normalized_bgr=normalized_bgr,
        confidence=best_conf,
        mask_area_ratio=best_area / float(h * w),
    )


def save_pdi_debug_panel(
    result: PreprocessResult,
    output_path: Path,
    title_prefix: str = "",
) -> None:
    """Gera painel comparativo 1x4 para inspeção e documentação no relatório técnico."""
    orig_annotated = result.original_bgr.copy()
    x1, y1, x2, y2 = result.bbox_xyxy
    thickness = max(2, int(round(min(orig_annotated.shape[:2]) * 0.005)))
    cv2.rectangle(orig_annotated, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    orig_rgb = cv2.cvtColor(orig_annotated, cv2.COLOR_BGR2RGB)
    bg_removed_rgb = cv2.cvtColor(result.bg_removed_bgr, cv2.COLOR_BGR2RGB)
    normalized_rgb = cv2.cvtColor(result.normalized_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = [
        (orig_rgb, "1. Original + BBox (+5%)", None),
        (result.binary_mask, "2. Máscara Binária (YOLO-Seg)", "gray"),
        (bg_removed_rgb, "3. Fundo Removido", None),
        (normalized_rgb, "4. Normalizado (LAB-CLAHE)", None),
    ]

    for ax, (img, subtitle, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.axis("off")

    prefix_str = f"{title_prefix} | " if title_prefix else ""
    fig.suptitle(
        f"{prefix_str}Confiança YOLO: {result.confidence:.1%} | Área da Máscara: {result.mask_area_ratio:.1%}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_foreground_mask(img_bgr: np.ndarray, bg_threshold: int = 0) -> np.ndarray:
    """Reconstrói a máscara binária de primeiro plano ignorando o fundo preto [0, 0, 0]."""
    fg_bool = np.any(img_bgr > bg_threshold, axis=2)
    return np.where(fg_bool, 255, 0).astype(np.uint8)


def rotate_image_and_mask(
    img_bgr: np.ndarray,
    fg_mask: np.ndarray,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotaciona imagem e máscara mantendo o fundo estritamente preto [0, 0, 0]."""
    h, w = img_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)

    rot_img = cv2.warpAffine(
        img_bgr,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rot_mask = cv2.warpAffine(
        fg_mask,
        rot_mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rot_img[rot_mask == 0] = 0
    return rot_img, rot_mask


def augment_training_image(
    img_bgr: np.ndarray,
    fg_mask: np.ndarray,
    enable_flip: bool = True,
    rotation_angles: Iterable[float] = (-10.0, 10.0),
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Gera variações leves (espelhamento horizontal e rotações) apenas para treino."""
    views: list[tuple[str, np.ndarray, np.ndarray]] = [("orig", img_bgr, fg_mask)]

    if enable_flip:
        flipped_img = cv2.flip(img_bgr, 1)
        flipped_mask = cv2.flip(fg_mask, 1)
        views.append(("flip", flipped_img, flipped_mask))

    for angle in rotation_angles:
        if abs(float(angle)) < 1e-6:
            continue
        rot_img, rot_mask = rotate_image_and_mask(img_bgr, fg_mask, float(angle))
        views.append((f"rot_{angle:+.0f}", rot_img, rot_mask))

    return views


def letterbox_square(
    img: np.ndarray,
    target_size: int = 224,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Redimensiona mantendo a proporção original e centraliza em um quadrado preto."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        if img.ndim == 2:
            return np.zeros((target_size, target_size), dtype=img.dtype)
        return np.zeros((target_size, target_size, img.shape[2]), dtype=img.dtype)

    scale = target_size / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Usa interpolação suave ao ampliar quando não for máscara discreta (INTER_NEAREST)
    interp = interpolation
    if scale > 1.0 and interpolation == cv2.INTER_AREA:
        interp = cv2.INTER_LINEAR

    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    if img.ndim == 2:
        canvas = np.zeros((target_size, target_size), dtype=img.dtype)
    else:
        canvas = np.zeros((target_size, target_size, img.shape[2]), dtype=img.dtype)

    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def extract_pdi_features(
    img_bgr: np.ndarray,
    fg_mask: np.ndarray | None = None,
    hsv_bins: tuple[int, int, int] = DEFAULT_HSV_BINS,
    lbp_configs: tuple[tuple[int, int], ...] = DEFAULT_LBP_CONFIGS,
    lbp_canvas_size: int = DEFAULT_LBP_CANVAS_SIZE,
) -> np.ndarray:
    """Extrai descritores clássicos de PDI: histograma HSV mascarado + LBP multiescala.

    Ignora totalmente pixels de fundo preto [0, 0, 0] via máscara binária para
    que cor e textura representem exclusivamente a pelagem do gato.
    """
    mask = fg_mask if fg_mask is not None else get_foreground_mask(img_bgr)
    if np.count_nonzero(mask) == 0:
        mask = np.full(img_bgr.shape[:2], 255, dtype=np.uint8)

    # 1. Histograma de cores 3D no espaço HSV (calculado somente na máscara do gato)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        mask,
        list(hsv_bins),
        [0, 180, 0, 256, 0, 256],
    ).flatten().astype(np.float32)

    hsv_hist /= hsv_hist.sum() + 1e-8
    hsv_hist /= np.linalg.norm(hsv_hist) + 1e-8

    # 2. Textura LBP (Local Binary Patterns) uniforme multiescala em escala de cinza
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_std = letterbox_square(gray, target_size=lbp_canvas_size, interpolation=cv2.INTER_AREA)
    mask_std = letterbox_square(mask, target_size=lbp_canvas_size, interpolation=cv2.INTER_NEAREST)

    lbp_parts: list[np.ndarray] = []
    for points, radius in lbp_configs:
        lbp_map = local_binary_pattern(gray_std, P=points, R=radius, method="uniform")

        # Erode levemente a máscara para excluir bordas artificiais entre pelo e fundo preto
        k_size = 2 * radius + 1
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        inner_mask = cv2.erode(mask_std, erode_kernel)
        if np.count_nonzero(inner_mask) < 16:
            inner_mask = mask_std

        valid_lbp = lbp_map[inner_mask > 0]
        if valid_lbp.size == 0:
            hist = np.zeros(points + 2, dtype=np.float32)
        else:
            hist_counts, _ = np.histogram(
                valid_lbp,
                bins=points + 2,
                range=(0, points + 2),
            )
            hist = hist_counts.astype(np.float32)
            hist /= hist.sum() + 1e-8
        lbp_parts.append(hist)

    lbp_hist = np.concatenate(lbp_parts).astype(np.float32)
    lbp_hist /= np.linalg.norm(lbp_hist) + 1e-8

    return np.concatenate([hsv_hist, lbp_hist]).astype(np.float32)


class DINOv2Extractor:
    """Extrator de embeddings visuais usando o backbone pré-treinado DINOv2 (ViT-S/14)."""

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "dinov2_vits14",
        img_size: int = DEFAULT_DINO_IMG_SIZE,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.model_name = model_name
        self.img_size = img_size

        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                trust_repo=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao carregar '{model_name}' via torch.hub. "
                "Verifique a conexão com a internet no primeiro uso para baixar os pesos "
                f"ou o cache local do Torch Hub. Erro original: {exc}"
            ) from exc

        self.model.eval()
        self.model.to(self.device)

    def _preprocess_batch(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Converte lista de imagens BGR em lote NCHW float32 normalizado para ImageNet."""
        batch_rgb = []
        for img_bgr in images_bgr:
            sq_bgr = letterbox_square(
                img_bgr,
                target_size=self.img_size,
                interpolation=cv2.INTER_AREA,
            )
            sq_rgb = cv2.cvtColor(sq_bgr, cv2.COLOR_BGR2RGB)
            batch_rgb.append(sq_rgb)

        arr = np.stack(batch_rgb, axis=0).astype(np.float32) / 255.0
        arr = np.transpose(arr, (0, 3, 1, 2))
        arr = (arr - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return arr

    def extract_batch(self, images_bgr: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Extrai embeddings DINOv2 para uma lista de imagens em mini-lotes."""
        if not images_bgr:
            return np.empty((0, 384), dtype=np.float32)

        embeddings: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(images_bgr), batch_size):
                chunk = images_bgr[start : start + batch_size]
                batch_np = self._preprocess_batch(chunk)
                tensor = self.torch.from_numpy(batch_np).to(self.device, non_blocking=True)
                feats = self.model(tensor)
                embeddings.append(feats.detach().cpu().numpy().astype(np.float32))

        return np.vstack(embeddings)

    def extract_single(self, img_bgr: np.ndarray) -> np.ndarray:
        """Extrai o vetor de embedding DINOv2 para uma única imagem BGR."""
        return self.extract_batch([img_bgr], batch_size=1)[0]
