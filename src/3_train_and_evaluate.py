"""Etapa 3: Treinamento, avaliação comparativa e exportação do melhor pipeline Re-ID.

Carrega outputs/features.npz (respeitando estritamente a divisão manual train/ e test/,
sem train_test_split aleatório) e compara SVM (Linear e RBF) e k-NN (Cosseno) em 3 cenários:
  1. Apenas descritores clássicos de PDI (HSV + LBP).
  2. Apenas embeddings DINOv2 (com fundo removido).
  3. Abordagem Híbrida (concatenação PDI + DINOv2).

Gera:
  - Relatório completo de métricas por classe e médias no terminal.
  - outputs/comparison_scenarios.png
  - outputs/confusion_matrix.png
  - outputs/cat_reid_model.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pdi_core import PROJECT_ROOT, configure_stdio_utf8, setup_logging

SCENARIO_LABELS = {
    "pdi": "1. PDI Clássico (HSV + LBP)",
    "dino": "2. DINOv2 (Fundo Removido)",
    "hybrid": "3. Híbrido (PDI + DINOv2)",
}


def parse_args() -> argparse.Namespace:
    """Define e interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Etapa 3: Comparação de cenários (PDI vs DINOv2 vs Híbrido) e treinamento final."
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "features.npz",
        help="Arquivo .npz gerado na Etapa 2 (padrão: outputs/features.npz).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Diretório para salvar gráficos e modelo .pkl (padrão: outputs/).",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="Número de vizinhos k para o classificador k-NN com métrica cosseno (padrão: 5).",
    )
    parser.add_argument(
        "--select-scenario",
        type=str,
        choices=["auto", "pdi", "dino", "hybrid"],
        default="auto",
        help="Força exportação de um cenário específico ou escolhe o melhor ('auto').",
    )
    return parser.parse_args()


def _build_classifiers(n_train: int, min_class_count: int, knn_k: int) -> list[tuple[str, Any]]:
    """Constrói os classificadores leves com calibração segura para bases pequenas."""
    if min_class_count >= 5:
        svm_linear = SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
        svm_rbf = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
    elif min_class_count >= 2:
        svm_linear = CalibratedClassifierCV(
            SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=42),
            cv=min_class_count,
        )
        svm_rbf = CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42),
            cv=min_class_count,
        )
    else:
        raise ValueError(
            "Cada classe de treino precisa ter ao menos 2 amostras para estimar probabilidades (Platt/CV). "
            "Adicione mais imagens ou ative o Data Augmentation na Etapa 2."
        )

    effective_k = max(1, min(int(knn_k), n_train))
    knn_cosine = KNeighborsClassifier(
        n_neighbors=effective_k,
        metric="cosine",
        weights="distance",
    )

    return [
        ("SVM (Linear)", svm_linear),
        ("SVM (RBF)", svm_rbf),
        (f"k-NN (Cosseno, k={effective_k})", knn_cosine),
    ]


def _build_pipeline(
    scenario_key: str,
    classifier: Any,
    d_pdi: int,
    d_dino: int,
) -> Pipeline:
    """Cria um Pipeline scikit-learn com normalização adequada para cada cenário."""
    if scenario_key in ("pdi", "dino"):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("l2", Normalizer(norm="l2")),
                ("clf", classifier),
            ]
        )

    if scenario_key == "hybrid":
        # Normaliza cada bloco (PDI e DINOv2) separadamente antes da normalização L2 conjunta
        # para evitar que um bloco com maior dimensionalidade domine o outro.
        block_norm = ColumnTransformer(
            transformers=[
                (
                    "pdi_block",
                    Pipeline([("scaler", StandardScaler()), ("l2", Normalizer(norm="l2"))]),
                    slice(0, d_pdi),
                ),
                (
                    "dino_block",
                    Pipeline([("scaler", StandardScaler()), ("l2", Normalizer(norm="l2"))]),
                    slice(d_pdi, d_pdi + d_dino),
                ),
            ],
            remainder="drop",
        )
        return Pipeline(
            [
                ("blocks", block_norm),
                ("l2_joint", Normalizer(norm="l2")),
                ("clf", classifier),
            ]
        )

    raise ValueError(f"Cenário desconhecido: {scenario_key}")


def _plot_comparison_scenarios(results: list[dict[str, Any]], output_path: Path) -> None:
    """Salva gráfico de barras comparando os 3 cenários e os 3 classificadores."""
    sns.set_theme(style="whitegrid", context="notebook")

    scenarios_order = ["pdi", "dino", "hybrid"]
    clf_names = []
    for r in results:
        if r["classifier"] not in clf_names:
            clf_names.append(r["classifier"])

    x = np.arange(len(scenarios_order))
    width = 0.25
    palette = sns.color_palette("viridis", n_colors=len(clf_names))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    metrics_to_plot = [
        ("accuracy", "Acurácia no Conjunto de Teste"),
        ("f1_macro", "F1-Score Macro no Conjunto de Teste"),
    ]

    for ax, (metric_key, title) in zip(axes, metrics_to_plot):
        for idx, clf_name in enumerate(clf_names):
            vals = []
            for sc in scenarios_order:
                match = next(
                    (r for r in results if r["scenario"] == sc and r["classifier"] == clf_name),
                    None,
                )
                vals.append(match[metric_key] * 100.0 if match else 0.0)

            offset = (idx - (len(clf_names) - 1) / 2.0) * width
            bars = ax.bar(
                x + offset,
                vals,
                width=width * 0.92,
                label=clf_name,
                color=palette[idx],
                edgecolor="#222222",
                linewidth=0.6,
            )
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2.0, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios_order], fontsize=9.5)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Desempenho (%)" if metric_key == "accuracy" else "")
        ax.legend(loc="lower right", frameon=True, fontsize=9)

    fig.suptitle(
        "Comparação Experimental — Cat Re-ID (Divisão por Ambiente: Treino vs Teste)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    best_record: dict[str, Any],
    output_path: Path,
) -> None:
    """Salva a matriz de confusão anotada do melhor modelo selecionado."""
    sns.set_theme(style="white", context="notebook")

    label_ids = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    fig_w = max(7.5, 0.85 * len(class_names) + 2.5)
    fig_h = max(6.0, 0.72 * len(class_names) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Número de Amostras"},
        linewidths=0.5,
        linecolor="#d0d7de",
        ax=ax,
    )
    ax.set_xlabel("Classe Predita", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Classe Real (Verdadeira)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        f"Matriz de Confusão — {best_record['scenario_label']} | {best_record['classifier']}\n"
        f"Acurácia: {best_record['accuracy']:.2%} | F1 Macro: {best_record['f1_macro']:.2%}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Treina e compara os 3 cenários e exporta gráficos e melhor modelo treinado."""
    configure_stdio_utf8()
    logger = setup_logging()
    args = parse_args()

    features_file: Path = args.features_file.resolve()
    outputs_dir: Path = args.outputs_dir.resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not features_file.exists():
        logger.error(
            f"Arquivo de features não encontrado: {features_file}. "
            "Execute primeiro 'python src/2_extract_features.py'."
        )
        sys.exit(1)

    data = np.load(features_file, allow_pickle=False)
    X_train_pdi = data["X_train_pdi"].astype(np.float32)
    X_train_dino = data["X_train_dino"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)

    X_test_pdi = data["X_test_pdi"].astype(np.float32)
    X_test_dino = data["X_test_dino"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    class_names = [str(c) for c in data["class_names"].tolist()]
    metadata: dict[str, Any] = {}
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"].item()))

    if len(y_train) == 0 or len(y_test) == 0 or len(class_names) < 2:
        logger.error(
            f"Dados inválidos para treinamento/avaliação: "
            f"treino={len(y_train)}, teste={len(y_test)}, classes={len(class_names)}."
        )
        sys.exit(1)

    d_pdi = int(X_train_pdi.shape[1])
    d_dino = int(X_train_dino.shape[1])
    min_class_count = int(np.min(np.bincount(y_train, minlength=len(class_names))))

    logger.info(
        f"Features carregadas: {len(class_names)} gatos | "
        f"Treino={len(y_train)} | Teste={len(y_test)} | "
        f"Dims PDI={d_pdi}, DINOv2={d_dino}"
    )

    scenarios: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "pdi": (X_train_pdi, X_test_pdi),
        "dino": (X_train_dino, X_test_dino),
        "hybrid": (
            np.hstack([X_train_pdi, X_train_dino]),
            np.hstack([X_test_pdi, X_test_dino]),
        ),
    }

    experiments: list[dict[str, Any]] = []
    label_ids = list(range(len(class_names)))

    for scenario_key in ("pdi", "dino", "hybrid"):
        X_tr, X_te = scenarios[scenario_key]
        classifiers = _build_classifiers(
            n_train=len(y_train),
            min_class_count=min_class_count,
            knn_k=args.knn_k,
        )

        for clf_name, clf in classifiers:
            pipeline = _build_pipeline(scenario_key, clf, d_pdi=d_pdi, d_dino=d_dino)
            pipeline.fit(X_tr, y_train)
            y_pred = pipeline.predict(X_te)

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
            f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

            experiments.append(
                {
                    "scenario": scenario_key,
                    "scenario_label": SCENARIO_LABELS[scenario_key],
                    "classifier": clf_name,
                    "accuracy": acc,
                    "precision_macro": prec,
                    "recall_macro": rec,
                    "f1_macro": f1_m,
                    "f1_weighted": f1_w,
                    "y_pred": y_pred,
                    "pipeline": pipeline,
                }
            )

    # Tabela comparativa completa no terminal
    print("\n" + "=" * 96)
    print("COMPARAÇÃO EXPERIMENTAL — CAT RE-ID (PDI vs DINOv2 vs HÍBRIDO)")
    print("=" * 96)
    header = (
        f"{'Cenário':<30} | {'Classificador':<22} | "
        f"{'Acc':>7} | {'Prec(M)':>7} | {'Rec(M)':>7} | {'F1(M)':>7} | {'F1(W)':>7}"
    )
    print(header)
    print("-" * 96)
    for rec_item in experiments:
        print(
            f"{rec_item['scenario_label']:<30} | "
            f"{rec_item['classifier']:<22} | "
            f"{rec_item['accuracy']*100:>6.2f}% | "
            f"{rec_item['precision_macro']*100:>6.2f}% | "
            f"{rec_item['recall_macro']*100:>6.2f}% | "
            f"{rec_item['f1_macro']*100:>6.2f}% | "
            f"{rec_item['f1_weighted']*100:>6.2f}%"
        )
    print("=" * 96)

    # Relatório detalhado por classe do melhor classificador dentro de cada cenário
    for sc_key in ("pdi", "dino", "hybrid"):
        sc_candidates = [e for e in experiments if e["scenario"] == sc_key]
        best_sc = max(sc_candidates, key=lambda r: (r["f1_macro"], r["accuracy"]))
        print(f"\n--- Melhor de {SCENARIO_LABELS[sc_key]}: {best_sc['classifier']} ---")
        print(
            classification_report(
                y_test,
                best_sc["y_pred"],
                labels=label_ids,
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
        )

    # Seleção do modelo campeão geral (ou restrito ao cenário solicitado)
    pool = (
        experiments
        if args.select_scenario == "auto"
        else [e for e in experiments if e["scenario"] == args.select_scenario]
    )
    best_overall = max(pool, key=lambda r: (r["f1_macro"], r["accuracy"]))

    print("=" * 96)
    print(
        f"MODELO SELECIONADO: {best_overall['scenario_label']} + {best_overall['classifier']} "
        f"(Acurácia: {best_overall['accuracy']:.2%} | F1 Macro: {best_overall['f1_macro']:.2%})"
    )
    print("=" * 96 + "\n")

    comparison_path = outputs_dir / "comparison_scenarios.png"
    cm_path = outputs_dir / "confusion_matrix.png"
    model_path = outputs_dir / "cat_reid_model.pkl"

    _plot_comparison_scenarios(experiments, comparison_path)
    _plot_confusion_matrix(
        y_true=y_test,
        y_pred=best_overall["y_pred"],
        class_names=class_names,
        best_record=best_overall,
        output_path=cm_path,
    )

    bundle = {
        "pipeline": best_overall["pipeline"],
        "scenario": best_overall["scenario"],
        "scenario_label": best_overall["scenario_label"],
        "classifier_name": best_overall["classifier"],
        "class_names": class_names,
        "d_pdi": d_pdi,
        "d_dino": d_dino,
        "metrics": {
            "accuracy": best_overall["accuracy"],
            "precision_macro": best_overall["precision_macro"],
            "recall_macro": best_overall["recall_macro"],
            "f1_macro": best_overall["f1_macro"],
            "f1_weighted": best_overall["f1_weighted"],
        },
        "metadata": metadata,
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    logger.info(f"Gráfico comparativo salvo em : {comparison_path}")
    logger.info(f"Matriz de confusão salva em  : {cm_path}")
    logger.info(f"Pipeline exportado em        : {model_path}")


if __name__ == "__main__":
    main()
