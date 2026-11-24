# Cat Re-ID "In the Wild" — Re-identificação de Gatos Individuais com PDI + DINOv2

Projeto acadêmico de Visão Computacional e Processamento Digital de Imagens (PDI) para identificar gatos individuais por nome e identidade a partir de uma base própria capturada em ambientes, ângulos e iluminações variadas, **sem utilizar datasets públicos de gatos para treinamento**.

## PRIMEIRO PASSO

- Baixar O repositório inteiro pelo github
- Ir no Drive https://drive.google.com/drive/folders/1dZcGJ0dvUOZXxlSpVINtg_0dBO7_-iYL?usp=sharing e baixar as pastas faltantes (DATASET e VENV)
- Substituir DATASET e VENV na sua pasta e rodar.

## Arquitetura do Pipeline

1. **Eliminação de Viés de Cenário e Normalização de Iluminação (`src/1_preprocess_pdi.py`)**:
   - Detecta e segmenta a silhueta do gato principal via **YOLO Segmentation** (`yolo11n-seg.pt` / `yolov8n-seg.pt`, classe COCO `cat = 15`).
   - Remove completamente o fundo (substituindo-o por preto `[0, 0, 0]`) e recorta na *bounding box* com margem de **5%**.
   - Converte o recorte para o espaço de cor **CIE LAB**, aplica **CLAHE** apenas no canal de luminância **L** (evitando halos na borda do fundo preto) e executa suavização leve (**Filtro Bilateral** ou **Gaussiano**).
   - Gera painéis comparativos passo a passo em `outputs/debug_pdi/` para o relatório técnico.

2. **Extração Híbrida de Características + Data Augmentation (`src/2_extract_features.py`)**:
   - Aplica *Data Augmentation* leve **apenas no conjunto de treino** (espelhamento horizontal e rotações de `±10°`), mantendo o conjunto de teste intacto.
   - **Descritores Clássicos de PDI (Baseline)**: Histograma normalizado no espaço **HSV** (mascarado para ignorar o fundo `[0, 0, 0]`) concatenado com textura **LBP (*Local Binary Patterns*) uniforme multiescala** em escala de cinza.
   - **Embeddings Profundos (sem fine-tuning)**: Vetor visual extraído com **DINOv2 (`dinov2_vits14`)** pré-treinado em modo `eval()`.

3. **Avaliação Comparativa e Treinamento (`src/3_train_and_evaluate.py`)**:
   - Respeita estritamente a separação manual por ambiente/sessão (`train/` vs `test/`), **sem** `train_test_split` aleatório.
   - Avalia **SVM (Linear)**, **SVM (RBF)** e **k-NN (Métrica Cosseno)** em 3 cenários:
     1. Apenas PDI Clássico (HSV + LBP)
     2. Apenas DINOv2 (com fundo removido)
     3. Abordagem Híbrida (PDI + DINOv2)
   - Exporta gráficos comparativos, matriz de confusão e o melhor pipeline em `outputs/cat_reid_model.pkl`.

4. **Inferência Individual com Regra de 3 Faixas (`src/4_predict_single.py` + `src/inference_utils.py`)**:
   - Recebe qualquer foto nova via CLI, executa todo o pipeline e aplica uma regra de decisão de 3 faixas sobre a maior probabilidade prevista (`max_prob`):
     - `max_prob >= 50%`: **CONFIRMADO** (verde) — identificação aceita com alta confiança.
     - `40% <= max_prob < 50%`: **INCERTO** (laranja) — mantém o palpite principal com `[ALERTA: BAIXA CONFIANÇA / INCERTEZA]`.
     - `max_prob < 40%`: **DESCONHECIDO** (vermelho) — classifica como **`Gato Desconhecido`**.
   - Gera o card visual `outputs/prediction_result.png` com foto original + bounding box, recorte CLAHE, veredito colorido e ranking **Top-3**.

5. **Inferência em Lote e Análise de Erros (`src/5_predict_batch.py`)**:
   - Percorre automaticamente `dataset/test/<nome_do_gato>/` (ou outra pasta via `--test_dir`), gera cards individuais com rótulo real vs. previsto em `outputs/batch_predictions/`, copia erros e desconhecidos para `outputs/batch_predictions/erros/` e monta o mosaico resumo `outputs/batch_summary_grid.png`.

---

## Estrutura de Pastas

```text
Projeto_Final_PDI/
├── dataset/
│   ├── train/                      # Fotos de treino separadas por ambiente/sessão
│   │   ├── Frajola/
│   │   ├── Tom/
│   │   └── ...
│   ├── test/                       # Fotos de teste em ambientes/ângulos inéditos
│   │   ├── Frajola/
│   │   ├── Tom/
│   │   └── ...
│   └── processed/                  # Gerado automaticamente pela Etapa 1 (.png)
├── outputs/
│   ├── debug_pdi/                  # Painéis: Original -> Máscara -> Sem Fundo -> CLAHE
│   ├── batch_predictions/          # Cards individuais gerados pela Etapa 5
│   │   └── erros/                  # Cópias de erros e casos 'Gato Desconhecido'
│   ├── features.npz                # Vetores PDI e DINOv2 de treino e teste
│   ├── comparison_scenarios.png    # Gráfico comparativo dos 3 cenários experimentais
│   ├── confusion_matrix.png        # Matriz de confusão do melhor modelo
│   ├── cat_reid_model.pkl          # Pipeline treinado exportado
│   ├── prediction_result.png       # Resultado visual da inferência individual
│   └── batch_summary_grid.png      # Mosaico resumo com até 12 exemplos variados
├── src/
│   ├── __init__.py
│   ├── pdi_core.py                 # Núcleo compartilhado de PDI, YOLO-Seg e DINOv2
│   ├── inference_utils.py          # Motor de inferência, regra de 3 faixas e cards
│   ├── 1_preprocess_pdi.py         # Etapa 1: Segmentação e normalização PDI
│   ├── 2_extract_features.py       # Etapa 2: Augmentation + Features PDI e DINOv2
│   ├── 3_train_and_evaluate.py     # Etapa 3: Treino, métricas e comparação
│   ├── 4_predict_single.py         # Etapa 4: Predição em imagem única
│   └── 5_predict_batch.py          # Etapa 5: Predição em lote e mosaico resumo
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Instalação

Recomenda-se Python **3.10+** e uso de ambiente virtual (`venv`). O código detecta automaticamente **CUDA (GPU)** ou **CPU**.

### Windows (PowerShell)

DENTRO DA PASTA DO PROJETO!!

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Dica (GPU NVIDIA no Windows/WSL):** Caso o `pip install torch torchvision` instale a versão somente-CPU, instale primeiro o PyTorch com suporte a CUDA pelo índice oficial (`https://pytorch.org/get-started/locally/`) e depois execute `pip install -r requirements.txt`.

### WSL / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Observação de primeiro uso:** Na primeira execução, a biblioteca `ultralytics` fará o download automático dos pesos do YOLO (`yolo11n-seg.pt`, ~6 MB) e o `torch.hub` baixará os pesos do `dinov2_vits14` (~85 MB) para o cache local.

---

## Ordem de Execução

### 0. Organizar as imagens

Coloque as fotos de cada indivíduo nas pastas correspondentes (formatos aceitos: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`):
Se baixou o dataset do drive PULAR ESSA PARTE!

```text
dataset/train/Frajola/foto_sala_01.jpg
dataset/train/Tom/foto_sofa_01.jpg
dataset/test/Frajola/foto_quintal_01.jpg
dataset/test/Tom/foto_varanda_01.jpg
```

Os nomes das classes são lidos **dinamicamente** a partir das pastas. Para adicionar um novo gato no futuro, basta criar `dataset/train/<Novo_Gato>/` e `dataset/test/<Novo_Gato>/` e reexecutar as Etapas 1 a 3 — sem alterar o código.

### 1. Pré-processamento PDI (YOLO-Seg + Fundo Preto + LAB-CLAHE)

```bash
python src/1_preprocess_pdi.py
```

Opções úteis:
- `--yolo-model yolov8n-seg.pt` (caso prefira YOLOv8-seg em vez do padrão `yolo11n-seg.pt`)
- `--smooth gaussian` (alternativa ao padrão `bilateral`)
- `--conf 0.25` (confiança mínima para detecção da classe `cat`)
- `--max-debug-per-class 2` (quantidade de painéis salvos em `outputs/debug_pdi/`)

Se o YOLO não detectar nenhum gato em alguma foto, o script registra um aviso (`WARNING`) no log e continua normalmente sem interromper o lote.

### 2. Extração de Características (PDI Clássico + DINOv2)

```bash
python src/2_extract_features.py
```

Opções úteis:
- `--batch-size 32` (reduza para `8` ou `16` em GPUs/CPUs com pouca memória)
- `--rotation-deg 10.0` (rotações de `±10°` aplicadas apenas no treino)
- `--no-flip` (desativa espelhamento horizontal no treino, caso gatos com manchas assimétricas percam identidade ao espelhar)

### 3. Treinamento, Comparação Experimental e Exportação

```bash
python src/3_train_and_evaluate.py
```

Opções úteis:
- `--knn-k 5` (número de vizinhos no k-NN com métrica cosseno)
- `--select-scenario hybrid` (força exportação de `pdi`, `dino` ou `hybrid`; padrão `auto` seleciona o melhor por F1-Macro e Acurácia)

### 4. Inferência em Uma Foto Nova

```bash
python src/4_predict_single.py --image caminho/para/foto.jpg
```

Regra de decisão por faixa de confiança (`max_prob`):
- **`>= 50%` (`--confirm-threshold 0.50`)**: `CONFIRMADO` (verde) — identificação aceita com alta confiança.
- **`40%` a `50%`**: `INCERTO` (laranja) — exibe o palpite mais provável acompanhado da flag `[ALERTA: BAIXA CONFIANÇA / INCERTEZA]`.
- **`< 40%` (`--unknown-threshold 0.40`)**: `DESCONHECIDO` (vermelho) — retorna **`Gato Desconhecido`**.

Opções úteis:
- `--confirm-threshold 0.50` (ou `--threshold 0.50`, limiar de confirmação em verde)
- `--unknown-threshold 0.40` (limiar abaixo do qual classifica como `Gato Desconhecido`)
- `--output outputs/prediction_result.png`

### 5. Inferência em Lote sobre Conjunto de Teste

```bash
python src/5_predict_batch.py
```

Para testar um diretório alternativo (ex.: `dataset/test2`):

```bash
python src/5_predict_batch.py --test_dir dataset/test2
```

Saídas geradas automaticamente:
- `outputs/batch_predictions/`: um card visual por foto contendo `Real: <true_label> | Previsto: <pred_label> (<conf>%)`.
- `outputs/batch_predictions/erros/`: cópias automáticas de todos os erros de classe e casos `Gato Desconhecido`.
- `outputs/batch_summary_grid.png`: mosaico com até 12 exemplos variados (acertos, alertas e falhas) pronto para relatório e apresentação.
- Resumo quantitativo completo impresso no terminal.

---

## Como Calibrar os Limiares de Incerteza (40%) e Confirmação (50%)

1. Execute `python src/5_predict_batch.py` sobre `dataset/test/` para inspecionar a distribuição de acertos confirmados (`>= 50%`), acertos com alerta (`40%–50%`), desconhecidos (`< 40%`) e erros.
2. Caso tenha uma pasta de gatos externos (ex.: `dataset/test/fakes/`), avalie quantos foram rejeitados abaixo de `40%` em `outputs/batch_predictions/erros/`.
3. Ajuste `--unknown-threshold` e `--confirm-threshold` caso queira tornar a rejeição de desconhecidos mais rigorosa (ex.: `0.45` e `0.60`) ou mais tolerante.
