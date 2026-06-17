# 🤟 Intérprete de Libras em Tempo Real

Sistema de reconhecimento da Língua Brasileira de Sinais (Libras) que captura sinais pela webcam, classifica-os com uma rede neural recorrente (LSTM) e devolve o resultado em texto na tela e em voz (síntese de fala).

---

## 📑 Sumário

1. [Objetivo Principal](#1-objetivo-principal)
2. [Tecnologias Utilizadas](#2-tecnologias-utilizadas)
3. [Recursos de IA e Funcionamento](#3-recursos-de-ia-e-funcionamento)
4. [Acurácia Alcançada](#4-acurácia-alcançada)
5. [Análise Crítica e Trabalhos Futuros](#5-análise-crítica-e-trabalhos-futuros)
6. [Como Executar](#6-como-executar)
7. [Estrutura do Projeto](#7-estrutura-do-projeto)

---

## 1. Objetivo Principal

Construir um intérprete capaz de reconhecer **sinais estáticos e dinâmicos** de Libras em tempo real a partir de vídeo de webcam, traduzindo o gesto reconhecido em **texto** e em **áudio**.

O sistema cobre dois tipos de sinais:

- **Palavras/expressões** (sinais com movimento): `OI`, `TUDO_BEM`, `OBRIGADO`, `POR_FAVOR`, `EU`, `VOCE`, `NOS`, `GOSTAR`, `QUERER`, `COMER`, `BEBER`, `AGUA`, `CAFE`, `CASA`, `ESCOLA`.
- **Alfabeto datilológico** (letras), incluindo letras com movimento como `H`, `J`, `X`, `Z`.

Diferente de classificadores que olham apenas uma foto, este projeto analisa **sequências de 20 frames**, o que permite capturar o **movimento** do sinal — essencial em Libras, onde muitos sinais não são poses estáticas.

---

## 2. Tecnologias Utilizadas

| Tecnologia | Função no projeto | Por que foi escolhida |
|---|---|---|
| **MediaPipe (Hand Landmarker / Tasks API)** | Extração de 21 pontos (landmarks) de cada mão a partir da imagem | Detecção de mãos **eficiente em tempo real**, mesmo em CPU. Entrega coordenadas normalizadas (x, y, z) prontas para uso, evitando treinar um detector do zero. |
| **TensorFlow / Keras** | Definição, treinamento e inferência da rede **LSTM** | Framework maduro para redes recorrentes, com suporte a `EarlyStopping`, `ModelCheckpoint` e exportação de modelo (`.keras`). |
| **OpenCV (`opencv-python`)** | Captura da webcam, processamento de frames e interface visual (overlay, FPS, barra de confiança) | Padrão de mercado para visão computacional em tempo real. |
| **pyttsx3** | Síntese de voz (Text-to-Speech) offline | Fala o sinal reconhecido **sem depender de internet**; roda em thread separada para não travar o vídeo. |
| **scikit-learn** | `LabelEncoder`, divisão treino/teste estratificada e métricas (`classification_report`, matriz de confusão) | Ferramentas consolidadas de pré-processamento e avaliação. |
| **pandas / NumPy** | Manipulação do dataset (CSV) e remodelagem dos tensores | Base de qualquer pipeline de dados em Python. |
| **matplotlib / seaborn** | Gráficos de histórico de treino e matriz de confusão | Visualização dos resultados. |

> **Decisão de projeto:** Optamos pelo **MediaPipe** por sua eficiência em tempo real e por entregar diretamente os *landmarks* das mãos. Isso transforma o problema de "classificar imagens" (caro, exige CNNs grandes) em "classificar sequências de coordenadas" (leve, ideal para uma LSTM). O resultado roda fluido até em máquinas sem GPU.

---

## 3. Recursos de IA e Funcionamento

### Pipeline de ponta a ponta

```
┌─────────────┐   ┌──────────────────┐   ┌────────────────────┐   ┌──────────────┐   ┌──────────────┐
│   ENTRADA   │ → │ EXTRAÇÃO DE      │ → │  PRÉ-PROCESSAMENTO │ → │CLASSIFICAÇÃO │ → │    SAÍDA     │
│  Webcam     │   │ FEATURES         │   │  (sequência)       │   │   (LSTM)     │   │ Texto + Voz  │
│ (vídeo)     │   │ MediaPipe →      │   │ buffer de 20       │   │ TensorFlow/  │   │ OpenCV +     │
│             │   │ 21 landmarks/mão │   │ frames × 126 vals  │   │ Keras        │   │ pyttsx3      │
└─────────────┘   └──────────────────┘   └────────────────────┘   └──────────────┘   └──────────────┘
```

### Detalhe de cada etapa

**1. Entrada** — A webcam fornece frames a ~30 FPS. Cada frame é espelhado (efeito "espelho") e convertido para RGB.

**2. Extração de features** — O MediaPipe detecta até **2 mãos** e retorna **21 landmarks por mão**, cada um com coordenadas `(x, y, z)`. Isso gera:

```
21 landmarks × 3 coordenadas = 63 valores por mão
63 (mão esquerda) + 63 (mão direita) = 126 valores por frame
```

Quando uma mão não aparece, seus 63 valores ficam zerados. A separação esquerda/direita usa o rótulo de *handedness* do próprio MediaPipe.

**3. Pré-processamento (a parte que captura o movimento)** — Os vetores de 126 valores são acumulados em um **buffer deslizante de 20 frames** (`deque`). A entrada do modelo é, portanto, um tensor:

```
(20 frames, 126 features)  →  sequência temporal do gesto
```

Na detecção em tempo real, se ainda não há 20 frames, o último frame é repetido para completar a sequência (permite previsões rápidas com latência mínima).

**4. Classificação (a IA)** — Uma rede **LSTM (Long Short-Term Memory)** processa a sequência. A LSTM é uma rede recorrente que "lembra" frames anteriores, ideal para reconhecer **movimento ao longo do tempo**. Arquitetura:

```
Input (20, 126)
   │
   ├─ LSTM(128, return_sequences=True)  + Dropout(0.3)
   ├─ LSTM(64,  return_sequences=True)  + Dropout(0.3)
   ├─ LSTM(32)                          + Dropout(0.3)
   ├─ Dense(64, relu)                   + Dropout(0.3)
   ├─ Dense(32, relu)
   └─ Dense(N_classes, softmax)   →  probabilidade de cada sinal
```

- **Otimizador:** Adam · **Loss:** `sparse_categorical_crossentropy` · **Métrica:** acurácia
- **Regularização:** `Dropout(0.3)` em todas as camadas para evitar overfitting
- **Callbacks:** `EarlyStopping` (paciência 15), `ReduceLROnPlateau` e `ModelCheckpoint` (salva o melhor modelo por acurácia de validação)
- **Treino:** até 150 épocas, batch 32, divisão **80% treino / 20% teste** estratificada

**5. Saída** — A previsão só é aceita acima de **75% de confiança**. O sinal reconhecido é:
- exibido na tela (com barra de confiança, FPS e status), e
- **falado** via `pyttsx3` em uma *thread* separada (com intervalo mínimo de 2s entre falas iguais, para não repetir sem parar).

### Como o modelo é treinado

1. **Coleta** (`coletor_dados.py`): grava sequências de 20 frames de cada sinal e salva em `dados/dataset_libras_2maos.csv`.
2. **Treino** (`treinador.py`): carrega o CSV, remodela em `(amostras, 20, 126)`, treina a LSTM e salva o modelo + encoders em `modelos/`.
3. **Uso** (`detector_tempo_real.py`): carrega o modelo treinado e roda a inferência ao vivo.

---

## 4. Acurácia Alcançada

### Como a acurácia é medida

A acurácia final é calculada sobre o **conjunto de teste (20% dos dados)**, que o modelo **nunca viu durante o treino**. Ao rodar `treinador.py`, o sistema imprime:

```
Acuracia no teste: 0.XXXX
Loss no teste:     0.XXXX
```

e gera dois artefatos em `resultados/`:
- `historico_treinamento_<timestamp>.png` — curvas de acurácia/loss (treino vs. validação);
- `matriz_confusao_<timestamp>.png` — onde o modelo acerta e onde confunde sinais.

> **⚠️ Preencher com o valor real da sua execução.** Exemplo de redação esperada no relatório:
> *"O modelo atingiu uma acurácia de **XX%** no reconhecimento de 22 sinais estáticos/dinâmicos."*

### Dataset atual

O dataset de treino reunido contém **687 sequências** distribuídas em **22 classes**:

| Sinal | Amostras | | Sinal | Amostras |
|---|---|---|---|---|
| A | 58 | | D | 32 |
| AGUA | 51 | | ESCOLA | 31 |
| EU | 48 | | CAFE | 31 |
| TUDO_BEM | 43 | | BEBER | 31 |
| OI | 39 | | CASA | 30 |
| GOSTAR | 39 | | OBRIGADO | 26 |
| B | 36 | | E | 26 |
| QUERER | 34 | | C | 23 |
| NOS | 34 | | COMER | 21 |
| | | | VOCE / G / F / POR_FAVOR | 15 / 15 / 13 / 11 |

### Análise dos resultados

- **Se a acurácia ficou baixa**, as hipóteses prováveis são:
  - **Poucos dados** em algumas classes (`POR_FAVOR` tem só 11 amostras vs. 58 de `A`) → **desbalanceamento** que prejudica as classes minoritárias.
  - **Variação de iluminação** e fundo entre as coletas.
  - **Sinais parecidos** entre si (ex.: letras do alfabeto com poses semelhantes) geram confusão — visível na matriz de confusão.
  - Uma única pessoa coletando os dados → o modelo pode não generalizar para outras mãos/estilos.
- **Referência da literatura:** sistemas maduros de reconhecimento de sinais alcançam **precisão acima de 97%**. Diferenças em relação a esse patamar geralmente vêm de volume e diversidade de dados, não da arquitetura.
- **Boa prática:** balancear o número de amostras por classe e analisar a **matriz de confusão** para identificar exatamente quais sinais o modelo troca.

---

## 5. Análise Crítica e Trabalhos Futuros

### Principais dificuldades

- **Coleta de dados:** montar um dataset balanceado e consistente é a parte mais trabalhosa; sinais com movimento (`H`, `J`, `X`, `Z`) exigem gravar o gesto inteiro dentro da janela de 20 frames.
- **Captura do movimento:** classificar pose única não basta para Libras — a escolha de **sequências + LSTM** resolve isso, mas aumenta a complexidade do pipeline (buffer, timestamps, sincronização).
- **Latência vs. precisão:** o limiar de confiança (75%) e o intervalo entre falas foram ajustados para equilibrar resposta rápida e estabilidade.
- **Detecção de duas mãos:** separar corretamente mão esquerda/direita e tratar o caso de uma mão ausente (zeros) exigiu cuidado no pré-processamento.

### Generalização (pessoas e iluminação)

O modelo atual tende a funcionar melhor para **quem coletou os dados** e em **condições parecidas de iluminação**. Como o MediaPipe normaliza os landmarks (coordenadas relativas), há alguma robustez a posição e tamanho da mão, mas **variações fortes de iluminação, ângulo de câmera ou estilo de sinalização** ainda degradam o desempenho. Para generalizar de verdade, é preciso **coletar dados de várias pessoas e ambientes**.

### Trabalhos futuros — escalando para frases e vocabulário maior

- **Vocabulário maior:** ampliar e balancear o dataset; idealmente com **data augmentation** (ruído, leve rotação/escala nos landmarks) e coleta multipessoa.
- **Frases completas:** evoluir de classificação isolada para **reconhecimento contínuo de gestos**, usando:
  - arquiteturas sequência-a-sequência (Seq2Seq) ou **Transformers** para mapear sequências de sinais → sentenças;
  - **modelos de linguagem** para corrigir/encadear os sinais reconhecidos em frases gramaticais.
- **Aprendizado contínuo (online learning):** permitir que o sistema incorpore novos sinais sem retreinar do zero.
- **Robustez:** treinar com mais variação de fundo/iluminação e avaliar com validação cruzada por pessoa.
- **Distribuição:** já existe um `.spec` do PyInstaller — empacotar como executável standalone facilita o uso por não-programadores.

---

## 6. Como Executar

### Pré-requisitos

- Python 3.10+ e uma **webcam**.

### Instalação

```bash
# (recomendado) criar ambiente virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# instalar dependências
pip install -r requirements.txt
```

> É necessário ter o arquivo `projeto_libras/modelos/hand_landmarker.task` (modelo do MediaPipe). Ele já acompanha o repositório.

### Uso

A partir da pasta `projeto_libras/`:

```bash
cd projeto_libras
python main.py
```

Menu principal:

| Opção | Ação |
|---|---|
| 1 | Coletar dados para treinamento (sequências) |
| 2 | Treinar o modelo de IA (LSTM) |
| 3 | Detecção em tempo real |
| 4 | Estatísticas do dataset |
| 5 | Sair |

**Atalhos do coletor:** `ESPAÇO` grava/para a sequência · `N`/`P` próxima/anterior · `S` salvar e sair · `ESC` sair sem salvar.
**Detector em tempo real:** `ESC` para sair.

> **⚠️ Ajuste do índice da câmera:** o código usa `cv2.VideoCapture(2, cv2.CAP_V4L2)` (detector) e `(1, ...)` (coletor) com backend **V4L2 (Linux)**. No Windows ou se a câmera não abrir, troque o índice para `0` e remova `cv2.CAP_V4L2`, por exemplo: `cv2.VideoCapture(0)`.

---

## 7. Estrutura do Projeto

```
interprete_libras/
├── requirements.txt
├── README.md
└── projeto_libras/
    ├── main.py                  # Menu principal
    ├── coletor_dados.py         # Coleta sequências (20 frames) pela webcam
    ├── treinador.py             # Treina a LSTM e gera métricas/gráficos
    ├── detector_tempo_real.py   # Inferência ao vivo + texto + voz (TTS)
    ├── executar_libras.py       # Atalho de execução
    ├── teste.py                 # Script de teste
    ├── utils/
    │   └── visualizacao.py      # Funções auxiliares de visualização
    ├── dados/
    │   └── dataset_libras_2maos.csv   # Dataset (sequências de landmarks)
    └── modelos/
        ├── hand_landmarker.task        # Modelo MediaPipe (detecção de mãos)
        ├── melhor_modelo_libras.keras  # Modelo LSTM treinado
        ├── label_encoder.pkl           # Codificador de rótulos
        ├── scaler.pkl                  # Normalizador
        └── config.pkl                  # Configuração (tamanho da sequência, classes)
```

---

*Projeto acadêmico — Universidade Católica de Brasília (Católica EAD).*
