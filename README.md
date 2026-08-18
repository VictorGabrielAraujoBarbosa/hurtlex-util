# HurtLex Util

Um pipeline padronizado e modular de Processamento de Linguagem Natural (PLN) para pesquisa científica, desenvolvido para garantir reprodutibilidade, transparência e clareza metodológica na análise de toxicidade de textos.

Este pacote substitui scripts monolíticos por funções "puras" e rigorosamente documentadas, permitindo que pesquisadores construam pipelines de limpeza de dados explícitos e executem análises estatísticas como o **TF-IDF Diferencial**.

---

## 🛠 Instalação

### Para Usuários (Uso no Jupyter Notebook)
Se você deseja apenas importar as funções e usá-las em sua pesquisa, instale o pacote diretamente do GitHub:

```bash
pip install git+https://github.com/VictorGabrielAraujoBarbosa/hurtlex-util.git
```

**Dica de Reprodutibilidade:** Para fixar uma versão específica no seu paper ou projeto, você pode apontar para um commit ou tag específica:
```bash
pip install git+https://github.com/VictorGabrielAraujoBarbosa/hurtlex-util.git@main
```

### Para Desenvolvedores (Manutenção do Laboratório)
Se você vai editar o código do pacote ou rodar os testes (`pytest`):

```bash
git clone https://github.com/VictorGabrielAraujoBarbosa/hurtlex-util.git
cd hurtlex-util
pip install -e ".[dev]"
```

---

## 🧪 Filosofia Científica

O código neste pacote segue regras estritas para validação acadêmica:
1. **Sem Caixas Pretas:** Regexes e funções listam explicitamente o que estão transformando (ex: lista explícita de acentos do português).
2. **Declaração de Regras:** O pacote não assume stopwords ou regras ocultas. O pesquisador deve passá-las como argumento.
3. **Modularidade:** Cada função faz exatamente uma coisa.

---

## 🚀 Como Usar (Exemplo no Jupyter Notebook)

Abaixo está um exemplo completo de como construir seu pipeline científico de limpeza e aplicar o cálculo do TF-IDF Diferencial dentro de um Jupyter Notebook.

```python
# 1. IMPORTAÇÕES
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Importando do nosso pacote
from hurtlex_util.cleaning import (
    filtrar_por_idioma,
    filtrar_por_toxicidade,
    remover_urls,
    remover_caracteres_especiais,
    remover_capitalizacao,
    remover_espacos_extras,
    remover_stopwords
)
from hurtlex_util.analysis import calcular_tfidf_diferencial

# 2. DEFINIÇÃO EXPLÍCITA DE REGRAS (Para metodologia do paper)
nltk.download('stopwords', quiet=True)
MINHAS_STOPWORDS = set(stopwords.words('portuguese'))
MINHAS_STOPWORDS.update(['user', 'rt', 'http']) 

# 3. CARREGANDO OS DADOS
# Supondo que você tenha um dataframe com as colunas 'text' e 'toxic'
df = pd.read_csv('meus_dados_brutos.csv')

# 4. FILTRAGEM DE LINHAS
# Mantém apenas textos em português
df = filtrar_por_idioma(df, coluna_texto='text', idioma_desejado='pt')

# 5. PIPELINE DE LIMPEZA DE TEXTO
def meu_pipeline_cientifico(texto: str) -> str:
    """Aplica as transformações de texto em ordem estrita."""
    texto = remover_urls(texto)
    texto = remover_caracteres_especiais(texto)
    texto = remover_capitalizacao(texto)
    texto = remover_espacos_extras(texto)
    texto = remover_stopwords(texto, lista_stopwords=MINHAS_STOPWORDS)
    return texto

# Aplicando a limpeza
print("Limpando os textos...")
df['clean_text'] = df['text'].apply(meu_pipeline_cientifico)

# 6. ANÁLISE: TF-IDF DIFERENCIAL
# Compara a classe tóxica (1) com a não-tóxica (0) para achar vocabulário exclusivo
print("Calculando TF-IDF Diferencial...")
resultados = calcular_tfidf_diferencial(
    df=df, 
    texto_col='clean_text', 
    classe_col='toxic', 
    max_features=5000, 
    min_df=5
)

# 7. EXIBINDO OS RESULTADOS
print("\nTop 10 Termos Mais Exclusivos da Classe Tóxica:")
print("-" * 50)
for termo, score in resultados[:10]:
    if score > 0.001: 
        print(f"Termo: {termo.ljust(20)} | Score Diff: {score:.4f}")
```

---

## 📂 Estrutura dos Módulos

* **`cleaning.py`**: Funções atômicas (puras) que recebem texto e retornam texto limpo, ou recebem DataFrames e filtram linhas indesejadas (como detecção de idioma e rótulo).
* **`analysis.py`**: Contém o orquestrador `calcular_tfidf_diferencial`, que implementa a métrica baseada em frequência global, isolamento de classes e subtração de médias colunares.

## ✅ Testes

Para rodar a suíte completa de verificação científica:
```bash
pytest -vv
```
## Analysis

The analysis functions expect cleaned text. Documents are tokenized explicitly
with `str.split()`, so spaces separate terms and cleaning is a separate step.

The calculation is composed of four inspectable stages:

```text
text -> TF -> IDF -> TF-IDF -> differential score
```

`calcular_tf` supports these TF formulas:

- `raw`: term occurrence count.
- `relative`: occurrence count divided by the full cleaned-document token
  length (the default), including tokens removed by vocabulary limits.
- `log`: `1 + log(count)` for terms that occur.
- `binary`: one for presence, zero for absence.

`calcular_idf` supports:

- `reciprocal`: `N / df`.
- `log`: `log(N / df)` (the default).
- `smooth_log`: `log((N + 1) / (df + 1)) + 1`.

Use `max_terms` when an upper bound on vocabulary size is needed. It defaults to
`None`, meaning that all terms meeting `min_df` are retained. Terms are chosen
by corpus frequency with alphabetical tie-breaking.

Empty documents are retained as zero rows in the TF and TF-IDF matrices, but a
warning is issued and they are excluded from IDF and differential statistics.
This preserves row alignment without allowing documents containing no
analyzable terms to affect the scores.
