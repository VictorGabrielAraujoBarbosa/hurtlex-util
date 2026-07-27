"""
=============================================================================
MÓDULO DE ANÁLISE: TF-IDF DIFERENCIAL
=============================================================================

Fundamentação Científica:
-------------------------
O objetivo deste módulo é identificar quais palavras são estatisticamente 
mais exclusivas e representativas de uma classe específica de textos 
(ex: textos tóxicos) em comparação a uma classe de fundo (ex: não-tóxicos).

A técnica tradicional de TF-IDF (Term Frequency-Inverse Document Frequency) 
mede a importância de uma palavra em um documento específico em relação a todo 
o corpus. No entanto, o TF-IDF padrão não é projetado para comparar a 
importância de uma palavra entre *classes* de documentos.

Para resolver isso, aplicamos a técnica de "TF-IDF Diferencial":

1. Vetorização Global: 
   Calculamos o TF-IDF para todo o conjunto de dados simultaneamente. 
   Isso garante que a penalidade de frequência inversa no documento (IDF) 
   seja aplicada de forma consistente e global. Palavras comuns à língua 
   como um todo são penalizadas uniformemente.

2. Isolamento de Classes: 
   A matriz global é dividida em dois sub-corpora: Alvo (ex: Tóxico) e 
   Fundo (ex: Não-tóxico).

3. Cálculo de Médias Colunares: 
   Para cada termo matematicamente representado na matriz, calculamos a média 
   do seu score TF-IDF dentro da classe Alvo e dentro da classe de Fundo.

4. Score Diferencial: 
   A métrica final é obtida pela subtração simples das médias:
   
   $$Score_{diff}(t) = \mu_{alvo}(t) - \mu_{fundo}(t)$$

Interpretação dos Resultados:
-----------------------------
* Score > 0 : O termo é matematicamente mais característico da classe Alvo. 
              Quanto maior o valor, mais exclusivo e forte é o termo.
* Score < 0 : O termo é mais característico da classe de Fundo.
* Score ≈ 0 : O termo é igualmente distribuído entre as classes (neutro) ou 
              tem relevância estatística insignificante.

Desta forma, esta técnica filtra ruídos semânticos e isola o vocabulário 
que de fato define a anomalia (toxicidade) sendo estudada.
=============================================================================
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple

# ==========================================
# FUNÇÕES INTERNAS DE PROCESSAMENTO MATEMÁTICO
# ==========================================

def _gerar_matriz_tfidf_global(df: pd.DataFrame, texto_col: str, max_features: int, min_df: int) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Ajusta o modelo TF-IDF em todo o corpus para garantir que os pesos 
    globais (IDF) sejam consistentes entre as classes.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
    matriz_tfidf = vectorizer.fit_transform(df[texto_col])
    termos = vectorizer.get_feature_names_out()
    
    return matriz_tfidf, termos

def _separar_matrizes_por_classe(df: pd.DataFrame, matriz_tfidf: sp.csr_matrix, classe_col: str) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Filtra as linhas da matriz esparsa isolando as classes Alvo (1) e Fundo (0).
    """
    idx_alvo = df.index[df[classe_col] == 1].tolist()
    idx_fundo = df.index[df[classe_col] == 0].tolist()
    
    matriz_alvo = matriz_tfidf[idx_alvo]
    matriz_fundo = matriz_tfidf[idx_fundo]
    
    return matriz_alvo, matriz_fundo

def _calcular_medias_por_termo(matriz: sp.csr_matrix) -> np.ndarray:
    """
    Calcula a média do score TF-IDF de cada termo ao longo de um sub-corpus.
    Usa axis=0 para achatar a matriz em um array 1D (A1) com as médias.
    """
    # Se a matriz estiver vazia (ex: não há textos na classe), retorna array de zeros
    if matriz.shape[0] == 0:
        return np.zeros(matriz.shape[1])
        
    return matriz.mean(axis=0).A1

# ==========================================
# FUNÇÃO PRINCIPAL (ORQUESTRADORA)
# ==========================================

def calcular_tfidf_diferencial(
    df: pd.DataFrame, 
    texto_col: str = 'clean_text', 
    classe_col: str = 'toxic', 
    max_features: int = 5000, 
    min_df: int = 5
) -> List[Tuple[str, float]]:
    """
    Calcula o TF-IDF Diferencial para encontrar termos exclusivos da classe alvo.
    
    Input:
        df (pd.DataFrame): O conjunto de dados contendo o texto e os rótulos.
        texto_col (str): O nome da coluna com os textos já limpos.
        classe_col (str): O nome da coluna binária (1 para Alvo, 0 para Fundo).
        max_features (int): O limite máximo de termos a serem avaliados.
        min_df (int): Número mínimo de documentos em que um termo deve aparecer.
        
    Output:
        List[Tuple[str, float]]: Uma lista de tuplas (termo, score_diferencial), 
                                 ordenada do maior score (mais exclusivo ao alvo) 
                                 para o menor.
    """
    # 1. Ajustar o vetorizador em todo o corpus
    matriz_tfidf, termos = _gerar_matriz_tfidf_global(
        df, texto_col, max_features, min_df
    )
    
    # 2. Isolar as matrizes por classe
    matriz_alvo, matriz_fundo = _separar_matrizes_por_classe(
        df, matriz_tfidf, classe_col
    )
    
    # 3. Calcular as médias colunares (por termo) para cada classe
    media_alvo = _calcular_medias_por_termo(matriz_alvo)
    media_fundo = _calcular_medias_por_termo(matriz_fundo)
    
    # 4. Calcular o score diferencial subtraindo o fundo do alvo
    score_diferencial = media_alvo - media_fundo
    
    # 5. Parear os termos com seus scores e ordenar do maior para o menor
    itens = zip(termos, score_diferencial)
    itens_ordenados = sorted(itens, key=lambda x: x[1], reverse=True)
    
    return itens_ordenados