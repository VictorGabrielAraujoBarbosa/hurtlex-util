import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from langdetect import detect, DetectorFactory

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U0000200D"
    "\U0000FE0F"
    "\U00002300-\U000023FF"
    "\U00002600-\U000026FF"
    "]+"
)

# Garante reprodutibilidade científica na detecção de idioma
DetectorFactory.seed = 42 

nltk.download('stopwords')  # type: ignore[attr-defined]

# ==========================================
# 1. FUNÇÕES DE FILTRAGEM DE LINHAS (DATAFRAME)
# ==========================================

def filtrar_por_toxicidade(df: pd.DataFrame, coluna_alvo: str = 'toxic', manter_valor: int = 1) -> pd.DataFrame:
    """
    Filtra o conjunto de dados com base no rótulo de toxicidade.
    """
    return df[df[coluna_alvo] == manter_valor].copy()

def filtrar_por_idioma(df: pd.DataFrame, coluna_texto: str, idioma_desejado: str = 'pt') -> pd.DataFrame:
    """
    Filtra o conjunto de dados mantendo apenas as linhas em que o texto
    pertence ao idioma desejado.
    """
    def checar_idioma(texto):
        try:
            return detect(str(texto)) == idioma_desejado
        except:
            return False
            
    mascara = df[coluna_texto].apply(checar_idioma)
    return df[mascara].copy()
    

# ==========================================
# 2. FUNÇÕES DE TRANSFORMAÇÃO DE TEXTO (STR -> STR)
# ==========================================

def remover_urls(texto: str) -> str:
    """Remove links de internet (HTTP/HTTPS e WWW)."""
    return re.sub(r'http[s]?://\S+|www\.\S+', '', texto)

def remover_caracteres_especiais(texto: str) -> str:
    """Remove números, pontuação e símbolos especiais do texto, mantendo apenas letras."""
    caracteres_permitidos = r'[^a-zA-Z\sáéíóúÁÉÍÓÚâêîôûÂÊÎÔÛãõÃÕçÇ]'
    return re.sub(caracteres_permitidos, '', texto)

def remover_letras_repetidas(texto: str) -> str:
    """Reduz sequências de uma mesma letra para apenas 2 ocorrências."""
    return re.sub(r'(.)\1{2,}', r'\1\1', texto)

def remover_capitalizacao(texto: str) -> str:
    return texto.lower()

def remover_espacos_extras(texto: str) -> str:
    """Normaliza o espaçamento do texto."""
    return re.sub(r'\s{2,}', ' ', texto).strip()

def remover_stopwords(texto: str, lista_stopwords: set[str] | None = None) -> str:
    if not isinstance(texto, str):
        raise TypeError(f"Esperado tipo 'str', recebido '{type(texto).__name__}'")

    if lista_stopwords is None:
        lista_stopwords = set(stopwords.words('portuguese'))  # type: ignore[attr-defined]
    palavras = texto.split()
    palavras_filtradas = [p for p in palavras if p.lower() not in lista_stopwords]
    return ' '.join(palavras_filtradas)

def remover_emojis(texto: str) -> str:
    return _EMOJI_PATTERN.sub('', texto)

def limpar_texto(texto: str, lista_stopwords: set[str] | None = None, idioma: str | None = None) -> str:
    """
    Aplica o pipeline completo de limpeza de texto.
    """
    if not isinstance(texto, str):
        raise TypeError(f"Esperado tipo 'str', recebido '{type(texto).__name__}'")

    # Verifica o idioma antes de aplicar as transformações
    if idioma:
        try:
            if detect(texto) != idioma:
                return ""
        except:
            return ""

    texto = remover_urls(texto)
    texto = remover_emojis(texto)
    texto = remover_caracteres_especiais(texto)
    texto = remover_letras_repetidas(texto)
    texto = remover_capitalizacao(texto)
    texto = remover_espacos_extras(texto)
    texto = remover_stopwords(texto, lista_stopwords)
        
    return texto
