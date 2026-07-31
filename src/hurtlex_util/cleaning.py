import re
import pandas as pd
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

# ==========================================
# 1. FUNÇÕES DE FILTRAGEM DE LINHAS (DATAFRAME)
# ==========================================

def filtrar_por_toxicidade(df: pd.DataFrame, coluna_alvo: str = 'toxic', manter_valor: int = 1) -> pd.DataFrame:
    """
    Filtra o conjunto de dados com base no rótulo de toxicidade.
    
    Input:
        df (pd.DataFrame): O conjunto de dados original.
        coluna_alvo (str): O nome da coluna que contém os rótulos (padrão: 'toxic').
        manter_valor (int): O valor da classe a ser mantida (padrão: 1).
    Output:
        pd.DataFrame: Um novo DataFrame contendo apenas as linhas que correspondem ao valor desejado.
    Transformação:
        Aplica uma máscara booleana estrita onde df[coluna_alvo] == manter_valor.
    """
    return df[df[coluna_alvo] == manter_valor].copy()

def filtrar_por_idioma(df: pd.DataFrame, coluna_texto: str, idioma_desejado: str = 'pt') -> pd.DataFrame:
    """
    Remove textos que não pertencem ao idioma especificado.
    
    Input:
        df (pd.DataFrame): O conjunto de dados original.
        coluna_texto (str): O nome da coluna contendo os textos.
        idioma_desejado (str): O código ISO 639-1 do idioma (ex: 'pt' para português).
    Output:
        pd.DataFrame: DataFrame contendo apenas os textos identificados como o idioma desejado.
    Transformação:
        Usa a biblioteca 'langdetect' para inferir o idioma de cada linha. Linhas que falham 
        na detecção ou não correspondem ao 'idioma_desejado' são descartadas.
    """
    def _detectar_seguro(texto):
        try:
            return detect(str(texto))
        except:
            return "desconhecido"
            
    idiomas_detectados = df[coluna_texto].apply(_detectar_seguro)
    return df[idiomas_detectados == idioma_desejado].copy()

# ==========================================
# 2. FUNÇÕES DE TRANSFORMAÇÃO DE TEXTO (STR -> STR)
# ==========================================

def remover_urls(texto: str) -> str:
    """
    Remove links de internet (HTTP/HTTPS e WWW).
    
    Input: 
        texto (str): Uma string de texto bruto.
    Output: 
        str: O texto original, mas com todas as URLs substituídas por um espaço vazio.
    Transformação: 
        Aplica a expressão regular r'http[s]?://\S+|www\.\S+' para identificar padrões 
        de links e apagá-los do texto.
    """
    if not isinstance(texto, str): 
        raise TypeError(f"Erro: esperava argumento string, recebeu ({type(texto).__name__})")
    return re.sub(r'http[s]?://\S+|www\.\S+', '', texto)

def remover_caracteres_especiais(texto: str) -> str:
    """
    Remove números, pontuação e símbolos especiais do texto, mantendo 
    apenas letras (incluindo acentos do português) e espaços.
    
    Input: 
        texto (str): Uma string de texto.
    Output: 
        str: O texto contendo apenas as letras permitidas e espaços.
    Transformação: 
        Aplica uma expressão regular com uma lista EXPLÍCITA de caracteres 
        permitidos. Tudo o que não estiver nessa lista será removido.
    """
    if not isinstance(texto, str): 
        raise TypeError(f"Erro: esperava argumento string, recebeu ({type(texto).__name__})")
    
    # [^ ...] significa "Tudo que NÃO for o que está listado aqui"
    # a-zA-Z -> Letras normais
    # \s     -> Espaços
    # áé...  -> Lista explícita de acentos e cedilha do português
    caracteres_permitidos = r'[^a-zA-Z\sáéíóúÁÉÍÓÚâêîôûÂÊÎÔÛãõÃÕçÇ]'
    
    return re.sub(caracteres_permitidos, '', texto)

def remover_capitalizacao(texto:str) -> str:
    return texto.lower()

def remover_espacos_extras(texto: str) -> str:
    """
    Normaliza o espaçamento do texto.
    
    Input: 
        texto (str): Uma string de texto.
    Output: 
        str: Texto com espaçamento padronizado, sem espaços nas extremidades.
    Transformação: 
        Substitui blocos de 2 ou mais espaços em branco (\\s{2,}) por exatamente 
        um espaço único (' '), e aplica .strip() para limpar o início e fim da string.
    """
    if not isinstance(texto, str): 
        raise TypeError(f"Erro: esperava argumento string, recebeu ({type(texto).__name__})")
    return re.sub(r'\s{2,}', ' ', texto).strip()

def remover_stopwords(texto: str, lista_stopwords: set) -> str:
    """
    Remove palavras comuns e jargões que não agregam valor semântico.
    
    Input: 
        texto (str): Uma string de texto.
        lista_stopwords (set): Um conjunto (set) de palavras a serem removidas. 
                               Exigido explicitamente para garantir transparência.
    Output: 
        str: Texto sem as palavras contidas na lista_stopwords.
    Transformação: 
        Divide o texto em tokens separados por espaço, converte cada token para 
        letras minúsculas e remove os tokens que tiverem correspondência exata 
        dentro do set 'lista_stopwords'.
    """
    if not isinstance(texto, str): 
        raise TypeError(f"Erro: esperava argumento string, recebeu ({type(texto).__name__})")
    
    palavras = texto.split()
    palavras_filtradas = [p for p in palavras if p.lower() not in lista_stopwords]
    return ' '.join(palavras_filtradas)

def remover_emojis(texto: str) -> str:
    """
    Remove todos os emojis do texto.

    Input: 
        texto (str): Uma string de texto bruto.
    Output: 
        str: O texto original com todos os emojis removidos.
    Transformação: 
        Aplica uma expressão regular que cobre blocos Unicode de emojis 
        (incluindo emoticons, pictogramas, bandeiras, símbolos e variações) 
        e os substitui por espaço vazio.
    """
    if not isinstance(texto, str): 
        raise TypeError(f"Erro: esperava argumento string, recebeu ({type(texto).__name__})")
    return _EMOJI_PATTERN.sub('', texto)

def limpar_texto(texto: str, lista_stopwords: set | None = None) -> str:
    """
    Aplica o pipeline completo de limpeza de texto.

    Input: 
        texto (str): Uma string de texto bruto.
        lista_stopwords (set | None): Conjunto opcional de stopwords a remover.
                                       Se None, a remoção de stopwords é ignorada.
        idioma (str | None): Código ISO 639-1 do idioma esperado (ex: 'pt').
                              Se informado, textos em outro idioma retornam
                              string vazia. Se None, ignora a detecção.
    Output: 
        str: Texto limpo e normalizado, ou string vazia se o idioma não
             corresponder ao esperado.
    Transformação: 
        Opcionalmente detecta o idioma e descarta textos que não correspondam.
        Em seguida aplica, em ordem: remoção de URLs, remoção de emojis,
        remoção de caracteres especiais (mantendo apenas letras e espaços),
        conversão para minúsculas, normalização de espaços e, se fornecida
        uma lista, remoção de stopwords.
    """
    if not isinstance(texto, str): 
        raise TypeError(f"Erro: esperava argumento string, recebeu ({type(texto).__name__})")


    texto = remover_urls(texto)
    texto = remover_emojis(texto)
    texto = remover_caracteres_especiais(texto)
    texto = remover_capitalizacao(texto)
    texto = remover_espacos_extras(texto)

    if lista_stopwords is not None:
        texto = remover_stopwords(texto, lista_stopwords)

    return texto