import pytest
import pandas as pd
from hurtlex_util.cleaning import (
    filtrar_por_toxicidade,
    filtrar_por_idioma,
    remover_urls,
    remover_caracteres_especiais,
    remover_capitalizacao,
    remover_espacos_extras,
    remover_stopwords
)

# ==========================================
# 1. TESTES DE FILTRAGEM (DATAFRAME)
# ==========================================

def test_filtrar_por_toxicidade():
    # Setup: Create a dummy dataframe with toxic and non-toxic rows
    df = pd.DataFrame({
        'texto': ['odio', 'paz', 'raiva'],
        'toxic': [1, 0, 1]
    })
    
    # Action: Apply the default filter (keeps 'toxic' == 1)
    df_filtrado = filtrar_por_toxicidade(df)
    
    # Assertions
    assert len(df_filtrado) == 2
    assert 0 not in df_filtrado['toxic'].values
    assert list(df_filtrado['texto']) == ['odio', 'raiva']

def test_filtrar_por_idioma():
    df = pd.DataFrame({
        'texto': [
            'Este é um texto em português limpo.', # pt
            'This is a text in english.',          # en
            'Hola, como estas mi amigo?'           # es
        ]
    })
    
    # Action: Keep only Portuguese text
    df_filtrado = filtrar_por_idioma(df, coluna_texto='texto', idioma_desejado='pt')
    
    # Assertions
    assert len(df_filtrado) == 1
    assert 'Este é um texto em português limpo.' in df_filtrado['texto'].values

# ==========================================
# 2. TESTES DE TRANSFORMAÇÃO DE TEXTO
# ==========================================

def test_remover_urls():
    texto = "Veja http://google.com e https://site.br ou www.uol.com"
    resultado = remover_urls(texto)
    
    # Expects all URLs to be replaced by empty strings
    assert "http://google.com" not in resultado
    assert "www.uol.com" not in resultado
    assert resultado == "Veja  e  ou "

def test_remover_caracteres_especiais():
    # Tests if the explicit list of accents is preserved while numbers/symbols are dropped
    texto = "Olá, mundo! 123 @user O coração é #azul!"
    resultado = remover_caracteres_especiais(texto)
    
    assert resultado == "Olá mundo  user O coração é azul"

def test_remover_caracteres_especiais_preserva_espacos():
    # Tests if the explicit list of accents is preserved while numbers/symbols are dropped
    texto = "! !!  !!!   seis espacos antes"
    resultado = remover_caracteres_especiais(texto)
    
    assert resultado == "      seis espacos antes"

def test_remover_capitalizacao():
    texto = "TUDO MaIúScUlO"
    resultado = remover_capitalizacao(texto)
    
    assert resultado == "tudo maiúsculo"

def test_remover_espacos_extras():
    # Tests leading/trailing spaces, and multi-spaces in the middle
    texto = "   Muitos    espaços   aqui  "
    resultado = remover_espacos_extras(texto)
    
    assert resultado == "Muitos espaços aqui"

def test_remover_stopwords():
    texto = "Eu gosto muito de estudar a linguagem python"
    minhas_stopwords = {'eu', 'muito', 'de', 'a'}
    
    resultado = remover_stopwords(texto, lista_stopwords=minhas_stopwords)
    
    # Should remove exact matches in the stopword set
    assert resultado == "gosto estudar linguagem python"

# ==========================================
# 3. TESTES DE TRATAMENTO DE ERRO (TIPAGEM)
# ==========================================

def test_erros_de_tipagem():
    """Garante que funções de texto rejeitam dados não-string (ex: NaN do Pandas)"""
    inputs_invalidos = [None, 123, 45.6, ['lista']]
    
    for entrada in inputs_invalidos:
        with pytest.raises(TypeError):
            remover_urls(entrada)
            
        with pytest.raises(TypeError):
            remover_caracteres_especiais(entrada)
            
        with pytest.raises(TypeError):
            remover_espacos_extras(entrada)
            
        with pytest.raises(TypeError):
            remover_stopwords(entrada, {'a'})