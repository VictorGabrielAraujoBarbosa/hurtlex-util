"""Explicit term-frequency and differential TF-IDF calculations.

The analysis module expects cleaned text. Its calculation stages are kept
separate and visible:

    text -> whitespace tokens -> term counts -> TF -> IDF -> TF-IDF -> delta

Empty documents are retained in the TF and TF-IDF matrices so their rows stay
aligned with the input DataFrame. They are excluded from IDF and delta
statistics because they contain no analyzable terms. A warning is emitted when
they are encountered.
"""

from collections import Counter
from typing import List, Literal, Tuple
import warnings

import numpy as np
import pandas as pd


TFMethod = Literal["raw", "relative", "log", "binary"]
IDFMethod = Literal["reciprocal", "log", "smooth_log"]


def _validate_texts(df: pd.DataFrame, texto_col: str) -> list[list[str]]:
    """Validate cleaned text and split each document on whitespace."""
    if texto_col not in df:
        raise KeyError(f"Column not found: {texto_col}")

    texts = df[texto_col].tolist()
    invalid_rows = [index for index, text in zip(df.index, texts) if not isinstance(text, str)]
    if invalid_rows:
        raise TypeError(f"Column {texto_col} contains non-string values at rows {invalid_rows}")

    tokenized_documents = [text.split() for text in texts]
    empty_count = sum(not tokens for tokens in tokenized_documents)
    if empty_count:
        warnings.warn(
            f"Found {empty_count} empty documents after tokenization; "
            "they will be excluded from IDF and delta calculations.",
            UserWarning,
            stacklevel=2,
        )

    return tokenized_documents


def _validate_vocabulary_options(max_terms: int | None, min_df: int) -> None:
    if max_terms is not None and max_terms < 1:
        raise ValueError("max_terms must be positive or None")
    if min_df < 1:
        raise ValueError("min_df must be positive")


def calcular_tf(
    df: pd.DataFrame,
    texto_col: str = "clean_text",
    max_terms: int | None = None,
    min_df: int = 5,
    metodo: TFMethod = "relative",
) -> pd.DataFrame:
    """Return one term-frequency row for every document in ``df``.

    The input is expected to contain cleaned text. Tokens are separated with
    ``str.split()``, so cleaning and tokenization use the same whitespace
    contract. ``raw`` is the occurrence count, ``relative`` divides counts by
    document length, ``log`` uses ``1 + log(count)``, and ``binary`` records
    only presence.

    ``min_df`` removes terms appearing in fewer documents. ``max_terms`` is an
    optional upper bound on the number of retained terms. Terms are selected
    by descending corpus frequency, with alphabetical tie-breaking, and the
    resulting columns are alphabetical. Empty documents remain as zero rows.
    """
    _validate_vocabulary_options(max_terms, min_df)
    if metodo not in {"raw", "relative", "log", "binary"}:
        raise ValueError(f"Unknown TF method: {metodo}")

    tokenized_documents = _validate_texts(df, texto_col)
    document_lengths = pd.Series(
        [len(tokens) for tokens in tokenized_documents], index=df.index, dtype=float
    )
    document_frequency = Counter()
    corpus_frequency = Counter()
    document_counters = []

    for tokens in tokenized_documents:
        counts = Counter(tokens)
        document_counters.append(counts)
        document_frequency.update(counts.keys())
        corpus_frequency.update(counts)

    terms = [term for term, frequency in document_frequency.items() if frequency >= min_df]
    terms.sort(key=lambda term: (-corpus_frequency[term], term))
    if max_terms is not None:
        terms = terms[:max_terms]
    terms.sort()

    if not terms:
        raise ValueError("No terms remain after applying min_df and max_terms")

    counts = pd.DataFrame(
        [[counter[term] for term in terms] for counter in document_counters],
        index=df.index,
        columns=terms,
        dtype=float,
    )

    if metodo == "raw":
        return counts
    if metodo == "binary":
        return (counts > 0).astype(float)
    if metodo == "log":
        positive_counts = counts.where(counts > 0)
        return (1 + np.log(positive_counts)).fillna(0.0)

    # Empty documents have no terms, so their relative TF values are zero.
    return counts.div(document_lengths.replace(0, np.nan), axis=0).fillna(0.0)


def _non_empty_rows(tf: pd.DataFrame) -> pd.Series:
    return tf.sum(axis=1) > 0


def calcular_idf(tf: pd.DataFrame, metodo: IDFMethod = "log") -> pd.Series:
    """Return inverse-document-frequency values for the terms in ``tf``.

    Empty-document rows are excluded. ``reciprocal`` uses ``N / df``, ``log``
    uses ``log(N / df)``, and ``smooth_log`` uses
    ``log((N + 1) / (df + 1)) + 1``.
    """
    if metodo not in {"reciprocal", "log", "smooth_log"}:
        raise ValueError(f"Unknown IDF method: {metodo}")
    if not isinstance(tf, pd.DataFrame):
        raise TypeError("tf must be a pandas DataFrame")

    non_empty = _non_empty_rows(tf)
    active_tf = tf.iloc[non_empty.to_numpy()]
    if active_tf.empty:
        raise ValueError("Cannot calculate IDF: all documents are empty")

    document_count = len(active_tf)
    document_frequency = (active_tf > 0).sum(axis=0)

    if metodo == "reciprocal":
        values = document_count / document_frequency
    elif metodo == "log":
        values = np.log(document_count / document_frequency)
    else:
        values = np.log((document_count + 1) / (document_frequency + 1)) + 1

    return pd.Series(values, index=tf.columns, name="idf", dtype=float)


def calcular_tf_idf(
    df: pd.DataFrame,
    texto_col: str = "clean_text",
    max_terms: int | None = None,
    min_df: int = 5,
    metodo_tf: TFMethod = "relative",
    metodo_idf: IDFMethod = "log",
) -> pd.DataFrame:
    """Return the TF-IDF matrix produced by the two explicit stages."""
    tf = calcular_tf(df, texto_col, max_terms, min_df, metodo_tf)
    idf = calcular_idf(tf, metodo_idf)
    return tf.mul(idf, axis="columns")


def calcular_delta_tfidf(
    tf_idf: pd.DataFrame, classes: pd.Series | np.ndarray
) -> List[Tuple[str, float]]:
    """Rank terms by target mean TF-IDF minus background mean TF-IDF."""
    class_values = np.asarray(classes)
    if len(class_values) != len(tf_idf):
        raise ValueError("classes must have one value for every TF-IDF row")
    if not np.isin(class_values, [0, 1]).all():
        raise ValueError("classes must contain only 0 and 1 labels")

    non_empty = _non_empty_rows(tf_idf).to_numpy()
    target = tf_idf.iloc[non_empty & (class_values == 1)]
    background = tf_idf.iloc[non_empty & (class_values == 0)]
    if target.empty or background.empty:
        raise ValueError("Both classes must contain at least one non-empty document")

    delta = target.mean(axis=0) - background.mean(axis=0)
    return sorted(delta.items(), key=lambda item: item[1], reverse=True)


def calcular_delta_tf(
    tf: pd.DataFrame, classes: pd.Series | np.ndarray
) -> List[Tuple[str, float]]:
    """Rank terms by target mean TF minus background mean TF."""
    class_values = np.asarray(classes)
    if len(class_values) != len(tf):
        raise ValueError("classes must have one value for every TF row")
    if not np.isin(class_values, [0, 1]).all():
        raise ValueError("classes must contain only 0 and 1 labels")

    non_empty = _non_empty_rows(tf).to_numpy()
    target = tf.iloc[non_empty & (class_values == 1)]
    background = tf.iloc[non_empty & (class_values == 0)]
    if target.empty or background.empty:
        raise ValueError("Both classes must contain at least one non-empty document")

    delta = target.mean(axis=0) - background.mean(axis=0)
    return sorted(delta.items(), key=lambda item: item[1], reverse=True)


def calcular_tfidf_diferencial(
    df: pd.DataFrame,
    texto_col: str = "clean_text",
    classe_col: str = "toxic",
    max_terms: int | None = None,
    min_df: int = 5,
    metodo_tf: TFMethod = "relative",
    metodo_idf: IDFMethod = "log",
) -> List[Tuple[str, float]]:
    """Rank terms by target mean TF-IDF minus background mean TF-IDF.

    Labels must be exactly ``1`` for the target class or ``0`` for the
    background class. Empty documents are excluded from both class means.
    """
    if classe_col not in df:
        raise KeyError(f"Column not found: {classe_col}")
    if not df[classe_col].isin([0, 1]).all():
        raise ValueError(f"Column {classe_col} must contain only 0 and 1 labels")
    if not (df[classe_col] == 1).any() or not (df[classe_col] == 0).any():
        raise ValueError("Delta TF-IDF requires both target (1) and background (0) documents")

    tf_idf = calcular_tf_idf(
        df,
        texto_col=texto_col,
        max_terms=max_terms,
        min_df=min_df,
        metodo_tf=metodo_tf,
        metodo_idf=metodo_idf,
    )
    return calcular_delta_tfidf(tf_idf, df[classe_col].to_numpy())
