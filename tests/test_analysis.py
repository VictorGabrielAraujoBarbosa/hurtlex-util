import numpy as np
import pandas as pd
import pytest

from hurtlex_util.analysis import calcular_idf, calcular_tf, calcular_tf_idf, calcular_tfidf_diferencial


@pytest.fixture
def documents():
    return pd.DataFrame(
        {
            "text": ["alpha alpha beta", "beta gamma", "gamma gamma"],
            "toxic": [1, 0, 1],
        },
        index=[10, 20, 40],
    )


def test_calcular_tf_methods(documents):
    expected_raw = pd.DataFrame(
        [[2, 1, 0], [0, 1, 1], [0, 0, 2]],
        index=documents.index,
        columns=["alpha", "beta", "gamma"],
        dtype=float,
    )

    pd.testing.assert_frame_equal(
        calcular_tf(documents, "text", min_df=1, metodo="raw"), expected_raw
    )
    pd.testing.assert_frame_equal(
        calcular_tf(documents, "text", min_df=1, metodo="relative"),
        expected_raw.div([3, 2, 2], axis=0),
    )
    pd.testing.assert_frame_equal(
        calcular_tf(documents, "text", min_df=1, metodo="binary"),
        (expected_raw > 0).astype(float),
    )

    expected_log = (1 + np.log(expected_raw.where(expected_raw > 0))).fillna(0.0)
    pd.testing.assert_frame_equal(
        calcular_tf(documents, "text", min_df=1, metodo="log"), expected_log
    )


def test_calcular_idf_methods(documents):
    tf = calcular_tf(documents, "text", min_df=1, metodo="raw")

    pd.testing.assert_series_equal(
        calcular_idf(tf, "reciprocal"),
        pd.Series([3, 1.5, 1.5], index=tf.columns, name="idf", dtype=float),
    )
    pd.testing.assert_series_equal(
        calcular_idf(tf, "log"),
        pd.Series([np.log(3), np.log(1.5), np.log(1.5)], index=tf.columns, name="idf"),
    )


def test_max_terms_keeps_most_frequent_terms_with_deterministic_ties(documents):
    tf = calcular_tf(documents, "text", min_df=1, max_terms=2, metodo="raw")

    assert list(tf.columns) == ["alpha", "gamma"]

    relative_tf = calcular_tf(
        documents, "text", min_df=1, max_terms=2, metodo="relative"
    )
    assert relative_tf.loc[10, "alpha"] == pytest.approx(2 / 3)


def test_empty_documents_are_warned_about_and_excluded_from_statistics():
    data = pd.DataFrame(
        {"text": ["alpha alpha", "   ", "beta"], "toxic": [1, 0, 0]},
        index=[4, 8, 12],
    )

    with pytest.warns(UserWarning, match="1 empty documents"):
        tf = calcular_tf(data, "text", min_df=1, metodo="raw")

    assert list(tf.index) == [4, 8, 12]
    assert tf.loc[8].sum() == 0
    assert calcular_idf(tf, "log").tolist() == pytest.approx([np.log(2), np.log(2)])

    with pytest.warns(UserWarning, match="1 empty documents"):
        scores = calcular_tfidf_diferencial(data, "text", min_df=1, metodo_tf="raw")
    assert scores[0][0] == "alpha"
    assert scores[0][1] == pytest.approx(2 * np.log(2))


def test_tf_idf_is_explicit_composition(documents):
    tf = calcular_tf(documents, "text", min_df=1, metodo="relative")
    expected = tf.mul(calcular_idf(tf, "log"), axis="columns")

    pd.testing.assert_frame_equal(
        calcular_tf_idf(
            documents, "text", min_df=1, metodo_tf="relative", metodo_idf="log"
        ),
        expected,
    )


def test_delta_uses_dataframe_rows_and_sorts_target_terms(documents):
    scores = calcular_tfidf_diferencial(
        documents,
        texto_col="text",
        classe_col="toxic",
        min_df=1,
        metodo_tf="raw",
        metodo_idf="log",
    )

    assert [term for term, _ in scores] == ["alpha", "gamma", "beta"]
    assert scores[0][1] > 0
    assert scores[-1][1] < 0


def test_unknown_formula_is_rejected(documents):
    with pytest.raises(ValueError):
        calcular_tf(documents, "text", min_df=1, metodo="unknown")
    with pytest.raises(ValueError):
        calcular_idf(calcular_tf(documents, "text", min_df=1), "unknown")


def test_analysis_rejects_invalid_text_and_labels(documents):
    invalid_text = documents.copy()
    invalid_text.loc[10, "text"] = None
    with pytest.raises(TypeError):
        calcular_tf(invalid_text, "text", min_df=1)

    invalid_labels = documents.copy()
    invalid_labels.loc[10, "toxic"] = 2
    with pytest.raises(ValueError):
        calcular_tfidf_diferencial(invalid_labels, "text", min_df=1)
