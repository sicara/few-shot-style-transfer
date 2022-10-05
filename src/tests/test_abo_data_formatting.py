from src.abo_data_formatting import ABOFormatting
import pandas as pd
import numpy as np
import pytest


class TestReadMetadata:
    @staticmethod
    def test_read_metadata_makes_metadata_df_with_the_expected_columns(
        path_to_resources,
    ):
        abo_formatting = ABOFormatting(path_to_resources)
        abo_formatting.read_metadata()
        expected_metadata_df = pd.read_csv(
            path_to_resources / "expected_read_metadata_for_listings_0.csv", index_col=0
        )
        assert (
            abo_formatting.metadata_df.columns == expected_metadata_df.columns
        ).all()


class TestTranslationToEn:
    cases_grid = [
        {
            "text": "multi-colored",
            "src_language": "en",
            "expected_text": "multicolored",
        },
        {"text": "Blau", "src_language": "de", "expected_text": "blue"},
        {"text": "rouge", "src_language": "xy", "expected_text": "red"},
        {"text": "하얀색", "src_language": "ko", "expected_text": "white"},
        {"text": "a", "src_language": "en", "expected_text": np.nan},
    ]

    @staticmethod
    @pytest.mark.parametrize(
        "text,src_language,expected_text",
        [tuple(case.values()) for case in cases_grid],
    )
    def test_translation_to_en_returns_expected_str(
        text, src_language, expected_text, path_to_resources
    ):
        abo_formatting = ABOFormatting(path_to_resources)
        returned_text = abo_formatting.translation_to_en(text, src_language)
        np.testing.assert_equal(returned_text, expected_text)
