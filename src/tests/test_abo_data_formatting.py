from src.abo_data_formatting import ABOFormatting
import pandas as pd


class TestReadMetadata:
    @staticmethod
    def test_read_metadata_makes_expected_changes_to_metadata_df(path_to_resources):
        abo_formatting = ABOFormatting(path_to_resources)
        abo_formatting.read_metadata()
        expected_metadata_df = pd.read_csv(
            path_to_resources / "expected_read_metadata_for_listings_0.csv", index_col=0
        )
        assert expected_metadata_df.equals(abo_formatting.metadata_df)
        assert (
            abo_formatting.metadata_df.columns == expected_metadata_df.columns
        ).all()


# class TranslationToEn:
#    @staticmethod
#    def test_t
