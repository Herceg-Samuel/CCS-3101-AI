from modules.data_loader import CSVTrainingDataLoader


def test_csv_loader_creates_training_rows_from_all_csv_files():
    loader = CSVTrainingDataLoader()
    df = loader.load_training_frame()

    assert not df.empty
    assert 'label' in df.columns
    assert 'fever' in df.columns
    assert df['label'].notna().any()
