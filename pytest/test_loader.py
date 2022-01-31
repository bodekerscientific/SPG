from spg import data_loader, data_utils

def get_ds(data=None):
    if data is None:
        data = data_utils.load_data()

    return data_loader.PrecipitationDataset(data)

def test_dataset_hourly():
    data = data_utils.load_data_hourly()
    ds = get_ds(data)

    print(len(ds))


def test_get_dls():
    data = data_utils.load_data()
    tr_dl, valid_dl = data_loader.get_data_loaders(data, bs=128, num_valid=1000)
    
    tr_dl = iter(tr_dl)
    valid_dd = iter(tr_dl)

    assert next(tr_dl)[1].shape[0] == 128
    assert next(valid_dd)[0].shape[0] == 128

if __name__ == '__main__':
    test_dataset_hourly()