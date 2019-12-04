

def test_func(minimal_test_esM):
    # modify full load hour limit
    esM = minimal_test_esM


    assert optimization_without_full_load_hour == False

    # add full load hour constraint
    assert optimization_without_full_load_hour == True



