from typing import Dict, List, Sequence

import numpy as np
import pytest

from neural_lifetimes.data.datasets.sequence_dataset import SliceableDataset


class ExampleSliceableData(SliceableDataset):
    def __init__(self) -> None:
        super().__init__()
        # length = 10
        # num_var = 3
        # self.data = np.arange(0, length*num_var).reshape((length, num_var))

    def __len__(self):
        # return self.data.shape[0]
        return 10

    def _load_batch(self, s: Sequence[int]) -> Sequence[Dict[str, np.ndarray]]:
        return [{"s": np.array(s)}]  # odd construct to maintain type safety
        # data = self.data[s,:]
        # return {str(i): data[:,i] for i in range(data.shape[1])}


class IncompleteSliceableData(SliceableDataset):
    def __init__(self) -> None:
        super().__init__()
        length = 10
        num_var = 3
        self.data = np.arange(0, length * num_var).reshape((length, num_var))


class TestSliceableData:
    @staticmethod
    def _get_expected_slice_result(s: slice) -> List[int]:
        return list(range(10))[s]

    @staticmethod
    def _output_to_list(o: Sequence[Dict[str, np.ndarray]]) -> List[int]:
        return list(o[0]["s"])

    @staticmethod
    def get_dataset(well_defined: bool) -> SliceableDataset:
        if well_defined:
            return ExampleSliceableData()
        else:
            return IncompleteSliceableData()

    @staticmethod
    def get_test_slices(name):
        # slices to test negative slicing
        # if name == 'closed':
        #     return [slice(0,1), slice(-2,-1), slice(0,4), slice(-9,-3)]
        # if name == 'left-open':
        #     return [slice(None,4), slice(None,-3)]
        # if name == 'right-open':
        #     return [slice(4, None), slice(-3, None)]
        # if name == 'open':
        #     return [slice(None, None), slice(None, None, -1)]
        # if name == 'inverse':
        #     return [slice(4, 0, -1), slice(4, 0, 1), slice(5,2,2), slice(5,2,-2)]
        # if name == 'empty':
        #     return [slice(0,0), slice(-2, -2, 1), slice(0,2,-2)]

        if name == "closed":
            return [slice(0, 1), slice(0, 4)]
        if name == "left-open":
            return [slice(None, 4)]
        if name == "right-open":
            return [slice(4, None)]
        if name == "open":
            return [slice(None, None)]
        if name == "empty":
            return [slice(0, 0), slice(11, None, None)]

    def test_integer_getter(self):
        dataset = self.get_dataset(True)
        items = [0, 9, -1]
        for item in items:
            assert dataset[item] == [{"s": np.array([item])}]

    @pytest.mark.parametrize(
        "intervals", ("closed", "left-open", "right-open", "open", "empty")
    )  # TODO add 'inverse' here
    def test_slice_getter(self, intervals):
        # get correct dataset for test
        dataset = self.get_dataset(True)
        slices = self.get_test_slices(intervals)

        for s in slices:
            assert self._output_to_list(dataset[s]) == self._get_expected_slice_result(s)

    def test_seq_getter(self):
        dataset = self.get_dataset(True)
        items = [[], [0], [0, 2], [-2, -4]]
        for item in items:
            assert self._output_to_list(dataset[item]) == item

    @pytest.mark.parametrize(
        ("slice", "expected"),
        (
            (
                "A",
                "ExampleSliceableData indices must be integers, slices or iterable over integers, not str.",
            ),
            (
                {},
                "ExampleSliceableData indices must be integers, slices or iterable over integers, not dict.",
            ),
            (
                0.3,
                "ExampleSliceableData indices must be integers, slices or iterable over integers, not float.",
            ),
        ),
    )
    def test_getter_type_error(self, slice, expected):
        dataset = self.get_dataset(True)
        with pytest.raises(TypeError) as excinfo:
            dataset[slice]
        (msg,) = excinfo.value.args
        assert msg == expected

    def test_len_not_implemented_error(self):
        with pytest.raises(TypeError) as excinfo:
            self.get_dataset(False)
        (msg,) = excinfo.value.args
        assert msg.startswith("Can't instantiate abstract class IncompleteSliceableData")

    def test_get_bulk_not_implemented_error(self):
        with pytest.raises(TypeError) as excinfo:
            self.get_dataset(False)
        (msg,) = excinfo.value.args
        assert msg.startswith("Can't instantiate abstract class IncompleteSliceableData")


# d = IncompleteSliceableData()
# d._get_bulk([5])

# t = TestSliceableData()
# t.test_slice_getter('open')

# d = ExampleSliceableData()
# d[5]
