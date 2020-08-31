import pytest
from glrec.train.utils import resolve_scalars_for_tpu


@pytest.mark.parametrize('parsed_yaml, num_replicas, correct_dict', [
    ({
        'val': {'v': 1, 'tpu': 'lin'},
        'list': [
            {'v': 2, 'tpu': 'lin'},
            {'v': 3, 'tpu': 'lin'},
            {'v': 3.5, 'tpu': 'lin'},
        ],
        'dict': {
            'k1': {'v': 4.0, 'tpu': 'sqrt'},
            'k2': {'v': 5.1, 'tpu': 'lin'},
            'k3': {'v': 6, 'tpu': 'lin'},
        },
        'k0': 15
    }, 4, {
        'val': 4,
        'list': [8, 12, 14.0],
        'dict': {'k1': 8.0, 'k2': 20.4, 'k3': 24},
        'k0': 15
    }),
])
def test_resolve_scalars_for_tpu(parsed_yaml, num_replicas, correct_dict):
    output_dict = resolve_scalars_for_tpu(parsed_yaml, num_replicas)

    def fine_equality_check(item1, item2):
        if not type(item1) == type(item2):
            return False
        if isinstance(item1, dict):
            if not set(item1.keys()) == set(item2.keys()):
                return False
            output = True
            for k in item1.keys():
                output = output and fine_equality_check(item1[k], item2[k])
            return output

        elif isinstance(item1, list):
            if not len(item1) == len(item2):
                return False
            output = True
            for element1, element2 in zip(item1, item2):
                output = output and fine_equality_check(element1, element2)
            return output
        else:
            return item1 == item2

    assert fine_equality_check(output_dict, correct_dict)
