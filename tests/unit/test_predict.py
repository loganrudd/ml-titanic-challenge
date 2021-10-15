import unittest
from src.lgbm.predict_lgbm import predict


class TestPredict(unittest.TestCase):
    def test_prediction_is_one_or_zero(self):
        test_data = [[22.0, 1, 0, 7.2500, 0, 0, 1, 0, 0, 0, 1]]
        assert predict(test_data)[1] in ('1', '0')


if __name__ == '__main__':
    unittest.main()
