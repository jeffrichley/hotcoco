import numpy as np
import unittest
from coco_utils import compute_coco_distributed, Game


class TestStringMethods(unittest.TestCase):

    def test_distributed_coco(self):
        sample_size = 16
        num_players = 4
        sample_length = 1296
        num_actors = 8

        test_data = np.random.random((sample_size, num_players, sample_length))
        cache = np.full((sample_size, num_players), np.nan)

        coco_values = compute_coco_distributed(data=test_data, num_actors=num_actors,
                                               num_players=num_players, all_cached_coco_values=cache)

        for idx, entry in enumerate(test_data):
            coco_one = coco_values[idx]
            game = Game(nplayers=num_players, payoffs=np.reshape(entry, (4, 6, 6, 6, 6)))
            game_coco_values = game.coco_values()
            self.assertTrue(np.all(coco_one == game_coco_values))


    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')
    #
    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)


if __name__ == '__main__':
    unittest.main()
