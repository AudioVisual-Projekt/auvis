import unittest
from team_c.src.models.segmentation_experiment import SegmentationConfig


class TestExperimentConfig(unittest.TestCase):
    """
    Unit tests for the SegmentationConfig dataclass.
    Focuses on hash generation logic and parameter handling.
    """
    def test_hash_ignores_dataset(self):
        """
        Verifies that the generated hash remains identical even if the
        'dataset' parameter changes.
        The hash should only depend on hyperparameters, not the data source.
        """
        # 1. Setup - Create two configs that differ ONLY in the dataset
        conf_dev = SegmentationConfig(
            dataset="dev",
            onset=1.0, offset=0.8,
            min_duration_on=1.0, min_duration_off=0.5,
            max_chunk_size=10, min_chunk_size=1
        )

        conf_train = SegmentationConfig(
            dataset="train",  # <-- Difference here
            onset=1.0, offset=0.8,
            min_duration_on=1.0, min_duration_off=0.5,
            max_chunk_size=10, min_chunk_size=1
        )

        # 2. Action - Calculate hashes
        hash_dev = conf_dev.get_hash()
        hash_train = conf_train.get_hash()

        # 3. Assert - Check for equality
        self.assertEqual(
            hash_dev,
            hash_train,
            "The hash should be identical across different datasets if parameters are the same."
        )

    def test_hash_changes_on_parameter_change(self):
        """
        Verifies that changing a hyperparameter (e.g., onset) actually
        results in a different hash.
        """
        # Base configuration
        conf_a = SegmentationConfig(
            dataset="dev", onset=1.0, offset=0.8,
            min_duration_on=1.0, min_duration_off=0.5, max_chunk_size=10,
            min_chunk_size=1
        )

        # Modified configuration (onset changed)
        conf_b = SegmentationConfig(
            dataset="dev", onset=0.9,  # <-- Parameter changed
            offset=0.8, min_duration_on=1.0, min_duration_off=0.5,
            max_chunk_size=10, min_chunk_size=1
        )

        self.assertNotEqual(
            conf_a.get_hash(),
            conf_b.get_hash(),
            "Changing a hyperparameter must result in a different hash."
        )


if __name__ == '__main__':
    unittest.main()
