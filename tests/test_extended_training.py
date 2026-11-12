"""Schedule boundaries and streaming noise-corrected validation contracts."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ReviewModel
from postprocess import corrected_latent_risk
from train import CorrectedLatentRisk, epoch_learning_rate, predict


def test_fixed_decay_preserves_first_three_epochs():
    actual = [epoch_learning_rate(.001, i, 4, 2, .5) for i in range(1, 9)]
    assert actual == [.001, .001, .001, .0005, .0005, .00025, .00025, .000125]
    assert [epoch_learning_rate(.001, i) for i in range(1, 9)] == [.001] * 8
    with pytest.raises(ValueError):
        epoch_learning_rate(.001, 1, 4, 0)


def test_streaming_risk_matches_reference_with_asymmetric_transition():
    matrix = np.array([[.75,.08,.07,.06,.04], [.04,.78,.08,.06,.04],
                       [.03,.05,.80,.08,.04], [.05,.03,.06,.81,.05],
                       [.07,.02,.04,.05,.82]])
    predictions = np.array([1.2, 4.9, 3.8, 2.2, 3.6, 4.1, 1.9, 5.0, 3.2])
    labels = np.array([1, 5, 2, 2, 4, 4, 1, 3, 5])
    expected = corrected_latent_risk(predictions, labels, matrix)
    risk = CorrectedLatentRisk(torch.tensor(matrix))
    for start in range(0, len(labels), 2):
        risk.update(torch.tensor(predictions[start:start+2]), torch.tensor(labels[start:start+2]))
    actual = risk.report()
    assert actual['n'] == len(labels)
    assert actual['estimated_latent_mse'] == pytest.approx(expected['estimated_latent_mse'], abs=1e-12)
    assert actual['standard_error'] == pytest.approx(expected['standard_error'], abs=1e-12)
    assert risk.noisy_squared_error == pytest.approx(np.sum((predictions-labels)**2))


def test_negative_corrected_risk_is_not_clipped_or_square_rooted():
    risk = CorrectedLatentRisk(.75 * torch.eye(5) + .05 * torch.ones(5, 5))
    risk.update(torch.tensor([1., 5.]), torch.tensor([1., 5.]))
    assert risk.report()['estimated_latent_mse'] < 0
    assert 'rmse' not in risk.report()


@pytest.mark.parametrize('objective', ['ce', 'forward'])
def test_prediction_reports_risk_only_for_labelled_forward_models(objective):
    config = {'objective': objective, 'backbone': 'hashcnn', 'vocab_size': 32,
              'embedding_dim': 4, 'cnn_channels': 2, 'hidden_dim': 4,
              'dropout': 0., 'dense_dim': 0, 'kernels': [3, 5]}
    model = ReviewModel(config, .75 * torch.eye(5) + .05 * torch.ones(5, 5))
    rows = [{'input_ids': torch.tensor([1, 4, 5, 2, 6, 0, 0, 0]),
             'attention_mask': torch.tensor([1, 1, 1, 1, 1, 0, 0, 0]),
             'dense': torch.empty(0), 'rating': torch.tensor(float(i)),
             'title_star': 0, 'id': i} for i in range(1, 6)]
    class Rows(list):
        @property
        def row_count(self):
            return len(self)
    result = predict(model, DataLoader(Rows(rows), batch_size=2), 'cpu', False)
    assert np.isfinite(result['rmse_observed'])
    assert ('corrected_latent_risk' in result) == (objective == 'forward')
    if objective == 'forward':
        assert np.isfinite(result['corrected_latent_risk']['estimated_latent_mse'])
        for row in rows:
            row['rating'] = torch.tensor(float('nan'))
        unlabelled = predict(model, DataLoader(Rows(rows), batch_size=2), 'cpu', False,
                             require_labels=False)
        assert 'corrected_latent_risk' not in unlabelled
