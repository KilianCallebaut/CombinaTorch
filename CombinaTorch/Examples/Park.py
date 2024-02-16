import os
import sys

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from CombinaTorch.DataReaders.ExtractionMethod import NeutralExtractionMethod, PerDimensionStandardizing
from CombinaTorch.DataReaders.ParkAudiosetDataset import ParkAudiosetDataset
from CombinaTorch.MultiTask.ParkClassifier import ParkClassifier
from CombinaTorch.Tasks.TrainingSetCreator import ConcatTrainingSetCreator
from CombinaTorch.Training.Training import Training

drive = r"E:/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VGGISHExtract(NeutralExtractionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()

    def extract_features(self, sig_samplerate) -> torch.tensor:
        with torch.no_grad():
            if len(sig_samplerate[0]) < sig_samplerate[1]:
                sig_samplerate = (np.pad(sig_samplerate[0], (0, sig_samplerate[1] - len(sig_samplerate[0])), 'reflect'),
                                  sig_samplerate[1])
            return self.model.forward(*sig_samplerate)[None, :].cpu()


def main(argv):
    csc = ConcatTrainingSetCreator(nr_runs=4, random_state=444,
                                   recalculate=True
                                   )
    csc.add_data_reader(ParkAudiosetDataset(mode=1))

    csc.add_signal_preprocessing(dict(resample_to=16000))
    csc.add_extraction_method(extraction_method=PerDimensionStandardizing(
        VGGISHExtract(
            name='VGGishParkEmbedded'
        )
    )
    )
    # csc.add_transformation_call('normalize_fit')
    # csc.add_transformation_call('normalize_inputs')
    csc.add_sampling({
        'None of the above': 500,
        'White noise': 500,
        'Inside, small room': 500,
        'Pink noise': 500,
        'Throat clearing': 500,
        'Inside, large room or hall': 500,
        'Speech': 0,
        'Female speech, woman speaking': 0,
        'Male speech, man speaking': 0,
        'Cough': 0,
        'Breathing': 0
    })
    generator = csc.generate_training_splits()
    train, test, _ = next(generator)
    # train.datasets[0].sample_labels({
    #     'None of the above':500,
    #     'White noise':500,
    #     'Inside, small room':500,
    #     'Pink noise':500,
    #     'Throat clearing':500,
    #     'Inside, large room or hall':500
    # })
    # train.datasets[0].sample_labels(
    #     {'Speech': 0,
    #      'Female speech, woman speaking': 0,
    #      'Male speech, man speaking': 0,
    #      'Cough': 0,
    #      'Breathing': 0
    #      })

    # for i in range(2):
    model = ParkClassifier(train.get_task_list())
    n_epochs = 1000
    results = Training.create_results(modelname=model._get_name(),
                                      task_list=train.get_task_list(),
                                      results_path=os.path.join(drive, 'Thesis_Results', 'Park'),
                                      num_epochs=n_epochs)
    Training.run_gradient_descent(model=model,
                                  concat_dataset=train,
                                  test_dataset=test,
                                  results=results,
                                  batch_size=256,
                                  num_epochs=n_epochs,
                                  optimizer=optim.Adam(model.parameters(), lr=0.001),
                                  train_loader=DataLoader(train, shuffle=True, batch_size=256))

# tensorboard --logdir E:\Thesis_Results\Park\TensorBoard

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
