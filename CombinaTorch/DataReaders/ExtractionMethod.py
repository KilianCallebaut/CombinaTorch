import math
import statistics
from abc import abstractmethod, ABC
from typing import List

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from python_speech_features import logfbank, mfcc, fbank
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ExtractionMethod(ABC):

    def __init__(self,
                 name: str = '',
                 extraction_params=None,
                 preparation_params=None
                 ):
        super().__init__()
        if extraction_params is None:
            extraction_params = dict()
        if preparation_params is None:
            preparation_params = dict()
        self.name = name
        self.scalers = {}
        self.prep_calcs = {}

        self.extraction_params = extraction_params
        self.preparation_params = preparation_params

    @abstractmethod
    def extract_features(self, sig_samplerate) -> torch.tensor:
        pass

    @abstractmethod
    def partial_scale_fit(self, input_tensor):
        pass

    @abstractmethod
    def scale_transform(self, input_tensor) -> torch.tensor:
        pass

    @abstractmethod
    def inverse_scale_transform(self, input_tensor) -> torch.tensor:
        pass

    @abstractmethod
    def prepare_fit(self, input_tensor: torch.tensor):
        pass

    @abstractmethod
    def prepare_input(self, input_tensor: torch.tensor) -> List[torch.tensor]:
        pass

    # EXTRACTION
    @abstractmethod
    def scale_reset(self):
        pass

    @abstractmethod
    def prepare_reset(self):
        pass


class NeutralExtractionMethod(ExtractionMethod):

    def extract_features(self, sig_samplerate) -> torch.tensor:
        pass

    def partial_scale_fit(self, input_tensor):
        pass

    def scale_transform(self, input_tensor) -> torch.tensor:
        pass

    def inverse_scale_transform(self, input_tensor) -> torch.tensor:
        pass

    def prepare_fit(self, input_tensor: torch.tensor):
        pass

    def prepare_input(self, input_tensor: torch.tensor) -> List[torch.tensor]:
        pass

    def scale_reset(self):
        for s in self.scalers:
            self.scalers[s]._reset()

    def prepare_reset(self):
        self.prep_calcs = dict()


class BaseExtractionMethod(ExtractionMethod):

    def __init__(self, extraction_method: ExtractionMethod, **kwargs):
        self.extraction_method = extraction_method
        super().__init__(**kwargs)

    def extract_features(self, sig_samplerate) -> torch.tensor:
        return self.extraction_method.extract_features(sig_samplerate)

    def partial_scale_fit(self, input_tensor):
        self.extraction_method.partial_scale_fit(input_tensor)

    def scale_transform(self, input_tensor) -> torch.tensor:
        return self.extraction_method.scale_transform(input_tensor)

    def inverse_scale_transform(self, input_tensor) -> torch.tensor:
        return self.extraction_method.inverse_scale_transform(input_tensor)

    def prepare_fit(self, input_tensor: torch.tensor):
        self.extraction_method.prepare_fit(input_tensor)

    def prepare_input(self, input_tensor: torch.tensor) -> List[torch.tensor]:
        return self.extraction_method.prepare_input(input_tensor)

    def scale_reset(self):
        self.extraction_method.scale_reset()

    def prepare_reset(self):
        self.extraction_method.prepare_reset()

    @property
    def scalers(self):
        return self.extraction_method.scalers

    @scalers.setter
    def scalers(self, value):
        self.extraction_method.scalers = value

    @property
    def prep_calcs(self):
        return self.extraction_method.prep_calcs

    @prep_calcs.setter
    def prep_calcs(self, value):
        self.extraction_method.prep_calcs = value

    @property
    def extraction_params(self):
        return self.extraction_method.extraction_params

    @extraction_params.setter
    def extraction_params(self, value):
        self.extraction_method.extraction_params = value

    @property
    def preparation_params(self):
        return self.extraction_method.preparation_params

    @preparation_params.setter
    def preparation_params(self, value):
        self.extraction_method.preparation_params = value

    @property
    def name(self):
        return self.extraction_method.name

    @name.setter
    def name(self, value):
        self.extraction_method.name = value


#### EXTRACTION METHODS #####

class LogbankExtraction(BaseExtractionMethod):

    def extract_features(self, sig_samplerate) -> torch.tensor:
        return self.logbank(sig_samplerate, **self.extraction_params)

    def logbank(self, sig_samplerate, winlen=0.03, winstep=0.01, nfilt=24, nfft=512, lowfreq=0, highfreq=None,
                preemph=0.97):
        return torch.transpose(
            torch.tensor(logfbank(sig_samplerate[0], sig_samplerate[1], winlen=winlen, winstep=winstep,
                                  nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq,
                                  preemph=preemph)), 0, 1)


class LogbankSummaryExtraction(LogbankExtraction):

    def extract_features(self, sig_samplerate) -> torch.tensor:
        return self.logbank_summary(super().extract_features(sig_samplerate))

    def logbank_summary(self, lb):
        return torch.stack([torch.min(lb, 1).values, torch.max(lb, 1).values, torch.std(lb, 1), torch.mean(lb, 1),
                            torch.median(lb, 1).values, torch.tensor(np.percentile(lb, 25, axis=1)),
                            torch.tensor(np.percentile(lb, 75, axis=1))])


class MfccExtraction(BaseExtractionMethod):

    def extract_features(self, sig_samplerate):
        return self.mfcc(sig_samplerate, **self.extraction_params)

    def mfcc(self, sig_samplerate, winlen=0.03, winstep=0.01, nfilt=24, nfft=512, lowfreq=0, highfreq=None,
             preemph=0, numcep=13, ceplifter=0, appendEnergy=False, winfunc=lambda x: np.ones((x,))):
        # ( NUMFRAMES, NUMCEP)
        return torch.tensor(
            mfcc(sig_samplerate[0], sig_samplerate[1], winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft,
                 lowfreq=lowfreq, highfreq=highfreq, preemph=preemph, numcep=numcep, ceplifter=ceplifter,
                 appendEnergy=appendEnergy, winfunc=winfunc))


class FilterBankExtraction(BaseExtractionMethod):

    def extract_features(self, sig_samplerate):
        return torch.tensor(fbank(sig_samplerate[0], sig_samplerate[1],
                                  **self.extraction_params)[0])


class MelSpectrogramExtraction(BaseExtractionMethod):

    def extract_features(self, sig_samplerate):
        feature = librosa.feature.melspectrogram(y=sig_samplerate[0], sr=sig_samplerate[1], **self.extraction_params)
        feature = librosa.power_to_db(feature, ref=np.max)
        feature = librosa.util.normalize(feature)
        # plot_feature(feature, 44100 )
        return torch.tensor(feature).T


#### SCALING METHODS #####

class PerDimensionStandardizing(BaseExtractionMethod):

    def partial_scale_fit(self, input_tensor):
        """
        Adds the current input to the scaling calculation
        :param input_tensor:
        :return:
        """
        if not self.scalers:
            self.scalers[0] = StandardScaler()
        self.scalers[0].partial_fit(input_tensor)

    def scale_transform(self, input_tensor) -> torch.tensor:
        """
        Scales an input for 2D matrix
        :param input_tensor: ndarray
        :return: transformed input
        """
        return torch.tensor(self.scalers[0].transform(input_tensor))

    def inverse_scale_transform(self, input_tensor) -> torch.tensor:
        """
        Undoes a performed scaling
        :param input_tensor: ndarray
        :return: original input
        """
        return torch.tensor(self.scalers[0].inverse_transform(input_tensor))


class PerCelStandardizing(BaseExtractionMethod):

    def partial_scale_fit(self, input_tensor):
        """
        Sets the scalers based on the input per row
        :param input_tensor: ndarray
        """
        if not self.scalers:
            for i in range(input_tensor.shape[0]):
                self.scalers[i] = StandardScaler()
        for i in range(input_tensor.shape[0]):
            self.scalers[i].partial_fit(input_tensor[i, :].reshape(1, -1))

    def scale_transform(self, input_tensor: torch.tensor) -> torch.tensor:
        """
        Scales an input for the first axis in a tensor
        :param input_tensor:
        :return: transformed input
        """
        device = input_tensor.device
        for i in range(input_tensor.shape[0]):
            input_tensor[i, :] = torch.tensor(self.scalers[i].transform(input_tensor[i, :].reshape(1, -1)))
        return input_tensor

    def inverse_scale_transform(self, input_tensor: torch.tensor) -> torch.tensor:
        """
        Undoes a performed scaling
        :param input_tensor:
        :return: original input
        """
        for i in range(input_tensor.shape[0]):
            input_tensor[i, :] = torch.tensor(self.scalers[i].inverse_transform(input_tensor[i, :].reshape(1, -1)))
        return input_tensor


class PerDimensionNormalizing(BaseExtractionMethod):

    def partial_scale_fit(self, input_tensor):
        """
        Adds the current input to the scaling calculation
        :param input_tensor:
        :return:
        """
        if not self.scalers:
            self.scalers[0] = MinMaxScaler()
        self.scalers[0].partial_fit(input_tensor)

    def scale_transform(self, input_tensor) -> torch.tensor:
        """
        Scales an input for 2D matrix
        :param input_tensor: ndarray
        :return: transformed input
        """
        return torch.tensor(self.scalers[0].transform(input_tensor))

    def inverse_scale_transform(self, input_tensor) -> torch.tensor:
        """
        Undoes a performed scaling
        :param input_tensor: ndarray
        :return: original input
        """
        return torch.tensor(self.scalers[0].inverse_transform(input_tensor))


#### PREPARATION METHODS #####

class FramePreparation(BaseExtractionMethod):

    def prepare_input(self, input_tensor: torch.tensor) -> List[torch.tensor]:
        assert 'window_size' in self.preparation_params, 'Framing transforms require window_size parameter'
        window_size = int(self.preparation_params['window_size'])
        if window_size > len(input_tensor):
            assert 'min_value' in self.preparation_params, 'Framing transforms require min_value parameter if padding needs to occur'
        if window_size > len(input_tensor):
            frame = torch.vstack(
                [input_tensor, torch.ones(
                    (window_size - input_tensor.shape[0], input_tensor.shape[1])) * self.preparation_params[
                     'min_value']])
        else:
            frame = input_tensor[0:window_size]
        return [frame]


class WindowPreparation(BaseExtractionMethod):

    def prepare_input(self, input_tensor: torch.tensor) -> List[torch.tensor]:
        assert 'window_size' in self.preparation_params, 'Window preparation requires a window_size parameter'
        # assert 'window_hop' in self.preparation_params, 'Window preparation requires a window_hop parameter'
        window_size = self.preparation_params['window_size']
        window_hop = window_size if 'window_hop' not in self.preparation_params else self.preparation_params[
            'window_hop']

        windowed_inputs = []
        start_frame = window_size

        end_frame = start_frame + window_hop * math.floor((float(input_tensor.shape[0] - start_frame) / window_hop))
        if start_frame > len(input_tensor):
            assert 'min_value' in self.preparation_params, 'Window transforms require min_value parameter if padding needs to occur'
        for frame_idx in range(start_frame, end_frame + 1, window_hop):
            window = input_tensor[frame_idx - window_size:frame_idx, :]
            assert window.shape == (window_size, input_tensor.shape[1])
            windowed_inputs.append(window)

        if start_frame > input_tensor.shape[0]:
            window = torch.vstack(
                [input_tensor,
                 torch.ones(start_frame - input_tensor.shape[0], input_tensor.shape[1]) * self.preparation_params[
                     'min_value']])
            windowed_inputs.append(window)
        elif end_frame < input_tensor.shape[0]:
            window = input_tensor[input_tensor.shape[0] - window_size:input_tensor.shape[0], :]
            windowed_inputs.append(window)
        return windowed_inputs


class SummaryPreparation(BaseExtractionMethod):

    def prepare_input(self, input_tensor: torch.tensor) -> List[torch.tensor]:
        return [self.logbank_summary(input_tensor)]

    def logbank_summary(self, lb):
        return torch.stack([torch.min(lb, 1).values, torch.max(lb, 1).values, torch.std(lb, 1), torch.mean(lb, 1),
                            torch.median(lb, 1).values, torch.tensor(np.percentile(lb, 25, axis=1)),
                            torch.tensor(np.percentile(lb, 75, axis=1))])


#### PREPARATION FITTER METHODS #####
class PreparationFitter(BaseExtractionMethod):
    def prepare_fit(self, input_tensor: torch.tensor):
        if 'min_value' not in self.preparation_params:
            self.preparation_params['min_value'] = torch.min(input_tensor).item()
        else:
            self.preparation_params['min_value'] = min(self.preparation_params['min_value'],
                                                       torch.min(input_tensor).item())


class MedianWindowSizePreparationFitter(PreparationFitter):

    def prepare_fit(self, input_tensor: torch.tensor):
        super().prepare_fit(input_tensor)
        if not 'cum_sizes' in self.prep_calcs:
            self.prep_calcs['cum_sizes'] = []
        self.prep_calcs['cum_sizes'].append(len(input_tensor))
        self.preparation_params['window_size'] = statistics.median(self.prep_calcs['cum_sizes'])


class MinWindowSizePreparationFitter(PreparationFitter):
    def prepare_fit(self, input_tensor: torch.tensor):
        super().prepare_fit(input_tensor)
        if not 'cum_sizes' in self.prep_calcs:
            self.prep_calcs['cum_sizes'] = []
        self.prep_calcs['cum_sizes'].append(len(input_tensor))
        self.preparation_params['window_size'] = min(self.prep_calcs['cum_sizes'])


class MaxWindowSizePreparationFitter(PreparationFitter):
    def prepare_fit(self, input_tensor: torch.tensor):
        super().prepare_fit(input_tensor)
        if not 'cum_sizes' in self.prep_calcs:
            self.prep_calcs['cum_sizes'] = []
        self.prep_calcs['cum_sizes'].append(len(input_tensor))
        self.preparation_params['window_size'] = max(self.prep_calcs['cum_sizes'])


############################################
# IMPLEMENTATIONS
############################################

class LogbankSummaryExtractionMethod(BaseExtractionMethod):

    def __init__(self,
                 preparation_params: dict = None,
                 extraction_params: dict = None
                 ):
        super().__init__(
            name='LogbankExtractionMethod',
            extraction_method=LogbankSummaryExtraction(PerCelStandardizing(
                NeutralExtractionMethod(preparation_params=preparation_params, extraction_params=extraction_params))),

        )


class MFCCExtractionMethod(BaseExtractionMethod):
    def __init__(self,
                 preparation_params: dict = None,
                 extraction_params: dict = None):
        super().__init__(
            name='MFCCExtractionMethod',
            extraction_method=MfccExtraction(
                PerDimensionStandardizing(FramePreparation(MedianWindowSizePreparationFitter(
                    NeutralExtractionMethod(preparation_params=preparation_params, extraction_params=extraction_params)
                ))))
        )


class FilterBankExtractionMethod(BaseExtractionMethod):
    def __init__(self, **kwargs):
        super().__init__(
            name='FilterBankExtractionMethod',
            extraction_method=FilterBankExtraction(PerDimensionStandardizing(FramePreparation(
                NeutralExtractionMethod(**kwargs)
            )))
        )


class MelSpectrogramExtractionMethod(BaseExtractionMethod):
    def __init__(self, **kwargs):
        super().__init__(
            name='MelSpectrogramExtractionMethod',
            extraction_method=MelSpectrogramExtraction(
                PerDimensionStandardizing(FramePreparation(MedianWindowSizePreparationFitter(
                    NeutralExtractionMethod()
                )))),
            **kwargs
        )


############################################

# class LogbankSummary(BaseExtractionMethod):
#
#     def __init__(self, preparation_params=None, extraction_params=None):
#         super().__init__(name='logbank_summary',
#                          preparation_params=preparation_params,
#                          extraction_params=extraction_params)
#
#     def extract_features(self, sig_samplerate):
#         lb = self.logbank(sig_samplerate, **self.extraction_params)
#         return self.logbank_summary(lb)
#
#     def logbank_summary(self, lb):
#         return torch.stack([torch.min(lb, 1).values, torch.max(lb, 1).values, torch.std(lb, 1), torch.mean(lb, 1),
#                             torch.median(lb, 1).values, torch.tensor(np.percentile(lb, 25, axis=1)),
#                             torch.tensor(np.percentile(lb, 75, axis=1))])
#
#     def partial_scale_fit(self, input_tensor):
#         self.partial_scale_fit_1st_dim(input_tensor)
#
#     def scale_transform(self, input_tensor: torch.tensor):
#         return torch.from_numpy(self.scale_transform_1st_dim(input_tensor))
#
#     def inverse_scale_transform(self, input_tensor):
#         return torch.from_numpy(self.inverse_scale_transform_1st_dim(input_tensor))
#
#     def prepare_fit(self, input_tensor: torch.tensor):
#         return input_tensor
#
#     def prepare_input(self, input_tensor):
#         return input_tensor
#
#
# class Mfcc(BaseExtractionMethod):
#
#     def __init__(self, preparation_params=None, extraction_params=None):
#         super().__init__(name='mfcc',
#                          preparation_params=preparation_params,
#                          extraction_params=extraction_params)
#
#     def extract_features(self, sig_samplerate):
#         return self.mfcc(sig_samplerate, **self.extraction_params)
#
#     # winfunc = np.hamming
#     # ceplifter = 22
#     # preemph=0.97
#     def mfcc(self, sig_samplerate, winlen=0.03, winstep=0.01, nfilt=24, nfft=512, lowfreq=0, highfreq=None,
#              preemph=0, numcep=13, ceplifter=0, appendEnergy=False, winfunc=lambda x: np.ones((x,))):
#         # ( NUMFRAMES, NUMCEP)
#         return torch.tensor(
#             mfcc(sig_samplerate[0], sig_samplerate[1], winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft,
#                  lowfreq=lowfreq, highfreq=highfreq, preemph=preemph, numcep=numcep, ceplifter=ceplifter,
#                  appendEnergy=appendEnergy, winfunc=winfunc))
#
#     def partial_scale_fit(self, input_tensor):
#         self.partial_scale_fit_2D(input_tensor)
#
#     def scale_transform(self, input_tensor):
#         return torch.from_numpy(self.scale_transform_2D(input_tensor.numpy()))
#
#     def inverse_scale_transform(self, input_tensor):
#         return torch.from_numpy(self.inverse_scale_transform_2D(input_tensor))
#
#     def prepare_fit(self, input_tensor: torch.tensor):
#         self.median_window_size_fit(input_tensor)
#
#     def prepare_input(self, input_tensor):
#         return self.frame_inputs(input_tensor, **self.preparation_params)
#
#
# class MelSpectrogram(BaseExtractionMethod):
#
#     def __init__(self, preparation_params=None, extraction_params=None):
#         super().__init__(name='MelSpectrogram',
#                          preparation_params=preparation_params,
#                          extraction_params=extraction_params)
#
#     def extract_features(self, sig_samplerate):
#         """
#         (numframes=1 + int(math.ceil((1.0*sig_len - frame_len=winlen*samplerate)/frame_step=winstep*samplerate)), nfilt)
#         for winlen=0.03, winstep=0.01, sr=44100:
#             numframes= 1 + math.ceil((sig_len - 1323.0)/441)
#         for 1 sec:
#             numframes = 1 + math.ceil((44100 - 1323)/441)
#         """
#         return torch.tensor(fbank(sig_samplerate[0], sig_samplerate[1],
#                                   **self.extraction_params)[0])
#
#     def partial_scale_fit(self, input_tensor):
#         self.partial_scale_fit_2D(input_tensor)
#
#     def scale_transform(self, input_tensor):
#         return torch.from_numpy(self.scale_transform_2D(input_tensor.numpy()))
#
#     def inverse_scale_transform(self, input_tensor):
#         return torch.from_numpy(self.inverse_scale_transform_2D(input_tensor.numpy()))
#
#     def prepare_fit(self, input_tensor: torch.tensor):
#         self.median_window_size_fit(input_tensor)
#
#     def prepare_input(self, input_tensor):
#         return self.frame_inputs(input_tensor, **self.preparation_params)
#
#
# class LibMelSpectrogram(BaseExtractionMethod):
#
#     def __init__(self, preparation_params=None, extraction_params=None):
#         super().__init__(name='LibMelSpectrogram',
#                          preparation_params=preparation_params,
#                          extraction_params=extraction_params)
#
#     def extract_features(self, sig_samplerate):
#         feature = librosa.feature.melspectrogram(y=sig_samplerate[0], sr=sig_samplerate[1], **self.extraction_params)
#         feature = librosa.power_to_db(feature, ref=np.max)
#         feature = librosa.util.normalize(feature)
#         # plot_feature(feature, 16000 )
#         return torch.tensor(feature).T
#
#     def partial_scale_fit(self, input_tensor):
#         self.partial_scale_fit_2D(input_tensor)
#
#     def scale_transform(self, input_tensor):
#         return torch.from_numpy(self.scale_transform_2D(input_tensor.numpy()))
#
#     def inverse_scale_transform(self, input_tensor):
#         return torch.from_numpy(self.inverse_scale_transform_2D(input_tensor.numpy()))
#
#     def prepare_fit(self, input_tensor: torch.tensor):
#         self.median_window_size_fit(input_tensor)
#
#     def prepare_input(self, input_tensor):
#         return self.frame_inputs(input_tensor, **self.preparation_params)
#
#
# extract_options = {
#     MelSpectrogram().name: MelSpectrogram,
#     Mfcc().name: Mfcc,
#     LogbankSummary().name: LogbankSummary,
#     LibMelSpectrogram().name: LibMelSpectrogram
# }
#

def plot_feature(S, sr):
    fig, ax = plt.subplots()
    # S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S, x_axis='time',
                                   y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Spectrogram')
    plt.show()
