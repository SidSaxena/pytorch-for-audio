import torch 
import torchaudio 

from cnn import CNNetwork
from urbansounddataset import UrbanSoundDataset
from train_usd import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, input, target, class_mapping):
    model.eval()
    model.train()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.1, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected 


if __name__ == '__main__':

    # load back the model
    cnn = CNNetwork()
    state_dict = torch.load('cnnet.pth', map_location=torch.device('cpu'))
    cnn.load_state_dict(state_dict)

    # load UrbanSoundDataset validation dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                                                            sample_rate = SAMPLE_RATE, 
                                                            n_fft = 1024,
                                                            hop_length = 512,
                                                            n_mels = 64)

    # ms = mel_spectrogram(signal)

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            'cpu')


    # get a sample from the UrbanSoundDataset for inference
    input, target = usd[0][0], usd[0][1] # [batch_size, num_channels, freq, time]
    input.unsqueeze_(0)


    # make an inference
    predicted, expected = predict(cnn, input, target, class_mapping)

    print(f'Predicted: {predicted}, Expected: {expected}')