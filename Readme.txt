

Install libraries.
pip install scikit-learn matplotlib pygame tk transformers fairseq pandas tensorboardX sounddevice soundfile silero-vad

Install python packages required for ccc-wav2vec embedding model.
git clone https://github.com/Speech-Lab-IITM/data2vec-aqc
cd data2vec-aqc
pip install --editable ./

Install python packages required for ccc-wav2vec embedding model.
git clone https://github.com/Speech-Lab-IITM/torchaudio-augmentations
cd torchaudio-augmentations
pip install --editable ./

Download the ccc-wav2vec embedding model from https://asr.iitm.ac.in/models/ and save it in model directory.
mkdir model
cd model
wget https://asr.iitm.ac.in/SPRING_INX/models/foundation/SPRING_INX_ccc_wav2vec2_SSL.pt


Note: If any missing libraries or errors are found during execution, please install libraries and correct the execution errors.

Test Data structure.
