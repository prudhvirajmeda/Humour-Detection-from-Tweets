#AUTHOR: BRETT HOLDEN

#!/bin/bash
sudo apt-get install python-pip
pip install NLTK
pip install numpy
pip install scikit-learn
pip install scipy
python -c "import nltk
nltk.download('cmudict')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')"