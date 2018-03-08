import subprocess
# how to setup the environment for cloud computing (install python tools and libraries, download database from
# google drive public link and run python file)

"""
sudo apt update
sudo apt install python python-dev python3 python3-dev
sudo apt-get install python3-setuptools
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo pip install --upgrade virtualenv
sudo pip install virtualenvwrapper
echo "export WORKON_HOME=$HOME/.virtualenvs" >> .bashrc
echo "export PROJECT_HOME=$HOME/Devel" >> .bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> .bashrc
echo "source "/usr/bin/virtualenvwrapper.sh"" >> .bashrc
echo "export WORKON_HOME="/opt/virtual_env/"" >> .bashrc
source `which virtualenvwrapper.sh`
mkvirtualenv -p /usr/bin/python3.5 ml1
sudo pip install pandas
sudo pip install requests
sudo pip install dotenv
sudo pip install 
git clone https://github.com/raph-m/safe_driver_prediction
cd safe_driver_prediction/proj2
python gdrive.py 1EQ0zE_2WLQdNIepWUjroPyGmi-dvN5KK ../../data.zip
cd ..
cd ..
sudo apt-get install unzip
unzip data.zip
cd safe_driver_prediction
git pull origin master
echo "ENV_NAME=vm" > .env
python proj2/feature_engineering.py train ../../churn/ 3000000
"""

# une version qui marche (sans virtualenv):
"""
sudo apt update
sudo apt install python python-dev python3 python3-dev
sudo apt-get install python3-setuptools
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
alias python=python3
sudo apt-get python3-setuptools
sudo easy_install3 pip
sudo pip3 install pandas
sudo pip3 install requests
sudo pip3 install dotenv
git clone https://github.com/raph-m/safe_driver_prediction
cd safe_driver_prediction/proj2
python gdrive.py 1EQ0zE_2WLQdNIepWUjroPyGmi-dvN5KK ../../data.zip
cd ..
cd ..
sudo apt-get install unzip
unzip data.zip
cd safe_driver_prediction
echo "ENV_NAME=vm" > .env
cd proj2
python feature_engineering.py
"""

# une autre façon de faire c'est avec `alias python=python3`

# pour automatiser ces commandes, il faudrait mettre les commandes dans ce bashCommand et lancer ce script:
bashCommand = "cwm --rdf test.rdf --ntriples > test.nt"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

"""
git clone https://github.com/raph-m/link-prediction
cd link-prediction/
# get and API token from kaggle (kaggle.json)
sudo pip install kaggle
mv kaggle.json .kaggle/
mkdir data
cd data
kaggle competitions download -c link-prediction-challenge-tm-and-nlp
sudo pip install nltk
sudo pip install tqdm

"""
