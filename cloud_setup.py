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
# bashCommand = "cwm --rdf test.rdf --ntriples > test.nt"
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()

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

import requests


# python script to download a file from a google drive public link


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


if __name__ == "__main__":
    import sys

    if len(sys.argv) is not 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        # TAKE ID FROM SHAREABLE LINK
        file_id = sys.argv[1]
        # DESTINATION FILE ON YOUR DISK
        destination = sys.argv[2]
        download_file_from_google_drive(file_id, destination)
